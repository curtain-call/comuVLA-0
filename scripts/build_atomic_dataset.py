import os
import json
import glob
import math
import argparse
import pickle
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def wrap_angle(a):
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a


def smooth(x, k=3):
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    return np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])], axis=1)


def _unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.zeros_like(vec)
    return vec / n


def _az_el_bins_from_vec(vec: np.ndarray, az_bins: int, el_bins: int) -> Tuple[int, int]:
    """Spherical binning for direction vector.
    az: (-pi, pi] split into az_bins; el uses equal-area bands via z=u_z with linspace(-1,1,el_bins+1).
    Returns (az_idx, el_idx).
    """
    u = _unit(vec)
    phi = math.atan2(float(u[1]), float(u[0]))  # (-pi, pi]
    az = int(math.floor((phi + math.pi) / (2.0 * math.pi) * az_bins))
    az = max(0, min(az, az_bins - 1))
    z = float(u[2])
    edges = np.linspace(-1.0, 1.0, el_bins + 1)
    el = int(np.searchsorted(edges, z, side="right") - 1)
    el = max(0, min(el, el_bins - 1))
    return az, el

def load_obs(traj_dir: str):
    """读取原始观测，假设 full_state[3:6] 为 RPY（roll, pitch, yaw，弧度）。"""
    with open(os.path.join(traj_dir, "obs_dict.pkl"), "rb") as f:
        x = pickle.load(f)
    fs = np.asarray(x["full_state"])  # [T, 7]  [x,y,z, roll, pitch, yaw, g]
    ts = np.asarray(x.get("time_stamp", np.arange(len(fs))), dtype=float)
    return fs, ts


def _grip_level(val: float, low: float, high: float, binary: bool = False, th: float = 0.5) -> int:
    """量化夹爪：binary=True时返回0/2（close/open）；否则三档0/1/2。"""
    if binary:
        return 2 if val > th else 0
    if val < low:
        return 0
    if val > high:
        return 2
    return 1


def detect_boundaries(fs: np.ndarray, ts: np.ndarray, args) -> List[int]:
    # fs: [T,7]
    T = fs.shape[0]
    pos = fs[:, :3]
    ypr = fs[:, 3:6]
    g = fs[:, 6]

    # smoothing
    pos_s = smooth(pos, args.smooth)
    ypr_s = smooth(ypr, args.smooth)

    # deltas
    dpos = pos_s[1:] - pos_s[:-1]
    dypr = ypr_s[1:] - ypr_s[:-1]
    dypr = np.vectorize(wrap_angle)(dypr)
    v = np.linalg.norm(dpos, axis=1)

    # translation boundaries（滞后 + 区间 + 忽略前缀步）
    trans_b = set()
    active_t = np.zeros_like(v, dtype=bool)
    for k in range(v.shape[0]):
        if k < args.trans_ignore_prefix:
            active_t[k] = False
            continue
        if k == 0:
            active_t[k] = v[k] > args.v_start
        else:
            if active_t[k - 1]:
                active_t[k] = not (v[k] < args.v_stop)
            else:
                active_t[k] = v[k] > args.v_start
    # 收集激活区间（基于位移增量索引k，对应帧边界在k与k+1之间）
    runs_t = []
    prev = False
    start_k = None
    for k in range(active_t.shape[0]):
        cur = active_t[k]
        if (not prev) and cur:  # 上跳沿
            start_k = k
        if prev and (not cur):  # 下跳沿
            if start_k is not None:
                runs_t.append((start_k, k - 1))
                start_k = None
        prev = cur
    if prev and (start_k is not None):
        runs_t.append((start_k, active_t.shape[0] - 1))
    # 将区间映射为边界；过滤过短与总位移过小的区间
    for ks, ke in runs_t:
        if (ke - ks + 1) >= getattr(args, 'min_trans_len', args.min_len):
            dp = np.sum(dpos[ks:ke + 1], axis=0)
            if np.linalg.norm(dp) > args.eps_trans:
                trans_b.add(ks)
                trans_b.add(ke + 1)

    # rotation boundaries (hysteresis + runs). Avoid spurious boundaries at start and when always-zero.
    rot_b = set()
    ang_speed = np.linalg.norm(dypr, axis=1)  # length T-1, for steps (t-1 -> t)
    # Build hysteresis active mask over deltas
    active = np.zeros_like(ang_speed, dtype=bool)
    for k in range(ang_speed.shape[0]):
        if k == 0:
            active[k] = ang_speed[k] > args.w_start
        else:
            if active[k - 1]:
                active[k] = not (ang_speed[k] < args.w_stop)
            else:
                active[k] = ang_speed[k] > args.w_start
    # Collect active runs (start,end) in delta index space
    runs = []
    prev = False
    start_k = None
    for k in range(active.shape[0]):
        cur = active[k]
        if (not prev) and cur: # 上跳沿
            start_k = k
        if prev and (not cur): # 下跳沿
            if start_k is not None:
                runs.append((start_k, k - 1))
                start_k = None
        prev = cur
    if prev and (start_k is not None):
        runs.append((start_k, active.shape[0] - 1))
    # Map runs to boundaries with min length (in steps)
    for ks, ke in runs:
        if (ke - ks + 1) >= getattr(args, 'min_rot_len', args.min_len):
            # delta index k corresponds to boundary between frames (k, k+1)
            rot_b.add(ks)   # rising edge
            rot_b.add(ke + 1)   # falling edge
    # Optional: sign/axis change boundaries within active runs
    if getattr(args, 'rot_sign_change', False):
        for t in range(2, T - 1):
            if active[t - 2] and active[t - 1]:
                if np.any(np.sign(dypr[t - 2]) != np.sign(dypr[t - 1])):
                    rot_b.add(t)

    # gripper boundaries by change runs (derivative hysteresis)
    grip_b = set()
    dg = g[1:] - g[:-1]
    active_g = np.zeros_like(dg, dtype=bool)
    for k in range(dg.shape[0]):
        if k < getattr(args, 'grip_ignore_prefix', 1):
            active_g[k] = False
            continue
        m = abs(dg[k])
        if k == 0:
            active_g[k] = m > args.g_start
        else:
            if active_g[k - 1]:
                active_g[k] = not (m < args.g_stop)
            else:
                active_g[k] = m > args.g_start
    runs_g = []
    prev = False
    start_k = None
    for k in range(active_g.shape[0]):
        cur = active_g[k]
        if (not prev) and cur:  # rising
            start_k = k
        if prev and (not cur):  # falling
            if start_k is not None:
                runs_g.append((start_k, k - 1))
                start_k = None
        prev = cur
    if prev and (start_k is not None):
        runs_g.append((start_k, active_g.shape[0] - 1))
    for ks, ke in runs_g:
        if (ke - ks + 1) >= getattr(args, 'min_grip_len', 2):
            total = float(g[ke + 1] - g[ks])
            if abs(total) > args.eps_grip:
                # 边界采用：开始取左端点ks，结束取右端点ke+1
                grip_b.add(max(0, ks))
                grip_b.add(min(T - 1, ke + 1))

    # merge boundaries
    all_b = sorted(set([0, T - 1] + list(trans_b | rot_b | grip_b)))

    # enforce min/max length
    segments = []
    s = all_b[0]
    for e in all_b[1:]:
        if e - s + 1 < args.min_len:
            continue
        while e - s + 1 > args.max_len and s < e:
            segments.append((s, s + args.max_len - 1))
            s = s + args.max_len - 1
        segments.append((s, e))
        s = e

    return segments


def _split_by_trans_threshold(pos: np.ndarray, a: int, b: int, dt: float, args) -> List[Tuple[int, int]]:
    """Within [a,b], split further if cumulative translation exceeds trans_max_m,
    also ensure min_len. Keep direction bin stable within subsegment (optional).
    Returns list of (si, sj) covering [a,b].
    """
    # feature switch
    if (not getattr(args, "enable_trans_max_split", True)) or args.trans_max_m <= 0:
        return [(a, b)]
    subs = []
    i = a
    while i < b:
        j = i
        cum = np.zeros(3, dtype=np.float64)
        # base bucket from first step direction (if available)
        base_bucket = None
        while j < b:
            step = pos[j + 1] - pos[j]
            cum = cum + step
            if base_bucket is None:
                base_bucket = _az_el_bins_from_vec(step, args.az_bins, args.el_bins)
            # optional: enforce bucket consistency
            if getattr(args, "enforce_bucket_consistency", True):
                cur_bucket = _az_el_bins_from_vec(cum, args.az_bins, args.el_bins)
                bucket_changed = (cur_bucket != base_bucket)
            else:
                bucket_changed = False
            # threshold check
            reach = (np.linalg.norm(cum) >= args.trans_max_m) and ((j - i + 1) >= args.min_len)
            if bucket_changed or reach:
                break
            j += 1
        # ensure at least min_len
        if (j - i + 1) < args.min_len and (j < b):
            # force extend to min_len if possible
            j = min(b, i + args.min_len - 1)
        subs.append((i, j))
        i = j
    # merge tail if too short
    if subs and (subs[-1][1] - subs[-1][0] + 1) < args.min_len and len(subs) >= 2:
        prev = subs[-2]
        subs[-2] = (prev[0], subs[-1][1])
        subs.pop()
    return subs


def summarize_segment(fs: np.ndarray, seg: Tuple[int, int], dt: float, args) -> Dict:
    a, b = seg
    pos = fs[:, :3]
    ypr = fs[:, 3:6]
    g = fs[:, 6]

    dpos = pos[a + 1 : b + 1] - pos[a:b]
    dypr = ypr[a + 1 : b + 1] - ypr[a:b]
    dypr = np.vectorize(wrap_angle)(dypr)

    dp_total = dpos.sum(axis=0)
    dypr_total = dypr.sum(axis=0)
    dur = (b - a) * dt

    # activity flags
    trans_active = np.linalg.norm(dp_total) > args.eps_trans
    rot_active = np.linalg.norm(dypr_total) > math.radians(args.eps_rot_deg)

    # gripper change by total delta within segment
    g_state = "hold"
    try:
        dg_total = float(g[b] - g[a])
        if abs(dg_total) > args.eps_grip:
            g_state = "open" if dg_total > 0 else "close"
    except Exception:
        pass

    # translation summary (direction + distance) with spherical bin indices
    trans = None
    if trans_active:
        dist = float(np.linalg.norm(dp_total))
        dir_vec = _unit(dp_total)
        az_idx, el_idx = _az_el_bins_from_vec(dir_vec, args.az_bins, args.el_bins)
        trans = {
            "distance_m": dist,
            "dir_vec": dir_vec.tolist(),
            "az_idx": int(az_idx),
            "el_idx": int(el_idx),
        }

    # rotation accumulation (rpy)
    rot = None
    if rot_active:
        roll, pitch, yaw = dypr_total.tolist()
        rot = {
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "yaw_rad": float(yaw),
        }

    return {
        "t_start": int(a),
        "t_end": int(b),
        "duration_s": float(dur),
        "state_start": fs[a, :7].tolist(),
        "delta_xyz": dp_total.tolist(),
        "delta_ypr": dypr_total.tolist(),
        "gripper": g_state,
        "translation": trans,
        "rotation": rot,
    }


def segment_trajectory(traj_dir: str, args) -> List[Dict]:
    fs, ts = load_obs(traj_dir)
    # infer dt (median)
    if len(ts) >= 2:
        dt = float(np.median(np.diff(ts)))
    else:
        dt = 0.1
    segs = detect_boundaries(fs, ts, args)
    # second pass: within each segment, split further by translation max threshold
    pos = fs[:, :3]
    refined = []
    for (a, b) in segs:
        subs = _split_by_trans_threshold(pos, a, b, dt, args)
        refined.extend(subs)
    out = [summarize_segment(fs, s, dt, args) for s in refined if s[1] > s[0]]
    return out


def to_text(seg: Dict) -> str:
    parts = []
    if seg["translation"] is not None:
        # 基于球面分桶的方向文本（用az/el索引简单映射为“某方向”），距离取整（cm）
        dist_cm = int(round(seg["translation"]["distance_m"] * 100))
        az = seg["translation"]["az_idx"]
        el = seg["translation"]["el_idx"]
        parts.append(f"沿球面方向(az={az}, el={el})移动{dist_cm}厘米")
    if seg["rotation"] is not None:
        roll_deg = int(round(math.degrees(seg["rotation"]["roll_rad"])) )
        pitch_deg = int(round(math.degrees(seg["rotation"]["pitch_rad"])) )
        yaw_deg = int(round(math.degrees(seg["rotation"]["yaw_rad"])) )
        rot_parts = []
        if abs(roll_deg) > 0:
            rot_parts.append(f"绕X轴{'顺时针' if roll_deg<0 else '逆时针'}旋转{abs(roll_deg)}度")
        if abs(pitch_deg) > 0:
            rot_parts.append(f"绕Y轴{'顺时针' if pitch_deg<0 else '逆时针'}旋转{abs(pitch_deg)}度")
        if abs(yaw_deg) > 0:
            rot_parts.append(f"绕Z轴{'顺时针' if yaw_deg<0 else '逆时针'}旋转{abs(yaw_deg)}度")
        if rot_parts:
            parts.append("，".join(rot_parts))
    if seg["gripper"] == "open":
        parts.append("打开夹爪")
    elif seg["gripper"] == "close":
        parts.append("闭合夹爪")
    if not parts:
        return "保持姿态"
    return "并".join(parts)


def _read_lang(traj_dir: str) -> str:
    p = os.path.join(traj_dir, "lang.txt")
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and "confidence" not in l]
        return lines[0] if lines else ""
    except Exception:
        return ""


def process_root(input_root: str, output_jsonl: str, args):
    # collect trajectories
    if os.path.exists(os.path.join(input_root, "obs_dict.pkl")):
        trajs = [input_root]
    else:
        trajs = glob.glob(os.path.join(input_root, "**/traj*"), recursive=True)

    total = 0

    if getattr(args, "per_traj_out", False):
        # write one JSONL per trajectory directory
        filename = args.per_traj_filename if getattr(args, "per_traj_filename", None) else "atomic.jsonl"
        for td in trajs:
            if not os.path.exists(os.path.join(td, "obs_dict.pkl")):
                continue
            out_path = os.path.join(td, filename)
            count = 0
            try:
                segs = segment_trajectory(td, args)
                global_text = _read_lang(td)
                # 加载状态以确定动作序列长度（A = T-1）
                fs, _ts = load_obs(td)
                T = int(fs.shape[0])
                A = max(0, T - 1)
                # 推断 action_shift（按 dated 目录名：<2021-07-23 => 1，否则 0）
                action_shift = args.action_shift
                if action_shift is None:
                    cutoff = datetime(2021, 7, 23)
                    parts = os.path.normpath(td).split(os.sep)
                    action_shift = 0
                    for comp in reversed(parts):
                        try:
                            dt = datetime.strptime(comp, "%Y-%m-%d_%H-%M-%S")
                            action_shift = 1 if dt < cutoff else 0
                            break
                        except Exception:
                            continue
                with open(out_path, "w", encoding="utf-8") as f:
                    for i, seg in enumerate(segs):
                        trans_token, rot_token, grip_token, dur_token = discretize_tokens(seg)
                        ti, ri, gi, di = tokens_to_indices(
                            trans_token, rot_token, grip_token, dur_token,
                            args.az_bins, args.el_bins
                        )
                        # 计算与原始动作序列对齐的索引（左闭右开）
                        # 约定：若存在 args.action_shift（帧t对应动作[t-shift]），
                        # 则 a_start = max(0, seg.t_start - shift), a_end = max(0, seg.t_end - shift)
                        a_start = int(max(0, seg.get("t_start", 0) - action_shift))
                        a_end = int(max(0, seg.get("t_end", 0) - 1 - action_shift))
                        # 左闭右开且夹到 [0, A]
                        a_start = min(max(0, a_start), A)
                        a_end = min(max(0, a_end), A)

                        item = {
                            "trajectory_dir": td,
                            "segment_index": i,
                            "text": to_text(seg),
                            "global_text": global_text,
                            "state_start": seg["state_start"],
                            "delta_xyz": seg["delta_xyz"],
                            "delta_ypr": seg["delta_ypr"],
                            "gripper": seg["gripper"],
                            "duration_s": seg["duration_s"],
                            "tokens": {
                                "translation": trans_token,
                                "rotation": rot_token,
                                "gripper": grip_token,
                                "duration": dur_token,
                            },
                            "tokens_idx": {
                                "translation": ti,
                                "rotation": ri,
                                "gripper": gi,
                                "duration": di,
                            },
                            "meta": {
                                "t_start": seg["t_start"],
                                "t_end": seg["t_end"],
                                "a_start": a_start,  # 左闭
                                "a_end": a_end,      # 右开
                            },
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        count += 1
                print("Wrote", count, "segments to", out_path)
                total += count
                # 可选：在每个traj目录下生成原始 vs 原子轨迹对比图
                if getattr(args, "viz_compare", False):
                    try:
                        fs, ts = load_obs(td)
                        pos = fs[:, :3]
                        atomic_poly = _reconstruct_atomic_polyline_from_segs(segs, only_translation=getattr(args, "viz_only_translation", False))
                        viz_dir = td if (not getattr(args, "viz_subdir", None)) else os.path.join(td, args.viz_subdir)
                        _plot_compare(pos, atomic_poly, viz_dir, title_suffix=f"[{os.path.basename(td)}]", show_points=getattr(args, "viz_show_points", False))
                        print("Saved compare figures to", viz_dir)
                        # 旋转与夹爪对比
                        ypr_orig = fs[:, 3:6]
                        T = fs.shape[0]
                        ypr_atomic = _reconstruct_atomic_ypr_series(segs, T, ypr0=fs[0, 3:6])
                        _plot_rotation_compare(ypr_orig, ypr_atomic, ts, viz_dir)
                        # 原始量化（支持二值/三档，与构建时一致）
                        # 连续轨迹对比：原始连续值 vs 原子动作重建连续值，再各自量化显示
                        g_cont_orig = fs[:, 6].astype(float)
                        g_cont_atomic = _reconstruct_atomic_gripper_series(segs, T, float(g_cont_orig[0]))
                        # 量化到0/1/2用于图示
                        gq_orig = np.asarray([
                            _grip_level(v, 0.33, 0.66, False, 0.5) for v in g_cont_orig
                        ], dtype=int)
                        gq_atomic = np.asarray([
                            _grip_level(v, 0.33, 0.66, False, 0.5) for v in g_cont_atomic
                        ], dtype=int)
                        _plot_gripper_compare(gq_orig, gq_atomic, ts, viz_dir)
                    except Exception as ve:
                        print("Visualize failed:", td, ve)
            except Exception as e:
                print("Failed:", td, e)
                continue
        print("Total segments written:", total)
    else:
        # write all to a single JSONL
        count = 0
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for td in trajs:
                if not os.path.exists(os.path.join(td, "obs_dict.pkl")):
                    continue
                try:
                    segs = segment_trajectory(td, args)
                    global_text = _read_lang(td)
                    fs, _ts = load_obs(td)
                    T = int(fs.shape[0])
                    A = max(0, T - 1)
                    action_shift = args.action_shift
                    if action_shift is None:
                        cutoff = datetime(2021, 7, 23)
                        parts = os.path.normpath(td).split(os.sep)
                        action_shift = 0
                        for comp in reversed(parts):
                            try:
                                dt = datetime.strptime(comp, "%Y-%m-%d_%H-%M-%S")
                                action_shift = 1 if dt < cutoff else 0
                                break
                            except Exception:
                                continue
                    for i, seg in enumerate(segs):
                        trans_token, rot_token, grip_token, dur_token = discretize_tokens(seg)
                        ti, ri, gi, di = tokens_to_indices(
                            trans_token, rot_token, grip_token, dur_token,
                            args.az_bins, args.el_bins
                        )
                        a_start = int(max(0, seg.get("t_start", 0) - action_shift))
                        a_end = int(max(0, seg.get("t_end", 0) - action_shift))
                        a_start = min(max(0, a_start), A)
                        a_end = min(max(0, a_end), A)

                        item = {
                            "trajectory_dir": td,
                            "segment_index": i,
                            "text": to_text(seg),
                            "global_text": global_text,
                            "state_start": seg["state_start"],
                            "delta_xyz": seg["delta_xyz"],
                            "delta_ypr": seg["delta_ypr"],
                            "gripper": seg["gripper"],
                            "duration_s": seg["duration_s"],
                            "tokens": {
                                "translation": trans_token,
                                "rotation": rot_token,
                                "gripper": grip_token,
                                "duration": dur_token,
                            },
                            "tokens_idx": {
                                "translation": ti,
                                "rotation": ri,
                                "gripper": gi,
                                "duration": di,
                            },
                            "meta": {
                                "t_start": seg["t_start"],
                                "t_end": seg["t_end"],
                                "a_start": a_start,
                                "a_end": a_end,
                            },
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        count += 1
                except Exception as e:
                    print("Failed:", td, e)
                    continue
        print("Wrote", count, "segments to", output_jsonl)


def build_argparser():
    p = argparse.ArgumentParser("Bridge v2 原子动作数据集构建")
    p.add_argument("input_root", type=str, help="指向包含traj*/obs_dict.pkl的根目录")
    p.add_argument("output", type=str, help="输出的JSONL文件路径")
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--v_stop", type=float, default=0.003, help="线速度停止阈值（滞后下限）")
    p.add_argument("--v_start", type=float, default=0.005, help="线速度启动阈值（滞后上限）")
    p.add_argument("--dir_change_deg", type=float, default=20.0)
    p.add_argument("--w_stop", type=float, default=0.05, help="角速度停止阈值（滞后下限）")
    p.add_argument("--w_start", type=float, default=0.08, help="角速度启动阈值（滞后上限）")
    # gripper thresholds
    # gripper change-by-derivative thresholds
    p.add_argument("--g_stop", type=float, default=0.02, help="夹爪变化停止阈值（每步幅度，滞后下限），单位与full_state[6]一致")
    p.add_argument("--g_start", type=float, default=0.035, help="夹爪变化启动阈值（每步幅度，滞后上限）")
    p.add_argument("--eps_grip", type=float, default=0.055, help="段内夹爪累计变化阈值，超过才标记为开/合")
    p.add_argument("--min_grip_len", type=int, default=2, help="最短夹爪变化段步数")
    p.add_argument("--grip_ignore_prefix", type=int, default=1, help="忽略序列前若干步（启动抖动）")
    p.add_argument("--eps_trans", type=float, default=0.01)
    p.add_argument("--eps_rot_deg", type=float, default=5.0)
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--min_trans_len", type=int, default=3, help="最短平移动作步数（在速度滞后区间内）")
    p.add_argument("--trans_ignore_prefix", type=int, default=2, help="忽略序列前若干步（启动抖动）")
    p.add_argument("--min_rot_len", type=int, default=3, help="最短旋转段步数（在角速度滞后区间内）")
    p.add_argument("--rot_sign_change", action="store_true", help="仅在旋转激活区间内将符号翻转视为边界")
    p.add_argument("--max_len", type=int, default=20)
    # spherical binning for translation
    p.add_argument("--az_bins", type=int, default=8, help="方位角分桶数")
    p.add_argument("--el_bins", type=int, default=8, help="俯仰分带数（等面积z分带）")
    p.add_argument("--trans_max_m", type=float, default=0.03, help="单个原子平移最大距离(米)，超过则在片段内二次切分")
    p.add_argument("--enable_trans_max_split", action="store_true", help="启用平移最大距离二次切分")
    p.add_argument("--enforce_bucket_consistency", action="store_true", help="二次切分时保持方向桶一致性")
    # output control
    p.add_argument("--per_traj_out", action="store_true", help="在每个traj*目录下各自写出atomic.jsonl")
    p.add_argument("--per_traj_filename", type=str, default="atomic.jsonl", help="每个traj目录下的输出文件名")
    # visualization compare
    p.add_argument("--viz_compare", action="store_true", help="为每个traj生成与原始轨迹的对比可视化")
    p.add_argument("--viz_subdir", type=str, default="", help="可选：对比图输出到traj下的子目录名（留空则输出到traj根）")
    p.add_argument("--viz_show_points", action="store_true", help="在可视化中标出所有轨迹点（散点）")
    p.add_argument("--viz_only_translation", action="store_true", help="仅使用含平移的段重建轨迹（默认包含所有段以保持连续性）")
    # action alignment meta
    p.add_argument("--action_shift", type=int, default=None, help="动作与帧的时延对齐：帧t对应动作[t-shift]；默认按日期推断")
    return p


# -----------------
# 简单离散化规则
# -----------------

_DIST_BINS_M = [0.03, 0.05]  # <=1cm: small, <=3cm: mid, else large
_ANG_BINS_DEG = [2.0, 5.0]   # <=2deg: small, <=5deg: mid, else large
_DUR_BINS_S = [0.2, 0.7]     # <=0.2s: short, <=0.7s: mid, else long


def _bin_by_threshold(val: float, bounds: List[float], labels: List[str]) -> str:
    for b, lab in zip(bounds, labels):
        if val <= b:
            return lab
    return labels[-1]


def discretize_tokens(seg: Dict) -> Tuple[str, str, str, str]:
    # translation token
    t_tok = "T_none"
    if seg["translation"] is not None:
        dist = float(seg["translation"]["distance_m"])
        lab = _bin_by_threshold(dist, _DIST_BINS_M, ["S", "M", "L"])  # small/mid/large
        az = int(seg["translation"]["az_idx"])  # 0..az_bins-1
        el = int(seg["translation"]["el_idx"])  # 0..el_bins-1
        t_tok = f"T_DIR{az}_{el}_MAG{lab}"

    # rotation token（主轴最大、按符号与角度分档）
    r_tok = "R_none"
    if seg["rotation"] is not None:
        rr = seg["rotation"]["roll_rad"]
        pr = seg["rotation"]["pitch_rad"]
        yr = seg["rotation"]["yaw_rad"]
        vals = [abs(rr), abs(pr), abs(yr)]
        idx = int(np.argmax(vals))
        axis = ["X", "Y", "Z"][idx]
        sign = [np.sign(rr), np.sign(pr), np.sign(yr)][idx]
        deg = math.degrees(vals[idx])
        lab = _bin_by_threshold(deg, _ANG_BINS_DEG, ["S", "M", "L"])  # deg bins
        r_tok = f"R_{axis}{'+' if sign>=0 else '-'}_{lab}"

    # gripper token（包含half档）：
    g_map = {"open": "G_open", "close": "G_close", "hold": "G_hold"}
    g_tok = g_map.get(seg["gripper"], "G_hold")

    # duration token
    dur = float(seg["duration_s"])
    d_tok = _bin_by_threshold(dur, _DUR_BINS_S, ["D_S", "D_M", "D_L"])  # short/mid/long

    return t_tok, r_tok, g_tok, d_tok


# -----------------
# token -> index 编码（稳定且可逆的规则）
# -----------------

def _parse_translation_token(tok: str) -> Tuple[int, int, str]:
    """从形如 T_DIR{az}_{el}_MAG{S|M|L} 提取 (az, el, mag_label)。
    若为 T_none 则抛出 ValueError 以便上层处理。
    """
    if tok == "T_none":
        raise ValueError("no translation in token")
    # 期望格式：T_DIR{az}_{el}_MAG{lab}
    try:
        head, mag = tok.split("_MAG")
        _, dir_part = head.split("T_DIR")
        az_str, el_str = dir_part.split("_")
        az = int(az_str)
        el = int(el_str)
        lab = mag
        return az, el, lab
    except Exception as e:
        raise ValueError(f"bad translation token: {tok}") from e


def _parse_rotation_token(tok: str) -> Tuple[str, str, str]:
    """从形如 R_{X|Y|Z}{+|-}_{S|M|L} 提取 (axis, sign, mag_label)。
    若为 R_none 则抛出 ValueError 以便上层处理。
    """
    if tok == "R_none":
        raise ValueError("no rotation in token")
    try:
        # 例：R_Z+_L => axis='Z', sign='+', lab='L'
        body = tok[2:]  # 去掉前缀 'R_'
        axis_sign, lab = body.split("_")
        axis = axis_sign[0]
        sign = axis_sign[1]
        return axis, sign, lab
    except Exception as e:
        raise ValueError(f"bad rotation token: {tok}") from e


def tokens_to_indices(t_tok: str, r_tok: str, g_tok: str, d_tok: str,
                      az_bins: int, el_bins: int) -> Tuple[int, int, int, int]:
    """将字符串 token 编码为稳定的整数索引。

    约定：
    - translation: 0 表示 T_none；其余按 (az, el, mag) 线性编码：
        idx = 1 + (((az * el_bins) + el) * 3 + mag_idx)
        其中 mag_idx: S->0, M->1, L->2
    - rotation: 0 表示 R_none；其余按 (axis, sign, mag) 线性编码：
        axis_idx: X->0, Y->1, Z->2；sign_idx: '+'->0, '-'->1；mag_idx 同上
        idx = 1 + ((((axis_idx * 2) + sign_idx) * 3) + mag_idx)
    - gripper: 离散三类：close->0, hold->1, open->2
    - duration: D_S->0, D_M->1, D_L->2
    """
    mag_map = {"S": 0, "M": 1, "L": 2}

    # translation
    try:
        az, el, lab = _parse_translation_token(t_tok)
        mag_idx = int(mag_map[lab])
        trans_idx = 1 + (((int(az) * int(el_bins)) + int(el)) * 3 + mag_idx)
    except ValueError:
        trans_idx = 0

    # rotation
    try:
        axis, sign, lab = _parse_rotation_token(r_tok)
        axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(axis, 0)
        sign_idx = 0 if sign == "+" else 1
        mag_idx = int(mag_map[lab])
        rot_idx = 1 + ((((axis_idx * 2) + sign_idx) * 3) + mag_idx)
    except ValueError:
        rot_idx = 0

    # gripper (three-way)
    grip_idx = {"G_close": 0, "G_hold": 1, "G_open": 2}.get(g_tok, 1)

    # duration (three-way)
    dur_idx = {"D_S": 0, "D_M": 1, "D_L": 2}.get(d_tok, 1)

    return int(trans_idx), int(rot_idx), int(grip_idx), int(dur_idx)


# -----------------
# 可视化对比（内置版）
# -----------------

def _ensure_outdir(outdir: str):
    if outdir and (not os.path.exists(outdir)):
        os.makedirs(outdir, exist_ok=True)


def _reconstruct_atomic_polyline_from_segs(segs: List[Dict], only_translation: bool = False) -> np.ndarray:
    if not segs:
        return np.zeros((0, 3), dtype=float)
    # 排序保证时间顺序
    items_sorted = sorted(segs, key=lambda s: int(s.get("t_start", 0)))
    pts = []
    for s in items_sorted:
        if only_translation and (s.get("translation") is None):
            continue
        state_start = np.asarray(s.get("state_start", [0, 0, 0, 0, 0, 0, 0]), dtype=float)
        start_p = state_start[:3]
        delta = np.asarray(s.get("delta_xyz", [0.0, 0.0, 0.0]), dtype=float)
        end_p = start_p + delta
        if len(pts) == 0:
            pts.append(start_p)
        else:
            if np.linalg.norm(pts[-1] - start_p) > 1e-9:
                pts.append(start_p)
        pts.append(end_p)
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.stack(pts, axis=0)


def _plot_compare(original_pos: np.ndarray, atomic_pos: np.ndarray, outdir: str, title_suffix: str = "", show_points: bool = False):
    outdir = outdir or ""
    _ensure_outdir(outdir)

    # 3D 视图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    if len(original_pos) > 0:
        ax.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], color='gray', alpha=0.7, label='original')
        if show_points:
            ax.scatter(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], color='gray', s=6, alpha=0.5, marker='o', label='original_pts')
    if len(atomic_pos) > 0:
        ax.plot(atomic_pos[:, 0], atomic_pos[:, 1], atomic_pos[:, 2], color='tab:blue', linewidth=2.0, label='atomic')
        if show_points:
            ax.scatter(atomic_pos[:, 0], atomic_pos[:, 1], atomic_pos[:, 2], color='tab:blue', s=10, alpha=0.7, marker='.', label='atomic_pts')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('EEF translation trajectory (3D) ' + title_suffix)
    ax.legend()
    # 设置三轴等比例范围，降低Z方向视觉误差
    try:
        xs, ys, zs = [], [], []
        if len(original_pos) > 0:
            xs.extend(original_pos[:, 0].tolist()); ys.extend(original_pos[:, 1].tolist()); zs.extend(original_pos[:, 2].tolist())
        if len(atomic_pos) > 0:
            xs.extend(atomic_pos[:, 0].tolist()); ys.extend(atomic_pos[:, 1].tolist()); zs.extend(atomic_pos[:, 2].tolist())
        if xs:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            if max_range > 0:
                x_mid = 0.5 * (x_min + x_max)
                y_mid = 0.5 * (y_min + y_max)
                z_mid = 0.5 * (z_min + z_max)
                r = 0.5 * max_range
                ax.set_xlim(x_mid - r, x_mid + r)
                ax.set_ylim(y_mid - r, y_mid + r)
                ax.set_zlim(z_mid - r, z_mid + r)
    except Exception:
        pass
    plt.tight_layout()
    path3d = os.path.join(outdir, 'compare_3d.png') if outdir else 'compare_3d.png'
    plt.savefig(path3d, dpi=200)
    plt.close(fig)

    # XY / XZ / YZ 投影视图
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    views = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]
    for ax, (i, j, name) in zip(axs, views):
        if len(original_pos) > 0:
            ax.plot(original_pos[:, i], original_pos[:, j], color='gray', alpha=0.7, label='original')
            if show_points:
                ax.scatter(original_pos[:, i], original_pos[:, j], color='gray', s=6, alpha=0.5, marker='o', label='original_pts')
        if len(atomic_pos) > 0:
            ax.plot(atomic_pos[:, i], atomic_pos[:, j], color='tab:blue', linewidth=2.0, label='atomic')
            if show_points:
                ax.scatter(atomic_pos[:, i], atomic_pos[:, j], color='tab:blue', s=10, alpha=0.7, marker='.', label='atomic_pts')
        ax.set_xlabel(['X','Y','Z'][i] + ' (m)')
        ax.set_ylabel(['X','Y','Z'][j] + ' (m)')
        ax.set_title(name + ' projection')
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
    plt.tight_layout()
    pathproj = os.path.join(outdir, 'compare_projections.png') if outdir else 'compare_projections.png'
    plt.savefig(pathproj, dpi=200)
    plt.close(fig)


def _plot_rotation_compare(ypr_orig: np.ndarray, ypr_atomic: np.ndarray, ts: np.ndarray, outdir: str):
    outdir = outdir or ""
    _ensure_outdir(outdir)
    t = np.arange(len(ypr_orig)) if ts is None or len(ts) != len(ypr_orig) else ts
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ['roll(rad)', 'pitch(rad)', 'yaw(rad)']
    for i in range(3):
        axs[i].plot(t, ypr_orig[:, i], color='gray', alpha=0.7, label='original')
        if ypr_atomic.size > 0:
            axs[i].plot(t, ypr_atomic[:, i], color='tab:orange', linewidth=1.8, label='atomic')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, linestyle='--', alpha=0.3)
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('time (s or idx)')
    plt.tight_layout()
    path = os.path.join(outdir, 'compare_rotation.png') if outdir else 'compare_rotation.png'
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _plot_gripper_compare(g_orig: np.ndarray, g_atomic: np.ndarray, ts: np.ndarray, outdir: str):
    outdir = outdir or ""
    _ensure_outdir(outdir)
    t = np.arange(len(g_orig)) if ts is None or len(ts) != len(g_orig) else ts
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.step(t, g_orig, where='post', color='gray', alpha=0.7, label='original')
    if g_atomic.size > 0:
        ax.step(t, g_atomic, where='post', color='tab:green', linewidth=1.8, label='atomic')
    ax.set_xlabel('time (s or idx)')
    ax.set_ylabel('gripper(0/1/2)')
    ax.set_title('Gripper state compare')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    path = os.path.join(outdir, 'compare_gripper.png') if outdir else 'compare_gripper.png'
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _reconstruct_atomic_ypr_series(segs: List[Dict], T: int, ypr0: np.ndarray = None) -> np.ndarray:
    """根据原子段重建ypr时间序列：将每段的总旋转量均匀分配到段内步（线性插值）。
    ypr0: 可选，作为基线的初始姿态（通常用原始ypr的第0帧），用于消除恒定偏移。
    注意：原始段时长信息在`t_start`,`t_end`内，假设均匀步进。
    """
    ypr = np.zeros((T, 3), dtype=float)
    if ypr0 is not None:
        ypr[0] = np.asarray(ypr0, dtype=float)
    for s in segs:
        a = int(s.get('t_start', 0)); b = int(s.get('t_end', a))
        if b <= a or a < 0 or b >= T:
            continue
        dypr = np.asarray(s.get('delta_ypr', [0.0, 0.0, 0.0]), dtype=float)
        length = b - a
        incr = dypr / float(length)
        for i in range(a + 1, b + 1):
            ypr[i] = ypr[i - 1] + incr
    return ypr


def _reconstruct_atomic_gripper_series(segs: List[Dict], T: int, init_val: float) -> np.ndarray:
    """根据原子段重建夹爪连续轨迹：按段内均匀插值累计变化，展示连续曲线更贴近原始。
    返回float数组，便于对比；可在可视化时阈值化为0/1/2显示。
    """
    g = np.ones((T,), dtype=float) * float(init_val)
    # 按时间排序，确保前后衔接正确
    items_sorted = sorted(segs, key=lambda s: int(s.get('t_start', 0)))
    for s in items_sorted:
        a = int(s.get('t_start', 0)); b = int(s.get('t_end', a))
        if b <= a or a < 0 or b >= T:
            continue
        # 段内累计变化方向由摘要决定
        st = s.get('gripper', 'hold')
        if st == 'hold':
            # 保持：延续当前值
            for i in range(a + 1, b + 1):
                g[i] = g[i - 1]
        else:
            # open/close：线性插值到一个目标增量（用段端点差近似）
            # 由于摘要不存连续值，这里近似为固定步幅，方向由st确定
            # 修正：在区间进入处(a)立即开始变化
            steps_cnt = (b - a + 1)
            step = (1.0 if st == 'open' else -1.0) * (0.5 / max(steps_cnt, 1))
            for i in range(a, b + 1):
                prev = g[i - 1] if i > 0 else float(init_val)
                g[i] = prev + step
    return g
if __name__ == "__main__":
    args = build_argparser().parse_args()
    process_root(args.input_root, args.output, args)

