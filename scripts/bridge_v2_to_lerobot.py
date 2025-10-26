import argparse
import os
import glob
import pickle
from typing import List, Tuple
from datetime import datetime

import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _wrap_angle(a: float) -> float:
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # img: HxWxC uint8
    h, w = img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(img)
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    out[top : top + new_h, left : left + new_w] = np.asarray(pil)
    return out


def _discover_cams(traj_dir: str, max_cams: int = 4) -> List[str]:
    cam_dirs: List[str] = []
    for i in range(max_cams):
        d = os.path.join(traj_dir, f"images{i}")
        cam_dirs.append(d if os.path.isdir(d) else "")
    # fallback single folder named "images" to slot 0
    if not any(cam_dirs):
        d = os.path.join(traj_dir, "images")
        if os.path.isdir(d):
            cam_dirs[0] = d
    return cam_dirs


def _load_obs(traj_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(traj_dir, "obs_dict.pkl"), "rb") as f:
        x = pickle.load(f)
    fs = np.asarray(x["full_state"], dtype=float)  # [T,7] = [x,y,z, roll, pitch, yaw, g]
    ts = np.asarray(x.get("time_stamp", np.arange(len(fs), dtype=float)), dtype=float)
    return fs, ts


def _load_atomic_segments(traj_dir: str) -> list[dict]:
    path = os.path.join(traj_dir, "atomic.jsonl")
    if not os.path.exists(path):
        return []
    segs: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    segs.append(pickle.loads(pickle.dumps(__import__('json').loads(line))))
                except Exception:
                    import json as _json
                    segs.append(_json.loads(line))
    except Exception:
        return []
    return segs


def _load_policy_out(traj_dir: str):
    p = os.path.join(traj_dir, "policy_out.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            act_list = pickle.load(f)
        if isinstance(act_list, list) and len(act_list) > 0:
            if isinstance(act_list[0], dict):
                act_list = [x.get("actions", None) for x in act_list]
        acts = np.asarray(act_list, dtype=np.float32)
        if acts.ndim == 2:
            if acts.shape[1] == 6:
                acts = np.concatenate([acts, np.zeros((acts.shape[0], 1), dtype=np.float32)], axis=1)
            return acts.astype(np.float32)
    except Exception:
        return None
    return None


def _align_actions_to_frames(actions: np.ndarray, T: int, shift: int) -> np.ndarray:
    """Align per-step actions to per-frame timeline with left-closed right-open semantics.

    For frame t, take action index (t - shift) if within [0, A), else zeros.
    This makes a[t] correspond to the transition t->t+1 when shift=0 and A=T-1.
    """
    A = int(actions.shape[0]) if actions is not None else 0
    aligned = np.zeros((T, actions.shape[1]), dtype=np.float32)
    for t in range(T):
        src = t - shift
        if 0 <= src < A:
            aligned[t] = actions[src]
    return aligned


def _infer_action_shift_from_path(traj_dir: str) -> int:
    """Infer latency shift from dated folder name like YYYY-MM-DD_HH-MM-SS.

    Prior to 2021-07-23 => shift=1, otherwise shift=0. If no date found, return 0.
    """
    cutoff = datetime(2021, 7, 23)
    parts = os.path.normpath(traj_dir).split(os.sep)
    for comp in reversed(parts):
        try:
            dt = datetime.strptime(comp, "%Y-%m-%d_%H-%M-%S")
            return 1 if dt < cutoff else 0
        except Exception:
            continue
    return 0


def _load_task_text(traj_dir: str, atomic_segments: list[dict]) -> str:
    # 1) lang.txt if exists
    lang_path = os.path.join(traj_dir, "lang.txt")
    try:
        if os.path.exists(lang_path):
            with open(lang_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and "confidence" not in s:
                        return s
    except Exception:
        pass
    # 2) first segment global_text
    try:
        if atomic_segments:
            gt = atomic_segments[0].get("global_text", "")
            if isinstance(gt, str) and gt:
                return gt
    except Exception:
        pass
    # 3) default
    return ""


def _segment_text(seg: dict) -> str:
    """Extract possible per-segment global text if present."""
    try:
        gt = seg.get("global_text", "")
        if isinstance(gt, str) and gt.strip():
            return gt.strip()
    except Exception:
        pass
    try:
        gt = seg.get("meta", {}).get("global_text", "")
        if isinstance(gt, str) and gt.strip():
            return gt.strip()
    except Exception:
        pass
    return ""


def _compute_actions_from_state(full_state: np.ndarray) -> np.ndarray:
    # full_state: [T,7] -> actions: [T,7] (delta, with angle wrap for rpy)
    T = full_state.shape[0]
    act = np.zeros_like(full_state, dtype=np.float32)
    if T <= 1:
        return act.astype(np.float32)
    # position delta
    act[1:, :3] = full_state[1:, :3] - full_state[:-1, :3]
    # rpy delta with wrapping
    d_rpy = full_state[1:, 3:6] - full_state[:-1, 3:6]
    d_rpy = np.vectorize(_wrap_angle)(d_rpy)
    act[1:, 3:6] = d_rpy
    # gripper delta (no wrap)
    act[1:, 6] = full_state[1:, 6] - full_state[:-1, 6]
    return act.astype(np.float32)


def convert_one_traj(
    traj_dir: str,
    output_root: str,
    repo_id: str,
    *,
    fps: int = 5,
    image_size: int = 256,
    robot_type: str = "widowx",
    prefer_policy_out: bool = True,
    action_shift: int | None = None,
    atomic_block_len: int = 10,
    require_text: bool = True,
    require_segment_text: bool = False,
) -> str:
    fs, ts = _load_obs(traj_dir)
    if action_shift is None:
        action_shift = _infer_action_shift_from_path(traj_dir)
    actions = None
    if prefer_policy_out:
        po = _load_policy_out(traj_dir)
        if po is not None:
            # align to frames with configurable shift
            actions = _align_actions_to_frames(po, fs.shape[0], action_shift)
    if actions is None:
        actions = _compute_actions_from_state(fs)
    T = fs.shape[0]
    cam_dirs = _discover_cams(traj_dir, max_cams=4)
    num_cams = len(cam_dirs)
    atomic_segments = _load_atomic_segments(traj_dir)
    task_text = _load_task_text(traj_dir, atomic_segments)
    if require_text and (not isinstance(task_text, str) or task_text.strip() == ""):
        print(f"[convert_one_traj] skip (no global text): {traj_dir}")
        return output_root

    # Create dataset if not exists
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type=robot_type,
        fps=fps,
        features={
            **{
                f"observation.images.image_{i}": {
                    "dtype": "image",
                    "shape": (image_size, image_size, 3),
                    "names": ["height", "width", "channels"],
                }
                for i in range(num_cams)
            },
            "camera_present": {
                "dtype": "bool",
                "shape": (num_cams,),
                "names": [f"image_{i}" for i in range(num_cams)],
            },
            "observation.state": {  # 7-dim: x,y,z, roll, pitch, yaw, gripper
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
            "action": {  # 7-dim action
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
            # atomic per-frame annotations
            "atomic.valid": {"dtype": "bool", "shape": (1,), "names": None},
            "atomic.segment_id": {"dtype": "int32", "shape": (1,), "names": None},
            "atomic.frame_in_segment": {"dtype": "int32", "shape": (1,), "names": None},
            "atomic.segment_start": {"dtype": "bool", "shape": (1,), "names": None},
            "atomic.cur_translation_idx": {"dtype": "int32", "shape": (1,), "names": None},
            "atomic.cur_rotation_idx": {"dtype": "int32", "shape": (1,), "names": None},
            "atomic.cur_gripper_idx": {"dtype": "int32", "shape": (1,), "names": None},
            "atomic.cur_duration_idx": {"dtype": "int32", "shape": (1,), "names": None},
            "atomic.raw_frame_index": {"dtype": "int32", "shape": (1,), "names": None},
        },
        image_writer_threads=2,
        image_writer_processes=0,
    )

    # Write frames in blocks per atomic segment
    seg_id = 0
    for seg in atomic_segments:
        seg_task_text = _segment_text(seg) or task_text
        if require_segment_text and (seg_task_text.strip() == ""):
            continue
        t_start = int(seg.get("meta", {}).get("t_start", seg.get("t_start", 0)))
        t_end = int(seg.get("meta", {}).get("t_end", seg.get("t_end", t_start)))
        tokens_idx = seg.get("tokens_idx", {})
        cur_ti = int(tokens_idx.get("translation", 0))
        cur_ri = int(tokens_idx.get("rotation", 0))
        cur_gi = int(tokens_idx.get("gripper", 1))
        cur_di = int(tokens_idx.get("duration", 1))

        # real frames count
        k_real = max(0, min(t_end, T) - max(0, t_start))

        for j in range(atomic_block_len):
            is_real = j < k_real and (t_start + j) < T and (t_start + j) >= 0
            cam_present = np.zeros((num_cams,), dtype=np.bool_)
            frame_imgs = {}
            if is_real:
                t = t_start + j
                for i_cam, d in enumerate(cam_dirs):
                    key = f"observation.images.image_{i_cam}"
                    if d:
                        jpg = os.path.join(d, f"im_{t}.jpg")
                        png = os.path.join(d, f"im_{t}.png")
                        path = jpg if os.path.exists(jpg) else (png if os.path.exists(png) else "")
                        if path:
                            try:
                                im = Image.open(path).convert("RGB")
                                im = np.asarray(im, dtype=np.uint8)
                                im = _resize_with_pad(im, image_size, image_size)
                                frame_imgs[key] = im
                                cam_present[i_cam] = True
                                continue
                            except Exception:
                                pass
                    frame_imgs[key] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                state_vec = fs[t].astype(np.float32)
                action_vec = actions[t]
                raw_idx = t
            else:
                for i_cam in range(num_cams):
                    key = f"observation.images.image_{i_cam}"
                    frame_imgs[key] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                state_vec = (fs[min(max(t_end - 1, 0), T - 1)].astype(np.float32) if T > 0 else np.zeros((7,), dtype=np.float32))
                action_vec = np.zeros((7,), dtype=np.float32)
                raw_idx = -1

            frame = {
                **frame_imgs,
                "camera_present": cam_present,
                "observation.state": state_vec,
                "action": action_vec,
                "task": seg_task_text,
                # atomic annotations
                "atomic.valid": np.asarray([is_real], dtype=bool),
                "atomic.segment_id": np.asarray([seg_id], dtype=np.int32),
                "atomic.frame_in_segment": np.asarray([j], dtype=np.int32),
                "atomic.segment_start": np.asarray([j == 0], dtype=bool),
                "atomic.cur_translation_idx": np.asarray([cur_ti], dtype=np.int32),
                "atomic.cur_rotation_idx": np.asarray([cur_ri], dtype=np.int32),
                "atomic.cur_gripper_idx": np.asarray([cur_gi], dtype=np.int32),
                "atomic.cur_duration_idx": np.asarray([cur_di], dtype=np.int32),
                "atomic.raw_frame_index": np.asarray([raw_idx], dtype=np.int32),
            }
            ds.add_frame(frame)

        seg_id += 1

    if getattr(ds, "episode_buffer", None) and ds.episode_buffer.get("size", 0) > 0:
        ds.save_episode()
    else:
        print(f"[convert_one_traj] no frames saved for {traj_dir} (likely filtered)")

    return os.path.join(output_root, repo_id)


def _discover_trajs(input_root: str) -> list[str]:
    """在给定根目录下递归查找包含 `obs_dict.pkl` 的 `traj*` 目录。

    与 build_atomic_dataset.py 一致：
    - 若根目录本身包含 `obs_dict.pkl`，则仅返回该目录；
    - 否则递归匹配 `**/traj*` 并筛选存在 `obs_dict.pkl` 的目录。
    """
    if os.path.exists(os.path.join(input_root, "obs_dict.pkl")):
        return [input_root]
    trajs: list[str] = []
    for td in glob.glob(os.path.join(input_root, "**/traj*"), recursive=True):
        if os.path.exists(os.path.join(td, "obs_dict.pkl")):
            trajs.append(td)
    return sorted(trajs)


def convert_root(
    input_root: str,
    output_root: str,
    repo_id: str,
    *,
    fps: int = 5,
    image_size: int = 256,
    robot_type: str = "widowx",
    prefer_policy_out: bool = True,
    action_shift: int | None = None,
    atomic_block_len: int = 10,
    # resume controls
    skip_existing_episodes: int = 0,
    max_episodes: int | None = None,
    # performance
    image_writer_threads: int = 0,
    image_writer_processes: int = 0,
    sync_image_writing: bool = False,
    require_text: bool = True,
    require_segment_text: bool = False,
    # splitting
    val_ratio: float = 0.0,
    val_output_root: str | None = None,
    val_repo_id: str | None = None,
    val_stride: int | None = None,
) -> str:
    """将根目录下所有轨迹写入同一个 LeRobot 数据集（每个轨迹为一个 episode）。"""
    trajs = _discover_trajs(input_root)
    if len(trajs) == 0:
        raise FileNotFoundError(f"No traj* with obs_dict.pkl found under: {input_root}")

    # 固定相机通道为 4 个，以兼容不同轨迹的相机配置；缺失的相机用全零填充。
    num_cams = 4

    # 定义 features 以便 train/val 共享
    features_spec = {
        **{
            f"observation.images.image_{i}": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channels"],
            }
            for i in range(num_cams)
        },
        "camera_present": {
            "dtype": "bool",
            "shape": (num_cams,),
            "names": [f"image_{i}" for i in range(num_cams)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        # atomic per-frame annotations
        "atomic.valid": {"dtype": "bool", "shape": (1,), "names": None},
        "atomic.segment_id": {"dtype": "int32", "shape": (1,), "names": None},
        "atomic.frame_in_segment": {"dtype": "int32", "shape": (1,), "names": None},
        "atomic.segment_start": {"dtype": "bool", "shape": (1,), "names": None},
        "atomic.cur_translation_idx": {"dtype": "int32", "shape": (1,), "names": None},
        "atomic.cur_rotation_idx": {"dtype": "int32", "shape": (1,), "names": None},
        "atomic.cur_gripper_idx": {"dtype": "int32", "shape": (1,), "names": None},
        "atomic.cur_duration_idx": {"dtype": "int32", "shape": (1,), "names": None},
        "atomic.raw_frame_index": {"dtype": "int32", "shape": (1,), "names": None},
    }

    def _open_or_create_ds(repo_id_local: str, root_path: str) -> LeRobotDataset:
        info_json_p = os.path.join(root_path, "meta", "info.json")
        if os.path.isfile(info_json_p):
            dsx = LeRobotDataset(repo_id=repo_id_local, root=root_path, download_videos=False)
            if not sync_image_writing and (image_writer_threads or image_writer_processes):
                dsx.start_image_writer(
                    num_processes=int(max(0, image_writer_processes)),
                    num_threads=int(max(0, image_writer_threads)),
                )
            try:
                dsx.hf_dataset = dsx.create_hf_dataset()
            except Exception:
                pass
            return dsx
        else:
            return LeRobotDataset.create(
                repo_id=repo_id_local,
                root=root_path,
                robot_type=robot_type,
                fps=fps,
                features=features_spec,
                image_writer_threads=int(0 if sync_image_writing else max(0, image_writer_threads)),
                image_writer_processes=int(0 if sync_image_writing else max(0, image_writer_processes)),
            )

    # 构建 train/val 数据集句柄
    ds_train = _open_or_create_ds(repo_id, output_root)
    ds_val = None
    val_set: set[str] = set()
    if (isinstance(val_stride, int) and val_stride and val_stride > 1) or (isinstance(val_ratio, float) and val_ratio > 0.0):
        n_total = len(trajs)
        if isinstance(val_stride, int) and val_stride and val_stride > 1:
            stride = val_stride
        else:
            # 根据比例推导均匀抽取步长（至少为 2，避免全部落入验证）
            ratio = min(0.5, max(1e-6, float(val_ratio)))
            stride = max(2, int(round(1.0 / ratio)))
        val_indices = {i for i in range(n_total) if (i % stride) == 0}
        val_set = {trajs[i] for i in val_indices}
        val_root = val_output_root or (output_root.rstrip("/") + "_val")
        val_repo = val_repo_id or (repo_id + "_val")
        ds_val = _open_or_create_ds(val_repo, val_root)

    out_path: str | None = None

    # 跳过已处理的 episode 数量（从列表起始处顺延），不改变原始遍历顺序
    start_idx = int(max(0, skip_existing_episodes))
    if start_idx > 0:
        print(f"[resume] skip first {start_idx} trajectories (by index order)")
    if isinstance(max_episodes, int) and max_episodes is not None and max_episodes > 0:
        trajs = trajs[start_idx:start_idx + max_episodes]
    else:
        trajs = trajs[start_idx:]

    for epi_idx, traj_dir in enumerate(trajs, start=start_idx):
        ds = ds_val if (ds_val is not None and traj_dir in val_set) else ds_train
        fs, ts = _load_obs(traj_dir)
        T = fs.shape[0]
        if action_shift is None:
            shift = _infer_action_shift_from_path(traj_dir)
        else:
            shift = action_shift

        actions = None
        if prefer_policy_out:
            po = _load_policy_out(traj_dir)
            if po is not None:
                actions = _align_actions_to_frames(po, T, shift)
        if actions is None:
            actions = _compute_actions_from_state(fs)

        cam_dirs = _discover_cams(traj_dir, max_cams=num_cams)  # 长度恒为 num_cams
        atomic_segments = _load_atomic_segments(traj_dir)
        task_text = _load_task_text(traj_dir, atomic_segments)
        if require_text and (not isinstance(task_text, str) or task_text.strip() == ""):
            print(f"[convert_root] skip traj (no global text): {traj_dir}")
            continue

        seg_id = 0
        any_frames = False
        for seg in atomic_segments:
            seg_task_text = _segment_text(seg) or task_text
            if require_segment_text and (seg_task_text.strip() == ""):
                continue
            t_start = int(seg.get("meta", {}).get("t_start", seg.get("t_start", 0)))
            t_end = int(seg.get("meta", {}).get("t_end", seg.get("t_end", t_start)))
            tokens_idx = seg.get("tokens_idx", {})
            cur_ti = int(tokens_idx.get("translation", 0))
            cur_ri = int(tokens_idx.get("rotation", 0))
            cur_gi = int(tokens_idx.get("gripper", 1))
            cur_di = int(tokens_idx.get("duration", 1))

            # real frames count
            k_real = max(0, min(t_end, T) - max(0, t_start))

            for j in range(atomic_block_len):
                is_real = j < k_real and (t_start + j) < T and (t_start + j) >= 0
                cam_present = np.zeros((num_cams,), dtype=np.bool_)
                frame_imgs = {}
                if is_real:
                    t = t_start + j
                    # 顺序加载相机图片（官方标准实现思路）
                    for i_cam, d in enumerate(cam_dirs):
                        key = f"observation.images.image_{i_cam}"
                        if d:
                            jpg = os.path.join(d, f"im_{t}.jpg")
                            png = os.path.join(d, f"im_{t}.png")
                            path = jpg if os.path.exists(jpg) else (png if os.path.exists(png) else "")
                            if path:
                                try:
                                    im = Image.open(path).convert("RGB")
                                    im = np.asarray(im, dtype=np.uint8)
                                    im = _resize_with_pad(im, image_size, image_size)
                                    frame_imgs[key] = im
                                    cam_present[i_cam] = True
                                    continue
                                except Exception:
                                    pass
                        frame_imgs[key] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                    state_vec = fs[t].astype(np.float32)
                    action_vec = actions[t]
                    raw_idx = t
                else:
                    for i_cam in range(num_cams):
                        key = f"observation.images.image_{i_cam}"
                        frame_imgs[key] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                    state_vec = (fs[min(max(t_end - 1, 0), T - 1)].astype(np.float32) if T > 0 else np.zeros((7,), dtype=np.float32))
                    action_vec = np.zeros((7,), dtype=np.float32)
                    raw_idx = -1

                frame = {
                    **frame_imgs,
                    "camera_present": cam_present,
                    "observation.state": state_vec,
                    "action": action_vec,
                    "task": seg_task_text,
                    # atomic annotations
                    "atomic.valid": np.asarray([is_real], dtype=bool),
                    "atomic.segment_id": np.asarray([seg_id], dtype=np.int32),
                    "atomic.frame_in_segment": np.asarray([j], dtype=np.int32),
                    "atomic.segment_start": np.asarray([j == 0], dtype=bool),
                    "atomic.cur_translation_idx": np.asarray([cur_ti], dtype=np.int32),
                    "atomic.cur_rotation_idx": np.asarray([cur_ri], dtype=np.int32),
                    "atomic.cur_gripper_idx": np.asarray([cur_gi], dtype=np.int32),
                    "atomic.cur_duration_idx": np.asarray([cur_di], dtype=np.int32),
                    "atomic.raw_frame_index": np.asarray([raw_idx], dtype=np.int32),
                }
                ds.add_frame(frame)
                any_frames = True

            seg_id += 1

        # 确保任何异步写图完成（冗余安全）
        if any_frames and getattr(ds, "episode_buffer", None) and ds.episode_buffer.get("size", 0) > 0:
            try:
                ds._wait_image_writer()
            except Exception:
                pass
            ds.save_episode()
            # 避免内存中累计 concat：保存后重置为一个空表
            try:
                ds.hf_dataset = ds.create_hf_dataset()
            except Exception:
                pass
            out_path = output_root
            split_name = "val" if (ds_val is not None and traj_dir in val_set) else "train"
            print(f"[convert_root] saved episode {epi_idx} ({split_name}) from {traj_dir}")
        else:
            print(f"[convert_root] skipped episode (no frames after filtering): {traj_dir}")

    # 结束时干净地关闭写图器，避免残留子进程/线程
    try:
        ds_train.stop_image_writer()
    except Exception:
        pass
    if ds_val is not None:
        try:
            ds_val.stop_image_writer()
        except Exception:
            pass
    return out_path if out_path is not None else output_root


def main():
    parser = argparse.ArgumentParser("Convert Bridge v2 data to a LeRobot dataset")
    parser.add_argument("traj_dir", type=str, help="Path to a traj dir (single) or a root dir (with --scan_all)")
    parser.add_argument("output_root", type=str, help="Output root directory for the LeRobot dataset")
    parser.add_argument("repo_id", type=str, help="Dataset repo_id (namespace/name) under output_root")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--robot_type", type=str, default="widowx")
    parser.add_argument("--prefer_policy_out", action="store_true")
    parser.add_argument("--action_shift", type=int, default=None, help="Latency alignment: frame t uses action[t-shift]; default inferred by date (<2021-07-23 => 1 else 0)")
    parser.add_argument("--atomic_block_len", type=int, default=10)
    parser.add_argument("--scan_all", action="store_true", help="Scan traj_dir recursively for traj* and write all to a single dataset")
    parser.add_argument("--require_text", action="store_true", default=True, help="Skip trajectories without global text")
    parser.add_argument("--require_segment_text", action="store_true", default=False, help="Skip segments without per-segment global text; when off, fall back to trajectory text")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Hold-out ratio for validation set (0.0~0.5)")
    parser.add_argument("--val_output_root", type=str, default=None, help="Output root for validation dataset")
    parser.add_argument("--val_repo_id", type=str, default=None, help="Repo id for validation dataset (default: <repo_id>_val)")
    parser.add_argument("--val_stride", type=int, default=None, help="Take every N-th trajectory as validation (overrides ratio if provided)")
    parser.add_argument("--skip_existing_episodes", type=int, default=0, help="Skip first N trajectories (by index order) for resume")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit number of trajectories to process this run")
    parser.add_argument("--image_writer_threads", type=int, default=8, help="Async image writer threads")
    parser.add_argument("--image_writer_processes", type=int, default=8, help="Async image writer processes")
    args = parser.parse_args()

    if args.scan_all:
        out = convert_root(
            input_root=args.traj_dir,
            output_root=args.output_root,
            repo_id=args.repo_id,
            fps=args.fps,
            image_size=args.image_size,
            robot_type=args.robot_type,
            prefer_policy_out=args.prefer_policy_out,
            action_shift=args.action_shift,
            atomic_block_len=args.atomic_block_len,
            skip_existing_episodes=args.skip_existing_episodes,
            max_episodes=args.max_episodes,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            require_text=args.require_text,
            require_segment_text=args.require_segment_text,
            val_ratio=args.val_ratio,
            val_output_root=args.val_output_root,
            val_repo_id=args.val_repo_id,
            val_stride=args.val_stride,
        )
    else:
        out = convert_one_traj(
            traj_dir=args.traj_dir,
            output_root=args.output_root,
            repo_id=args.repo_id,
            fps=args.fps,
            image_size=args.image_size,
            robot_type=args.robot_type,
            prefer_policy_out=args.prefer_policy_out,
            action_shift=args.action_shift,
            atomic_block_len=args.atomic_block_len,
            require_text=args.require_text,
            require_segment_text=args.require_segment_text,
        )
    print("LeRobot dataset written to:", out)


if __name__ == "__main__":
    main()


