#!/usr/bin/env python3
"""
将已选出的 CALVIN 子集（仅训练集 training，含 language 窗口与 episode_*.npz）转换为 LeRobot 数据集格式，
并保留原子动作划分（参考 segment_calvin_subset_atomic.py / build_atomic_dataset.py 的输出）。不处理 validation。

输入：
- --root: 子集根目录，包含 training/ 与/或 validation/，目录下有 episode_*.npz 与 lang_annotations/auto_lang_ann.npy
- --segments: 由 scripts/segment_calvin_subset_atomic.py 生成的原子段 JSON（单文件，含各 split 的 items）

输出：
- LeRobot 数据集目录（train 使用 --repo-id / --output-root；val 使用 --val-repo-id / --val-output-root，若未指定则使用 *_val）

说明：
- 每个 language window 作为一个 episode，按该 window 下的原子段依次写入固定长度块（--atomic-block-len，默认为 10）
- 图像通道固定为 2 个：image_0 = rgb_static, image_1 = rgb_gripper（不存在则写零图并在 camera_present 中置 False）
- 状态 observation.state 为 7 维：[x,y,z, roll, pitch, yaw, gripper_norm]，来自每步的 robot_obs
- 动作 action 为相邻帧 state 的差分（角度做 wrap），与 bridge_v2_to_lerobot.py 保持一致语义
- 原子标注：逐帧写入
  - atomic.valid, atomic.segment_id, atomic.frame_in_segment, atomic.segment_start, atomic.raw_frame_index
  - 兼容 bridge 的 4 个 token 索引：
      cur_translation_idx <- position64
      cur_rotation_idx    <- posture81 // 3   （27类：旋转三轴符号组合）
      cur_gripper_idx     <- posture81 % 3   （3类：close/hold/open）
      cur_duration_idx    <- 常量 1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


@dataclass(frozen=True)
class EpisodeNaming:
    prefix: str = "episode_"
    ndigits: int = 7
    suffix: str = ".npz"

    def path(self, split_dir: Path, idx: int) -> Path:
        return split_dir / f"{self.prefix}{idx:0{self.ndigits}d}{self.suffix}"


def infer_episode_naming(split_dir: Path) -> EpisodeNaming:
    for p in sorted(split_dir.glob("episode_*")):
        name = p.name
        head = name.split(".")[0]
        prefix = head[:-7]
        digits = head[-7:]
        if digits.isdigit():
            return EpisodeNaming(prefix=prefix, ndigits=len(digits), suffix=p.suffix)
    return EpisodeNaming()


def load_npz_safe(path: Path) -> Dict[str, Any] | None:
    try:
        return dict(np.load(path.as_posix(), allow_pickle=False))
    except Exception:
        return None


def load_frame(split_dir: Path, naming: EpisodeNaming, frame_idx: int) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    返回 (rgb_static, rgb_gripper, robot_obs, rel_actions)
    - rgb_*: HxWx3 uint8，可能为 None（文件缺失或键缺失）
    - robot_obs: (>=7,) float32，若缺失返回 None
    - rel_actions: (7,) float32，CALVIN 规范化相对动作；若缺失返回 None
    """
    p = naming.path(split_dir, frame_idx)
    if not p.exists():
        return None, None, None, None
    data = load_npz_safe(p)
    if data is None:
        return None, None, None, None

    rgb_static = data.get("rgb_static", None)
    if isinstance(rgb_static, np.ndarray) and rgb_static.ndim == 3 and rgb_static.shape[-1] in (3, 6):
        if rgb_static.shape[-1] == 6:
            # tactile sometimes has 6 channels; for rgb_static it should be 3. Just guard anyway.
            rgb_static = rgb_static[..., :3]
        rgb_static = rgb_static.astype(np.uint8, copy=False)
    else:
        rgb_static = None

    rgb_gripper = data.get("rgb_gripper", None)
    if isinstance(rgb_gripper, np.ndarray) and rgb_gripper.ndim == 3 and rgb_gripper.shape[-1] in (3, 6):
        if rgb_gripper.shape[-1] == 6:
            rgb_gripper = rgb_gripper[..., :3]
        rgb_gripper = rgb_gripper.astype(np.uint8, copy=False)
    else:
        rgb_gripper = None

    robot_obs = data.get("robot_obs", None)
    if isinstance(robot_obs, np.ndarray) and robot_obs.size >= 7:
        rob = robot_obs.astype(np.float32, copy=False).reshape(-1)
    else:
        rob = None
    rel_actions = data.get("rel_actions", None)
    if isinstance(rel_actions, np.ndarray) and rel_actions.size >= 7:
        ra = rel_actions.astype(np.float32, copy=False).reshape(-1)[:7]
    else:
        ra = None
    return rgb_static, rgb_gripper, rob, ra


def build_state_from_robot_obs(rob: np.ndarray | None) -> np.ndarray:
    """
    构造 7 维状态：[x,y,z, roll, pitch, yaw, g_norm]
    若 rob 缺失，返回零向量。
    """
    if rob is None or rob.size < 7:
        return np.zeros((7,), dtype=np.float32)
    x, y, z = rob[0:3].tolist()
    if rob.size >= 6:
        roll, pitch, yaw = rob[3:6].tolist()
    else:
        roll = pitch = yaw = 0.0
    g_open = float(rob[6]) if rob.size > 6 else 0.0
    g_norm = float(np.clip(g_open / 0.04, 0.0, 1.0))
    return np.asarray([x, y, z, roll, pitch, yaw, g_norm], dtype=np.float32)


def compute_action_from_states(prev_state: np.ndarray | None, cur_state: np.ndarray) -> np.ndarray:
    """
    由相邻两个状态计算 7 维动作差分；若 prev_state 缺失则全零。
    """
    if prev_state is None or prev_state.shape != (7,):
        return np.zeros((7,), dtype=np.float32)
    act = np.zeros((7,), dtype=np.float32)
    act[:3] = cur_state[:3] - prev_state[:3]
    d_rpy = cur_state[3:6] - prev_state[3:6]
    d_rpy = np.vectorize(_wrap_angle)(d_rpy)
    act[3:6] = d_rpy.astype(np.float32)
    act[6] = cur_state[6] - prev_state[6]
    return act


def compute_rel_action_from_states(prev_state: np.ndarray | None, cur_state: np.ndarray) -> np.ndarray:
    """
    回退路径：由状态差分估计 CALVIN 相对动作（规范化尺度）。
    - 平移乘以 50 并裁剪到 [-1, 1]
    - 欧拉角乘以 20 并裁剪到 [-1, 1]
    - 夹爪用符号近似（开=+1, 合=-1, 无变化=0）
    """
    if prev_state is None or prev_state.shape != (7,):
        return np.zeros((7,), dtype=np.float32)
    dpos = (cur_state[:3] - prev_state[:3]) * 50.0
    d_rpy = cur_state[3:6] - prev_state[3:6]
    d_rpy = np.vectorize(_wrap_angle)(d_rpy) * 20.0
    dg = float(cur_state[6] - prev_state[6])
    if abs(dg) < 1e-3:
        g = 0.0
    else:
        g = 1.0 if dg > 0 else -1.0
    out = np.zeros((7,), dtype=np.float32)
    out[:3] = np.clip(dpos, -1.0, 1.0).astype(np.float32)
    out[3:6] = np.clip(d_rpy, -1.0, 1.0).astype(np.float32)
    out[6] = np.float32(g)
    return out


def decode_posture81(posture_idx: int) -> Tuple[int, int, int, int]:
    """
    返回 (sx, sy, sz, grip_idx)
    - sx/sy/sz ∈ {-1,0,+1}
    - grip_idx ∈ {0,1,2} 映射 {close, hold, open}
    """
    rot_idx0 = int(posture_idx) // 3
    grip_idx = int(posture_idx) % 3
    sx = (rot_idx0 // 9) - 1
    sy = ((rot_idx0 % 9) // 3) - 1
    sz = (rot_idx0 % 3) - 1
    return sx, sy, sz, grip_idx


def open_or_create_ds(
    repo_id: str,
    root_path: str,
    *,
    fps: int,
    image_size: int,
    image_writer_threads: int,
    image_writer_processes: int,
    sync_image_writing: bool,
    robot_type: str,
) -> LeRobotDataset:
    """
    打开或创建 LeRobotDataset；定义固定 features（2 相机 + 状态/动作 + 原子标注）。
    """
    info_json_p = os.path.join(root_path, "meta", "info.json")
    if os.path.isfile(info_json_p):
        dsx = LeRobotDataset(repo_id=repo_id, root=root_path, download_videos=False)
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

    num_cams = 2
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
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root_path,
        robot_type=robot_type,
        fps=fps,
        features=features_spec,
        image_writer_threads=int(0 if sync_image_writing else max(0, image_writer_threads)),
        image_writer_processes=int(0 if sync_image_writing else max(0, image_writer_processes)),
    )


def group_segments_by_window(items: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    将 items 按 window_index 分组，并按 segment_index 升序。
    """
    from collections import defaultdict

    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        wi = int(it.get("window_index", -1))
        if wi >= 0:
            buckets[wi].append(it)
    for wi in list(buckets.keys()):
        buckets[wi].sort(key=lambda x: int(x.get("segment_index", 0)))
    return buckets


def convert_split(
    split_dir: Path,
    items: List[Dict[str, Any]],
    *,
    output_root: str,
    repo_id: str,
    fps: int,
    image_size: int,
    atomic_block_len: int,
    image_writer_threads: int,
    image_writer_processes: int,
    sync_image_writing: bool,
    robot_type: str,
    require_text: bool,
) -> None:
    """
    将一个 split（training/validation）的所有 window 转为多个 episode。
    """
    ds = open_or_create_ds(
        repo_id=repo_id,
        root_path=output_root,
        fps=fps,
        image_size=image_size,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        sync_image_writing=sync_image_writing,
        robot_type=robot_type,
    )

    naming = infer_episode_naming(split_dir)
    windows = group_segments_by_window(items)
    num_cams = 2

    for win_idx in sorted(windows.keys()):
        segs = windows[win_idx]
        any_frames = False
        # 逐原子段写入固定长度块
        for it in segs:
            # 文本优先使用 per-window 的 'text'，若缺失可退回 'task'
            seg_task_text = str(it.get("text", "") or it.get("task", "") or "")
            if require_text and len(seg_task_text.strip()) == 0:
                continue
            meta = it.get("segment", {})  # from segment_calvin_subset_atomic summarize_segment
            t_rel_start = int(meta.get("t_start", 0))
            t_rel_end = int(meta.get("t_end", t_rel_start))
            frame_start = int(it.get("frame_start", 0))
            # 原始全局帧范围 [gs, ge)
            gs = frame_start + t_rel_start
            ge = frame_start + t_rel_end
            k_real = max(0, ge - gs)

            # 标签 position64/posture81
            lab = it.get("labels", {})
            pos64 = int(lab.get("position64", 0))
            pst81 = int(lab.get("posture81", 1))  # 1 -> (rot_idx0=0, grip=1/hold)
            # 兼容 bridge token 映射
            rot_idx0 = pst81 // 3
            _sx, _sy, _sz, grip_idx = decode_posture81(pst81)
            dur_idx = 1

            for j in range(int(atomic_block_len)):
                is_real = (j < k_real)
                cam_present = np.zeros((num_cams,), dtype=np.bool_)
                frame_imgs: Dict[str, np.ndarray] = {}
                if is_real:
                    fidx = gs + j
                    rgb_static, rgb_gripper, rob, ra = load_frame(split_dir, naming, fidx)
                    # 相机 0: rgb_static
                    key0 = "observation.images.image_0"
                    if isinstance(rgb_static, np.ndarray):
                        frame_imgs[key0] = _resize_with_pad(rgb_static, image_size, image_size)
                        cam_present[0] = True
                    else:
                        frame_imgs[key0] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                    # 相机 1: rgb_gripper
                    key1 = "observation.images.image_1"
                    if isinstance(rgb_gripper, np.ndarray):
                        frame_imgs[key1] = _resize_with_pad(rgb_gripper, image_size, image_size)
                        cam_present[1] = True
                    else:
                        frame_imgs[key1] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                    # 状态与动作
                    cur_state = build_state_from_robot_obs(rob)
                    # prev 用 fidx-1
                    prev_state = None
                    if fidx - 1 >= 0:
                        _, _, rob_prev, _ = load_frame(split_dir, naming, fidx - 1)
                        prev_state = build_state_from_robot_obs(rob_prev) if rob_prev is not None else None
                    if isinstance(ra, np.ndarray) and ra.size >= 7:
                        action_vec = ra.astype(np.float32, copy=False)[:7]
                    else:
                        action_vec = compute_rel_action_from_states(prev_state, cur_state)
                    state_vec = cur_state
                    raw_idx = int(fidx)
                else:
                    # pad：使用段末状态（若可得），动作置零，raw_idx = -1
                    # 尝试 ge-1
                    fidx_last = ge - 1
                    if fidx_last >= 0:
                        _, _, rob_last, _ = load_frame(split_dir, naming, fidx_last)
                    else:
                        rob_last = None
                    state_vec = build_state_from_robot_obs(rob_last)
                    action_vec = np.zeros((7,), dtype=np.float32)
                    raw_idx = -1
                    frame_imgs["observation.images.image_0"] = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                    frame_imgs["observation.images.image_1"] = np.zeros((image_size, image_size, 3), dtype=np.uint8)

                frame = {
                    **frame_imgs,
                    "camera_present": cam_present,
                    "observation.state": state_vec.astype(np.float32, copy=False),
                    "action": action_vec.astype(np.float32, copy=False),
                    "task": seg_task_text,
                    # atomic annotations
                    "atomic.valid": np.asarray([is_real], dtype=bool),
                    "atomic.segment_id": np.asarray([int(it.get("segment_index", 0))], dtype=np.int32),
                    "atomic.frame_in_segment": np.asarray([j], dtype=np.int32),
                    "atomic.segment_start": np.asarray([j == 0], dtype=bool),
                    "atomic.cur_translation_idx": np.asarray([pos64], dtype=np.int32),
                    "atomic.cur_rotation_idx": np.asarray([rot_idx0], dtype=np.int32),
                    "atomic.cur_gripper_idx": np.asarray([grip_idx], dtype=np.int32),
                    "atomic.cur_duration_idx": np.asarray([dur_idx], dtype=np.int32),
                    "atomic.raw_frame_index": np.asarray([raw_idx], dtype=np.int32),
                }
                ds.add_frame(frame)
                any_frames = True

        if any_frames and getattr(ds, "episode_buffer", None) and ds.episode_buffer.get("size", 0) > 0:
            try:
                ds._wait_image_writer()
            except Exception:
                pass
            ds.save_episode()
            try:
                ds.hf_dataset = ds.create_hf_dataset()
            except Exception:
                pass
            print(f"[{split_dir.name}] saved episode window={win_idx}")
        else:
            print(f"[{split_dir.name}] skipped episode window={win_idx} (no frames)")

    try:
        ds.stop_image_writer()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CALVIN subset with atomic segments to LeRobot dataset")
    p.add_argument("--root", type=str, required=True, help="Subset root containing training/ and/or validation/")
    p.add_argument("--segments", type=str, required=True, help="Path to atomic_segments.json from segment_calvin_subset_atomic.py")
    p.add_argument("--output-root", type=str, required=True, help="Output root for training dataset")
    p.add_argument("--repo-id", type=str, required=True, help="Repo id for training dataset (namespace/name)")
    p.add_argument("--val-output-root", type=str, default=None, help="Output root for validation dataset (defaults to <output-root>_val)")
    p.add_argument("--val-repo-id", type=str, default=None, help="Repo id for validation dataset (defaults to <repo-id>_val)")
    p.add_argument("--splits", nargs="*", default=["training", "validation"], help="Which splits to convert")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--atomic-block-len", type=int, default=10)
    p.add_argument("--robot-type", type=str, default="panda")
    p.add_argument("--require-text", action="store_true", default=True, help="Skip segments without text")
    # image writer performance
    p.add_argument("--image-writer-threads", type=int, default=8)
    p.add_argument("--image-writer-processes", type=int, default=8)
    p.add_argument("--sync-image-writing", action="store_true", help="Disable async image writing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    with open(args.segments, "r", encoding="utf-8") as f:
        segs = json.load(f)
    all_splits = segs.get("splits", {})

    # train
    if "training" in args.splits and (root / "training").exists():
        tr_items: List[Dict[str, Any]] = list(all_splits.get("training", {}).get("items", []))
        convert_split(
            split_dir=root / "training",
            items=tr_items,
            output_root=args.output_root,
            repo_id=args.repo_id,
            fps=int(args.fps),
            image_size=int(args.image_size),
            atomic_block_len=int(args.atomic_block_len),
            image_writer_threads=int(args.image_writer_threads),
            image_writer_processes=int(args.image_writer_processes),
            sync_image_writing=bool(args.sync_image_writing),
            robot_type=str(args.robot_type),
            require_text=bool(args.require_text),
        )

    # validation
    if "validation" in args.splits and (root / "validation").exists():
        val_items: List[Dict[str, Any]] = list(all_splits.get("validation", {}).get("items", []))
        val_root = args.val_output_root or (args.output_root.rstrip("/") + "_val")
        val_repo = args.val_repo_id or (args.repo_id + "_val")
        convert_split(
            split_dir=root / "validation",
            items=val_items,
            output_root=val_root,
            repo_id=val_repo,
            fps=int(args.fps),
            image_size=int(args.image_size),
            atomic_block_len=int(args.atomic_block_len),
            image_writer_threads=int(args.image_writer_threads),
            image_writer_processes=int(args.image_writer_processes),
            sync_image_writing=bool(args.sync_image_writing),
            robot_type=str(args.robot_type),
            require_text=bool(args.require_text),
        )

    print("Done converting CALVIN subset to LeRobot.")


if __name__ == "__main__":
    main()


