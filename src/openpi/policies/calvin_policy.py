import dataclasses
from typing import Optional

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    x = np.asarray(image)
    if np.issubdtype(x.dtype, np.floating):
        x = (255 * x).astype(np.uint8)
    # Accept CHW and convert to HWC
    if x.ndim >= 3 and x.shape[-1] not in (1, 3) and x.shape[-3] in (1, 3):
        x = np.moveaxis(x, -3, -1)
    if x.ndim == 2:
        x = x[..., None]
    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)
    return x


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    """Inputs for CALVIN-style LeRobot datasets (as produced by scripts/calvin_to_lerobot.py).
    
    - Reads:
      - Images: image.image_0 (base), image.image_1 (wrist, optional)  [AFTER repack]
      - State:  state  (7,)                                           [AFTER repack]
      - Action: actions  (7,)                                         [AFTER repack]
      - Prompt: prompt or task    (str) [optional]
      - Atomic: atomic.* keys (optional)  [atomic/valid, atomic/cur_*]
    - Produces:
      - image:      {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb(zeros)}
      - image_mask: base=True; wrist masked by use_only_base
      - state, actions, prompt (if present)
      - (optional) atomic.cur_duration_idx derived from atomic.valid
      - atomic_valid and atomic_tokens packed for the model (from per-frame atomic.* keys)
    """

    action_dim: int
    action_horizon: int
    # If True, only use base_0_rgb (mask out wrist views)
    use_only_base: bool = True
    # Compute duration from atomic.valid
    compute_duration: bool = True
    # Duration clamp range and encoding
    duration_min: int = 3
    duration_max: int = 10
    duration_compact: bool = True  # if True, store (d - duration_min) ∈ [0, duration_max-duration_min]

    def __call__(self, data: dict) -> dict:
        # 1) State/actions/prompt (AFTER repack)
        state = np.asarray(data.get("state", np.zeros((self.action_dim,), dtype=np.float32)))
        state = transforms.pad_to_dim(state, self.action_dim)

        actions = None
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = transforms.pad_to_dim(actions, self.action_dim)

        # Prompt: prefer 'prompt', fallback to 'task'
        prompt: Optional[str] = None
        if "prompt" in data and isinstance(data["prompt"], (str, bytes)):
            prompt = data["prompt"].decode("utf-8") if isinstance(data["prompt"], (bytes, bytearray)) else data["prompt"]
        elif "task" in data and isinstance(data["task"], (str, bytes)):
            prompt = data["task"].decode("utf-8") if isinstance(data["task"], (bytes, bytearray)) else data["task"]

        # 2) Images (AFTER repack)
        images_dict = data.get("image", {})
        im0 = images_dict.get("image_0")
        im1 = images_dict.get("image_1")
        base_image = _parse_image(im0) if im0 is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        wrist_image = _parse_image(im1) if im1 is not None else np.zeros_like(base_image)

        image = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": np.zeros_like(base_image) if self.use_only_base else wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.False_ if self.use_only_base else np.True_,
            "right_wrist_0_rgb": np.False_,
        }

        out = {
            "state": state,
            "image": image,
            "image_mask": image_mask,
        }
        if actions is not None:
            out["actions"] = actions
        if prompt is not None:
            out["prompt"] = prompt

        # 3) Atomic duration (optional) (reads atomic/valid AFTER repack)
        if self.compute_duration:
            v = data.get("atomic/valid", data.get("atomic.valid"))
            if v is not None:
                v = np.asarray(v)
                if v.ndim == 2 and v.shape[-1] == 1:
                    v = v[..., 0]
                try:
                    d = int(np.sum(v.astype(bool)))
                    d = int(np.clip(d, self.duration_min, self.duration_max))
                    if self.duration_compact:
                        d = int(max(0, d - self.duration_min))
                    out["atomic/cur_duration_idx"] = np.asarray(d, dtype=np.int32)
                except Exception:
                    pass

        # 4) Pack atomic tokens and valid mask for the model
        #    - valid: bool[H]
        #    - tokens: [t_idx, r_idx, g_idx, d_idx] from first frame
        v = data.get("atomic/valid", data.get("atomic.valid"))
        if v is not None:
            v = np.asarray(v)
            if v.ndim == 2 and v.shape[-1] == 1:
                v = v[..., 0]
            if v.shape[0] < self.action_horizon:
                pad = np.zeros((self.action_horizon - v.shape[0],), dtype=v.dtype)
                atomic_valid = np.concatenate([v, pad], axis=0)
            else:
                atomic_valid = v[: self.action_horizon]
            out["atomic_valid"] = atomic_valid.astype(bool)

        # Read atomic token ids (default to zeros if absent)
        t_idx = int(np.asarray(data.get("atomic/cur_translation_idx", data.get("atomic.cur_translation_idx", 0))))
        r_idx = int(np.asarray(data.get("atomic/cur_rotation_idx", data.get("atomic.cur_rotation_idx", 0))))
        g_idx = int(np.asarray(data.get("atomic/cur_gripper_idx", data.get("atomic.cur_gripper_idx", 0))))
        d_idx = int(np.asarray(data.get("atomic/cur_duration_idx", data.get("atomic.cur_duration_idx", 0))))
        out["atomic_tokens"] = np.asarray([t_idx, r_idx, g_idx, d_idx], dtype=np.int32)

        return out


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    """Outputs for CALVIN-style datasets: slice actions back to 7 dims if padded."""
    out_dim: int = 7

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        acts = np.asarray(data["actions"])
        if acts.ndim >= 1 and acts.shape[-1] >= self.out_dim:
            acts = acts[..., : self.out_dim]
        return {**data, "actions": acts}


