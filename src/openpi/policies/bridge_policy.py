import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    x = np.asarray(image)
    if np.issubdtype(x.dtype, np.floating):
        x = (255 * x).astype(np.uint8)
    if x.ndim >= 3 and x.shape[-1] not in (1, 3) and x.shape[-3] in (1, 3):
        x = np.moveaxis(x, -3, -1)
    if x.ndim == 2:
        x = x[..., None]
    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)
    return x


@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    use_camera_present: bool = True

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        state = transforms.pad_to_dim(np.asarray(data["state"]), self.action_dim)

        imgs = data.get("image", {}) if isinstance(data.get("image"), dict) else {}
        im0 = imgs.get("image_0")
        im1 = imgs.get("image_1")
        im2 = imgs.get("image_2")

        def zeros_like_image(ref):
            try:
                shape = np.asarray(ref).shape
                dtype = getattr(ref, "dtype", np.uint8)
                if len(shape) == 3:
                    return np.zeros(shape, dtype=dtype)
            except Exception:
                pass
            return np.zeros((224, 224, 3), dtype=np.uint8)

        if im0 is None:
            im0 = im1 if im1 is not None else im2
        if im0 is None:
            im0 = zeros_like_image(im0)
        if im1 is None:
            im1 = zeros_like_image(im0)
        if im2 is None:
            im2 = zeros_like_image(im0)

        image = {
            "base_0_rgb": _parse_image(im0),
            "left_wrist_0_rgb": _parse_image(im1),
            "right_wrist_0_rgb": _parse_image(im2),
        }

        if self.use_camera_present and "camera_present" in data:
            cp = np.asarray(data["camera_present"]).astype(bool)
            def cp_i(i: int) -> bool:
                return bool(cp[i]) if cp.size > i else (False if mask_padding else True)
            image_mask = {
                "base_0_rgb": cp_i(0),
                "left_wrist_0_rgb": cp_i(1),
                "right_wrist_0_rgb": cp_i(2),
            }
        else:
            image_mask = {k: True for k in image}

        out = {
            "state": state,
            "image": image,
            "image_mask": image_mask,
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(np.asarray(data["actions"]), self.action_dim)
            out["actions"] = actions

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    out_dim: int = 7

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        acts = np.asarray(data["actions"])
        if acts.ndim >= 1 and acts.shape[-1] >= self.out_dim:
            acts = acts[..., : self.out_dim]
        return {**data, "actions": acts}


