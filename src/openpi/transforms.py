from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        def _to_numpy(x):
            try:
                return np.asarray(x)
            except Exception:
                # Handle torch.Tensor without importing torch explicitly
                try:
                    return x.detach().cpu().numpy()
                except Exception:
                    return np.asarray(x)

        def _ensure_hwc_uint8(arr):
            x = _to_numpy(arr)
            # Ensure last three dims are (H, W, C)
            if x.ndim == 2:
                x = x[..., None]  # H, W -> H, W, 1
            elif x.ndim >= 3:
                # If appears to be CHW in the last 3 dims (C,H,W) with C in {1,3}, move C to last
                if x.shape[-1] not in (1, 3) and x.shape[-3] in (1, 3):
                    x = np.moveaxis(x, -3, -1)
                # If last dim is not channel, leave as is to avoid corrupting layout
            # Convert to uint8
            if np.issubdtype(x.dtype, np.floating):
                x = image_tools.convert_to_uint8(x)
            else:
                x = x.astype(np.uint8, copy=False)
            return x

        data["image"] = {k: image_tools.resize_with_pad(_ensure_hwc_uint8(v), self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class RandomShiftsAug(DataTransformFn):
    """Random shifts augmentation with replicate padding per image key.
    
    - pad_default: default pad pixels for unspecified keys
    - pad_by_key:  overrides for specific keys, e.g.
        {"base_0_rgb": 10, "left_wrist_0_rgb": 4, "right_wrist_0_rgb": 4}
    """
    pad_default: int = 0
    pad_by_key: dict[str, int] = dataclasses.field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        imgs = data.get("image")
        if not isinstance(imgs, dict):
            return data

        def shift(img: np.ndarray, pad: int) -> np.ndarray:
            if pad is None or pad <= 0:
                return img
            x = np.asarray(img)
            if x.ndim == 2:
                x = x[..., None]
            if x.ndim != 3:
                return img
            h, w = x.shape[:2]
            # replicate padding
            y = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            dy = np.random.randint(-pad, pad + 1)
            dx = np.random.randint(-pad, pad + 1)
            top = pad + dy
            left = pad + dx
            return y[top : top + h, left : left + w, :]

        out_images = {}
        for k, v in imgs.items():
            pad = self.pad_by_key.get(k, self.pad_default)
            out_images[k] = shift(v, pad)
        return {**data, "image": out_images}


@dataclasses.dataclass(frozen=True)
class ScaleImagesToUnit(DataTransformFn):
    """Convert uint8 images to float32 in [0,1]."""

    def __call__(self, data: DataDict) -> DataDict:
        imgs = data.get("image")
        if not isinstance(imgs, dict):
            return data
        out = {}
        for k, v in imgs.items():
            x = np.asarray(v)
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            # if appears to be uint8 range, scale
            if x.max() > 1.0 or np.issubdtype(v.dtype, np.uint8):
                x = x / 255.0
            out[k] = x
        return {**data, "image": out}


@dataclasses.dataclass(frozen=True)
class NormalizeImages(DataTransformFn):
    """Channel-wise normalize images: (x - mean) / std.
    
    mean/std are lists of 3 floats (RGB) in [0,1].
    """
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __call__(self, data: DataDict) -> DataDict:
        imgs = data.get("image")
        if not isinstance(imgs, dict):
            return data
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3)
        out = {}
        for k, v in imgs.items():
            x = np.asarray(v, dtype=np.float32)
            # expect HWC, if CHW move to HWC
            if x.ndim == 3 and x.shape[-1] not in (1, 3) and x.shape[-3] in (1, 3):
                x = np.moveaxis(x, -3, -1)
            # If single channel, broadcast first channel stats
            if x.ndim == 3 and x.shape[-1] == 1:
                m = mean[..., :1]
                s = std[..., :1]
            else:
                m = mean
                s = std
            out[k] = (x - m) / (s + 1e-6)
        return {**data, "image": out}


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks, question_len = self.tokenizer.tokenize(prompt)

        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks, "question_len": question_len}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class ExtractAtomicFromSequence(DataTransformFn):
    """Extracts atomic tokens (from the first frame) and validity mask over the window.

    Expects the following keys (after repack):
      - "atomic/valid": a boolean sequence of length == action horizon (shape [H] or [H,1])
      - "atomic/cur_translation_idx": scalar int (taken from the first frame/anchor)
      - "atomic/cur_rotation_idx": scalar int
      - "atomic/cur_gripper_idx": scalar int
      - "atomic/cur_duration_idx": scalar int

    Outputs two new keys:
      - "atomic_valid": bool[H]
      - "atomic_tokens": int32[4]
    """

    action_horizon: int

    def __call__(self, data: DataDict) -> DataDict:
        # validity sequence
        valid = data.get("atomic/valid")
        if valid is None:
            # If not present, create an all-True mask to avoid breaking downstream
            atomic_valid = np.ones((self.action_horizon,), dtype=bool)
        else:
            valid = np.asarray(valid)
            # squeeze trailing singleton if present: [H,1] -> [H]
            if valid.ndim == 2 and valid.shape[-1] == 1:
                valid = valid[..., 0]
            # pad / trim to action_horizon defensively
            if valid.shape[0] < self.action_horizon:
                pad = np.zeros((self.action_horizon - valid.shape[0],), dtype=valid.dtype)
                atomic_valid = np.concatenate([valid, pad], axis=0)
            else:
                atomic_valid = valid[: self.action_horizon]

        # atomic tokens from anchor (first frame)
        t_idx = int(np.asarray(data.get("atomic/cur_translation_idx", 0)))
        r_idx = int(np.asarray(data.get("atomic/cur_rotation_idx", 0)))
        g_idx = int(np.asarray(data.get("atomic/cur_gripper_idx", 0)))
        d_idx = int(np.asarray(data.get("atomic/cur_duration_idx", 0)))
        atomic_tokens = np.asarray([t_idx, r_idx, g_idx, d_idx], dtype=np.int32)

        return {
            **data,
            "atomic_valid": atomic_valid,
            "atomic_tokens": atomic_tokens,
        }


@dataclasses.dataclass(frozen=True)
class ComputeAtomicDuration(DataTransformFn):
    """Compute atomic duration index from 'atomic/valid' sequence and write 'atomic/cur_duration_idx'.

    - Counts number of valid (True) steps within the current window.
    - Clips the count to [min_len, max_len].
    - Optionally maps to a compact index by subtracting min_len (for contiguous classes).
    """

    # Minimum and maximum duration in steps (inclusive clipping)
    min_len: int = 3
    max_len: int = 10
    # If True, store d - min_len (compact classes in [0, max_len-min_len])
    compact: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        valid = data.get("atomic/valid")
        if valid is None:
            return data

        v = np.asarray(valid)
        if v.ndim == 2 and v.shape[-1] == 1:
            v = v[..., 0]
        try:
            d = int(np.sum(v.astype(bool)))
        except Exception:
            # Fallback: leave unchanged if shape is unexpected
            return data

        d = int(np.clip(d, self.min_len, self.max_len))
        if self.compact:
            d = int(max(0, d - self.min_len))
        data["atomic/cur_duration_idx"] = np.asarray(d, dtype=np.int32)
        return data


@dataclasses.dataclass(frozen=True)
class PadStateActions(DataTransformFn):
    """Pad state/actions to model.action_dim along the last axis.

    - Pads data["state"] to (..., action_dim)
    - If present, pads data["actions"] to (..., action_horizon, action_dim)
    """

    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if "state" in out:
            out["state"] = pad_to_dim(np.asarray(out["state"]), self.action_dim, axis=-1)
        if "actions" in out:
            acts = np.asarray(out["actions"])
            acts = pad_to_dim(acts, self.action_dim, axis=-1)
            out["actions"] = acts
        return out


@dataclasses.dataclass(frozen=True)
class UnpadActions(DataTransformFn):
    """Slice action vectors back to dataset action dimension and ensure numpy.

    - If actions exist, returns actions[..., :out_dim] as a numpy array.
    """

    out_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        acts = np.asarray(data["actions"])
        if acts.ndim >= 1 and acts.shape[-1] >= self.out_dim:
            acts = acts[..., : self.out_dim]
        return {**data, "actions": acts}


@dataclasses.dataclass(frozen=True)
class RemapImagesForPi0(DataTransformFn):
    """Remap generic image/image_{i} keys to model-expected image keys and build masks.

    - Input expects after repack:
      - image/image_0, image/image_1, image/image_2 (and optionally image/image_3)
      - camera_present: bool[num_cams]
    - Output provides:
      - image: dict with keys base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
      - image_mask: dict with the same keys, from camera_present if available (else True)
    """

    use_camera_present: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        imgs = data.get("image")
        if not isinstance(imgs, dict):
            return data

        # get generic images if present
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

        # ensure required three views exist
        if im0 is None:
            im0 = im1 if im1 is not None else im2
        if im0 is None:
            im0 = zeros_like_image(im0)
        if im1 is None:
            im1 = zeros_like_image(im0)
        if im2 is None:
            im2 = zeros_like_image(im0)

        new_images = {
            "base_0_rgb": im0,
            "left_wrist_0_rgb": im1,
            "right_wrist_0_rgb": im2,
        }

        # build image masks from camera_present if requested
        if self.use_camera_present:
            if "camera_present" not in data:
                print("RemapImagesForPi0: missing required key 'camera_present' in sample")
                raise ValueError("camera_present is required to build image masks")
            try:
                cp = np.asarray(data["camera_present"]).astype(bool)
                # Require at least 3 entries for the three expected views
                if cp.size < 3:
                    print(f"RemapImagesForPi0: camera_present has insufficient length {cp.size}, expected >= 3")
                    raise ValueError("camera_present length must be at least 3")
                def cp_i(i: int) -> bool:
                    return bool(cp[i])
                image_mask = {
                    "base_0_rgb": cp_i(0),
                    "left_wrist_0_rgb": cp_i(1),
                    "right_wrist_0_rgb": cp_i(2),
                }
            except Exception as e:
                print(f"RemapImagesForPi0: failed to parse camera_present -> image_mask: {e}")
                raise
        else:
            image_mask = {k: True for k in new_images}

        out = {**data, "image": new_images, "image_mask": image_mask}
        return out

def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
