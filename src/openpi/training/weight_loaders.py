import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        # path = download.maybe_download(
        #     "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        # )
        path = pathlib.Path("/home/zhiyu/mzh/checkpoints/paligemma-3b-pt-224.npz")
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

    
    
def _flatten_param_keys(params: at.Params) -> set[str]:
    """Flattens a nested params dict to a set of string keys joined by '/'."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    return set(flat.keys())
    
    
def compare_weight_loader_keys(
    params: at.Params, loader_a: WeightLoader, loader_b: WeightLoader
) -> dict[str, set[str]]:
    """Compares the flattened param key sets produced by two loaders.
    
    Args:
        params: Reference parameter structure to guide merging.
        loader_a: First WeightLoader.
        loader_b: Second WeightLoader.
    
    Returns:
        A dict with key sets:
          - 'only_in_a': Keys present only in A-loaded params.
          - 'only_in_b': Keys present only in B-loaded params.
          - 'in_both': Keys present in both.
    """
    loaded_a = loader_a.load(params)
    loaded_b = loader_b.load(params)
    
    keys_a = _flatten_param_keys(loaded_a)
    keys_b = _flatten_param_keys(loaded_b)
    
    return {
        "only_in_a": keys_a - keys_b,
        "only_in_b": keys_b - keys_a,
        "in_both": keys_a & keys_b,
    }
    
    
def compare_checkpoint_and_paligemma_keys(
    params: at.Params, checkpoint_path: str
) -> dict[str, set[str]]:
    """Convenience wrapper to compare Checkpoint vs PaliGemma loader key sets.
    
    Args:
        params: Reference parameter structure (from your model init).
        checkpoint_path: Path or URI to the checkpoint params.
    
    Returns:
        Same structure as compare_weight_loader_keys.
    """
    a = CheckpointWeightLoader(params_path=checkpoint_path)
    b = PaliGemmaWeightLoader()
    result = compare_weight_loader_keys(params, a, b)
    
    logger.info("Key comparison (Checkpoint vs PaliGemma): "
                "only_in_checkpoint=%d, only_in_paligemma=%d, in_both=%d",
                len(result["only_in_a"]), len(result["only_in_b"]), len(result["in_both"]))
    return result