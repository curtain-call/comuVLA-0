import jax
import flax.nnx as nnx
from openpi.training.weight_loaders import (
    compare_weight_loader_keys,
    CheckpointWeightLoader,
    PaliGemmaWeightLoader,
)
import openpi.training.config as _config

def main(config: _config.TrainConfig):
    rng = jax.random.PRNGKey(0)
    model = config.model.create(rng)
    ref_params = nnx.state(model).to_pure_dict()  # 参考参数（键空间即为对比所需）
    diff = compare_weight_loader_keys(
    ref_params,
    CheckpointWeightLoader(params_path="/home/zhiyu/mzh/pi0_base/params"),
    PaliGemmaWeightLoader(),
    )
    print("only_in_checkpoint:", len(diff["only_in_a"]))
    print("only_in_paligemma:", len(diff["only_in_b"]))
    print("in_both:", len(diff["in_both"]))




if __name__ == "__main__":
    main(_config.cli())
