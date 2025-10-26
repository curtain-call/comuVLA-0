import dataclasses
import functools
import logging
import platform
import re
from gc import is_finalized
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _check_params_subset(expected_shape_tree: at.Params, loaded_tree: at.Params, *, check_shapes: bool = True, check_dtypes: bool = True) -> None:
    """Validate that loaded_tree is a subset of expected_shape_tree, and for overlapping keys, shapes/dtypes match."""
    flat_expected = traverse_util.flatten_dict(expected_shape_tree)
    flat_loaded = traverse_util.flatten_dict(loaded_tree)

    # Check for unexpected keys
    unexpected = [k for k in flat_loaded.keys() if k not in flat_expected]
    if unexpected:
        raise ValueError(f"Loaded params contain unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def _shape_dtype(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return x.shape, x.dtype
        # jnp.ndarray, np.ndarray, or nnx Param value
        try:
            return x.shape, x.dtype
        except Exception:
            return None, None

    # Check shapes/dtypes for overlapping keys
    for k, v in flat_loaded.items():
        exp = flat_expected[k]
        es, ed = _shape_dtype(exp)
        ls, ld = _shape_dtype(v)
        if check_shapes and (es is not None and ls is not None) and es != ls:
            raise ValueError(f"Shape mismatch at {k}: expected {es}, got {ls}")
        if check_dtypes and (ed is not None and ld is not None) and ed != ld:
            raise ValueError(f"Dtype mismatch at {k}: expected {ed}, got {ld}")


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads weights and validates as a subset of params_shape. Returns only concrete loaded weights (no shapes)."""
    loaded_params = loader.load(params_shape)
    _check_params_subset(params_shape, loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
        config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def is_action_expert_param(path, param):
        if isinstance(path, tuple):
            # 检查路径的第一个元素是否是投影层
            if len(path) > 0:
                first_key = path[0]
                is_action_projection = first_key in ['action_in_proj', 'action_time_mlp_in', 'action_time_mlp_out',
                                                     'action_out_proj', 'state_proj']
            else:
                is_action_projection = False

            # 检查路径字符串中是否包含 action expert 层（带数字后缀）
            path_str = '/'.join(str(p) for p in path)
            is_siglip = re.search(r'/img', path_str)
            is_embedder = re.search(r'/embedder', path_str)
            is_final_norm = re.search(r'/final_norm', path_str)
            is_action_expert_layer = re.search(r'_(1|2|3|4|5|6|7|8|9)', path_str)
        else:
            # 如果 path 是字符串，直接检查
            is_action_projection = path in ['action_in_proj', 'action_time_mlp_in', 'action_time_mlp_out',
                                            'action_out_proj', 'state_proj']
            is_siglip = re.search(r'/img', path)
            is_embedder = re.search(r'/embedder', path)
            is_final_norm = re.search(r'/final_norm', path)
            is_action_expert_layer = re.search(r'_(1|2|3|4|5|6|7|8|9)', path)

        # 返回 True 表示这些参数需要训练
        # return (not is_siglip and not is_embedder) and ( is_action_expert_layer or is_action_projection )

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)
        # 打印参数字典看一下
        # print(nnx.state(model).to_pure_dict()["PaliGemma"]["llm"]["layers"].keys())

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        # params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        # action_params = params.filter(is_action_expert_param)
        # opt_state = tx.init(action_params)
        # Initialize optimizer on the trainable subset to keep tree structures consistent with grads/updates.
        opt_state = tx.init(params.filter(config.trainable_filter))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            # opt_state=tx.init(params.filter(config.trainable_filter)),
            opt_state=opt_state,  
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    # replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


    # 先得到参考 shape tree
    params_shape_pure = train_state_shape.params.to_pure_dict()

    # 仅加载 VLM（PaliGemma）权重；其余模块保持随机初始化
    vlm_loader = _weight_loaders.PaliGemmaWeightLoader()
    merged_partial_params = vlm_loader.load(params_shape_pure)
    # 过滤掉占位的 ShapeDtypeStruct，只保留实际数组权重，避免传入 jitted init 报类型错误
    merged_partial_params = traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(merged_partial_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )

        # 检查 PaliGemma 权重覆盖后的状态
    # print("\n=== 检查 PaliGemma 权重覆盖 ===")
    # print("VLM 部分 (PaliGemma/llm/layers/attn):")
    # if 'PaliGemma' in merged_partial_params and 'llm' in merged_partial_params['PaliGemma']:
    #     vlm_attn = merged_partial_params['PaliGemma']['llm']['layers']['attn']
    #     print(f"  attn 参数数量: {len(jax.tree_util.tree_flatten(vlm_attn)[0])}")
    #     # 检查是否包含 PaliGemma 特有的参数
    #     if 'q_einsum' in vlm_attn:
    #         print(f"  q_einsum 权重形状: {vlm_attn['q_einsum']['w'].shape}")
    #         print(f"  q_einsum 权重均值: {jnp.mean(vlm_attn['q_einsum']['w']):.6f}")
    #         # 对比变化
    #         if 'q_einsum' in base_partial_params['PaliGemma']['llm']['layers']['attn']:
    #             old_mean = jnp.mean(base_partial_params['PaliGemma']['llm']['layers']['attn']['q_einsum']['w'])
    #             new_mean = jnp.mean(vlm_attn['q_einsum']['w'])
    #             print(f"  q_einsum 权重均值变化: {old_mean:.6f} -> {new_mean:.6f}")
    
    # print("Action Expert 部分 (应该保持不变):")
    # if 'action_in_proj' in merged_partial_params:
    #     print(f"  action_in_proj 权重形状: {merged_partial_params['action_in_proj']['kernel'].shape}")
    #     print(f"  action_in_proj 权重均值: {jnp.mean(merged_partial_params['action_in_proj']['kernel']):.6f}")
    #     # 检查是否与 pi0_base 一致
    #     if 'action_in_proj' in base_partial_params:
    #         old_mean = jnp.mean(base_partial_params['action_in_proj']['kernel'])
    #         new_mean = jnp.mean(merged_partial_params['action_in_proj']['kernel'])
    #         print(f"  action_in_proj 权重均值变化: {old_mean:.6f} -> {new_mean:.6f} (应该基本一致)")
    #

    # 可选校验：作为参考 shape 的子集（仅检查 VLM 覆盖部分）
    _check_params_subset(params_shape_pure, merged_partial_params, check_shapes=True, check_dtypes=True)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the merged partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, merged_partial_params)


    return train_state, state_sharding


@at.typecheck
def train_step(
        config: _config.TrainConfig,
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    # Use total loss (sum of weighted components) for gradient computation
    @at.typecheck
    def total_loss_fn(
            model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        total_loss, _ = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(total_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch


    # def is_trainable_param(path, param):
    #     if isinstance(path, tuple):
    #         # 检查路径的第一个元素是否是投影层
    #         if len(path) > 0:
    #             first_key = path[0]
    #             is_action_projection = first_key in ['action_in_proj', 'action_time_mlp_in', 'action_time_mlp_out',
    #                                                  'action_out_proj', 'state_proj']
    #         else:
    #             is_action_projection = False

    #         # 检查路径字符串中是否包含 action expert 层（带数字后缀）
    #         path_str = '/'.join(str(p) for p in path)
    #         is_siglip = re.search(r'/img', path_str)
    #         is_embedder = re.search(r'/embedder', path_str)
    #         is_final_norm = re.search(r'/final_norm', path_str)
    #         is_action_expert_layer = re.search(r'_(1|2|3|4|5|6|7|8|9)', path_str)
    #     else:
    #         # 如果 path 是字符串，直接检查
    #         is_action_projection = path in ['action_in_proj', 'action_time_mlp_in', 'action_time_mlp_out',
    #                                         'action_out_proj', 'state_proj']
    #         is_siglip = re.search(r'/img', path)
    #         is_embedder = re.search(r'/embedder', path)
    #         is_final_norm = re.search(r'/final_norm', path)
    #         is_action_expert_layer = re.search(r'_(1|2|3|4|5|6|7|8|9)', path)

    #     # 返回 True 表示这些参数需要训练
    #     return (not is_siglip and not is_embedder) and (is_action_expert_layer or is_action_projection)

    # # 仅 VLM 用 LoRA；Action Expert 全量训练（当且仅当：VLM 为 lora 变体，Action Expert 非 lora 变体）
    # def is_trainable_vlm_lora_and_action_full(path, param):
    #     pal_lora = hasattr(config.model, 'paligemma_variant') and ('lora' in str(config.model.paligemma_variant))
    #     ae_non_lora = hasattr(config.model, 'action_expert_variant') and ('lora' not in str(config.model.action_expert_variant))
    #     if not pal_lora or not ae_non_lora:
    #         return is_trainable_param(path, param)

    #     # path 解析
    #     path_str = '/'.join(str(p) for p in path) if isinstance(path, tuple) else str(path)
    #     is_img = re.search(r"/img", path_str) is not None
    #     is_embed = re.search(r"/embedder", path_str) is not None
    #     is_llm = re.search(r"/llm", path_str) is not None
    #     has_lora = re.search(r"lora", path_str) is not None
    #     # 经验性：action expert 分支通常以 _1 区分
    #     is_action_expert_branch = re.search(r"_1", path_str) is not None

    #     # 1) Action Expert: 全量训练（排除视觉编码/嵌入器）
    #     if is_action_expert_branch and (not is_img):
    #         return True

    #     # 2) 投影层：始终训练（state/action 投影用于控制）
    #     if isinstance(path, tuple):
    #         first_key = path[0] if len(path) > 0 else None
    #     else:
    #         first_key = path
    #     if first_key in ['action_in_proj', 'action_time_mlp_in', 'action_time_mlp_out', 'action_out_proj', 'state_proj']:
    #         return True

    #     # 3) VLM: 仅 LoRA 训练（排除视觉编码/嵌入器）
    #     if is_llm and has_lora and (not is_img) and (not is_embed) and (not is_action_expert_branch):
    #         return True

    #     return False


    # def ntp_loss_fn(model, rng, observation, actions):
    #     _, ntp_loss = model.compute_loss(rng, observation, actions, train=True)
    #     return jnp.mean(ntp_loss)

    # # vlm_diff_state = nnx.DiffState(0, is_vlm_param)
    # # ntp_loss, ntp_grads = nnx.value_and_grad(ntp_loss_fn, argnums=vlm_diff_state)(model, rng, observation, actions)

    # # ntp_weight = jax.lax.stop_gradient(action_loss / (ntp_loss + 1e-6))

    # def zero_like_tree(tree):
    #     return jax.tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, tree)

    # def tree_update(base, update):
    #     if isinstance(base, dict) and isinstance(update, dict):
    #         return {k: tree_update(base[k], update[k]) if k in update else base[k] for k in base}
    #     elif update is None:
    #         return update
    #     else:return base

    # def merge_grad(action_g, ntp_g):
    #     ntp_g_full = zero_like_tree(action_g)
    #     ntp_g_full = tree_update(ntp_g_full, ntp_g)
    #     return jax.tree_map(lambda a,n: a + n if n is not None else a, action_g, ntp_g_full)
    # 0.01统一量纲

    # Filter out frozen params via config.freeze_filter (trainable = Param AND NOT freeze_filter)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    total_loss, grads = nnx.value_and_grad(total_loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    # Select trainable subset according to config.freeze_filter
    params_sub = state.params.filter(config.trainable_filter)
    grads_sub = grads.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads_sub, state.opt_state, params_sub)
    new_params_sub = optax.apply_updates(params_sub, updates)

    full_pure   = state.params.to_pure_dict()
    subset_pure = new_params_sub.to_pure_dict()

        # 将更新后的参数合并回完整的参数结构
    def merge_updates_pure(full_params_pure: dict, updates_subset_pure: dict) -> dict:
        flat_full = traverse_util.flatten_dict(full_params_pure, sep="/")
        flat_upd  = traverse_util.flatten_dict(updates_subset_pure, sep="/")
        for k, v in flat_upd.items():
            flat_full[k] = v
        return traverse_util.unflatten_dict(flat_full, sep="/")
    
    # 更新完整参数结构
    merged_pure = merge_updates_pure(full_pure, subset_pure)

    # Update the model in place and return the new full state.
    # nnx.update(model, updated_full_params)
    # new_params = nnx.state(model)
    graphdef, cur_state = nnx.split(model)
    cur_state.replace_by_pure_dict(merged_pure)
    model = nnx.merge(graphdef, cur_state)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    action_grad_norm = optax.global_norm(grads)
    # ntp_grad_norm = optax.global_norm(ntp_grads)

    # Compute detailed losses for logging
    _, loss_dict = model.compute_loss(train_rng, observation, actions, train=True)

    info = {

        "loss_total": total_loss,
        "loss_action": loss_dict.get("action", jnp.asarray(0.0)),
        "loss_atomic": loss_dict.get("atomic", jnp.asarray(0.0)),
        "loss_ntp": loss_dict.get("ntp", jnp.asarray(0.0)),
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "action_grad_norm": action_grad_norm,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
