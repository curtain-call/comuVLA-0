import argparse
import dataclasses
import json
import logging
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import flax.nnx as nnx
import jax.sharding as jshard
from flax import traverse_util as _traverse

from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training import sharding as _sharding
from openpi.models import model as _model


def _build_config(
    config_name: str,
    *,
    repo_id: str | None,
    root: str | None,
    batch_size: int,
    num_workers: int,
) -> _config.TrainConfig:
    cfg = _config.get_config(config_name)
    if repo_id is not None or root is not None:
        # 需要覆盖 DataConfigFactory 自身的 repo_id/root，而不是仅修改 base_config，
        # 因为 create_base_config 会使用 factory.repo_id 与 factory.root。
        data_cfg = dataclasses.replace(
            cfg.data,
            repo_id=repo_id or getattr(cfg.data, "repo_id", None),
            root=root or getattr(cfg.data, "root", None),
            base_config=dataclasses.replace(
                (cfg.data.base_config or _config.DataConfig()),
                repo_id=repo_id or (getattr(cfg.data.base_config, "repo_id", None) if cfg.data.base_config else None),
                root=root or (getattr(cfg.data.base_config, "root", None) if cfg.data.base_config else None),
            ),
        )
    else:
        data_cfg = cfg.data

    cfg = dataclasses.replace(
        cfg,
        data=data_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        wandb_enabled=False,
    )
    return cfg


def init_logging():
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
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


def _forward_atomic_logits(model: _model.BaseModel, observation: _model.Observation) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)  # type: ignore[attr-defined]

    kv_policy = getattr(model.config, "kv_policy", "full")  # type: ignore[attr-defined]
    if kv_policy == "placeholder":
        batch = observation.state.shape[0]
        pal_width = model.paligemma_width  # type: ignore[attr-defined]
        P = 4
        ph_tokens = jnp.zeros((batch, P, pal_width), dtype=jnp.float32)
        ph_mask = jnp.ones((batch, P), dtype=jnp.bool_)
        ph_ar = jnp.array([True] + [False] * (P - 1))

        prefix_ext = jnp.concatenate([prefix_tokens, ph_tokens], axis=1)
        prefix_ext_mask = jnp.concatenate([prefix_mask, ph_mask], axis=1)
        prefix_ext_ar = jnp.concatenate([prefix_ar_mask, ph_ar], axis=0)

        def make_attn_mask(input_mask, mask_ar):
            mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
            cumsum = jnp.cumsum(mask_ar, axis=1)
            attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
            valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
            return jnp.logical_and(attn_mask, valid_mask)

        prefix_ext_attn = make_attn_mask(prefix_ext_mask, prefix_ext_ar)
        prefix_ext_pos = jnp.cumsum(prefix_ext_mask, axis=1) - 1
        (prefix_ext_out, _), _ = model.PaliGemma.llm([prefix_ext, None], mask=prefix_ext_attn, positions=prefix_ext_pos)  # type: ignore[attr-defined]
        pooled = jnp.mean(prefix_ext_out[:, -P:, :], axis=1)
    else:
        def make_attn_mask(input_mask, mask_ar):
            mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
            cumsum = jnp.cumsum(mask_ar, axis=1)
            attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
            valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
            return jnp.logical_and(attn_mask, valid_mask)

        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn, positions=positions)  # type: ignore[attr-defined]
        pooled = jnp.mean(prefix_out, axis=1)

    logits_t = model.atomic_head_t(pooled)  # type: ignore[attr-defined]
    logits_r = model.atomic_head_r(pooled)  # type: ignore[attr-defined]
    logits_g = model.atomic_head_g(pooled)  # type: ignore[attr-defined]
    logits_d = model.atomic_head_d(pooled)  # type: ignore[attr-defined]
    return logits_t, logits_r, logits_g, logits_d


def _topk_acc(logits: jax.Array, labels: jax.Array, k: int = 1) -> jax.Array:
    topk = jnp.argsort(logits, axis=-1)[:, ::-1][:, :k]
    return jnp.mean((topk == labels[..., None]).any(axis=-1))


def evaluate(
    *,
    config_name: str,
    params_path: str,
    repo_id: str | None,
    root: str | None,
    batch_size: int,
    num_workers: int,
    num_batches: int,
    action_sample_steps: int,
    skip_norm_stats: bool,
    output_json: str | None,
    plot_path: str | None,
    show_plot: bool,
    fsdp_devices: int,
    log_interval: int,
    atomic_only: bool,
    cast_bfloat16: bool,
):
    cfg = _build_config(config_name, repo_id=repo_id, root=root, batch_size=batch_size, num_workers=num_workers)

    # Build mesh for evaluation and set data sharding over (batch, fsdp) axes.
    mesh = _sharding.make_mesh(fsdp_devices)
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec(_sharding.DATA_AXIS))

    dl = _data.create_data_loader(
        cfg,
        sharding=data_sharding,
        shuffle=False,
        num_batches=num_batches,
        skip_norm_stats=skip_norm_stats,
    )

    # Load params and reshard using the same policy as training: derive sharding from model state shapes.
    params = _model.restore_params(params_path, restore_type=jax.Array)
    if cast_bfloat16:
        params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and x.dtype in (jnp.float32, jnp.bfloat16) else x,
            params,
        )
    model_shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    state_shape = nnx.state(model_shape).to_pure_dict()
    state_sharding = _sharding.fsdp_sharding(state_shape, mesh, log=False)
    # Align sharding tree to the loaded subset of params
    flat_params = _traverse.flatten_dict(params, sep="/")
    flat_shard = _traverse.flatten_dict(state_sharding, sep="/")
    flat_shard_aligned = {k: flat_shard[k] for k in flat_params.keys() if k in flat_shard}
    param_sharding = _traverse.unflatten_dict(flat_shard_aligned, sep="/")
    params = jax.tree.map(lambda v, s: jax.device_put(v, s), params, param_sharding)
    model = cfg.model.load(params, remove_extra_params=True)

    # Metric accumulators
    m = {
        "atomic/ce_t": 0.0,
        "atomic/ce_r": 0.0,
        "atomic/ce_g": 0.0,
        "atomic/ce_d": 0.0,
        "atomic/ce_total": 0.0,
        "atomic/top1_t": 0.0,
        "atomic/top1_r": 0.0,
        "atomic/top1_g": 0.0,
        "atomic/top1_d": 0.0,
        "atomic/top3_t": 0.0,
        "atomic/top3_r": 0.0,
        "atomic/top3_g": 0.0,
        "atomic/top3_d": 0.0,
        "action/mse": 0.0,
        "action/mae": 0.0,
        "n_batches": 0,
    }

    rng = jax.random.key(0)
    with _sharding.set_mesh(mesh), jax.default_matmul_precision("bfloat16"):
        data_iter = iter(dl)
        for i in range(num_batches):
            observation, actions = next(data_iter)
            logits_t, logits_r, logits_g, logits_d = _forward_atomic_logits(model, observation)
            y = observation.atomic_tokens
            ce_t = optax.softmax_cross_entropy_with_integer_labels(logits_t, y[..., 0]).mean()
            ce_r = optax.softmax_cross_entropy_with_integer_labels(logits_r, y[..., 1]).mean()
            ce_g = optax.softmax_cross_entropy_with_integer_labels(logits_g, y[..., 2]).mean()
            ce_d = optax.softmax_cross_entropy_with_integer_labels(logits_d, y[..., 3]).mean()
            ce_total = ce_t + ce_r + ce_g + ce_d

            acc1_t = _topk_acc(logits_t, y[..., 0], k=1)
            acc1_r = _topk_acc(logits_r, y[..., 1], k=1)
            acc1_g = _topk_acc(logits_g, y[..., 2], k=1)
            acc1_d = _topk_acc(logits_d, y[..., 3], k=1)
            acc3_t = _topk_acc(logits_t, y[..., 0], k=3)
            acc3_r = _topk_acc(logits_r, y[..., 1], k=3)
            acc3_g = _topk_acc(logits_g, y[..., 2], k=3)
            acc3_d = _topk_acc(logits_d, y[..., 3], k=3)

            if not atomic_only:
                rng, step_rng = jax.random.split(rng)
                pred_actions = model.sample_actions(step_rng, observation, num_steps=action_sample_steps)  # type: ignore[attr-defined]
                valid = observation.atomic_valid
                diff = pred_actions - actions
                sq = jnp.square(diff).mean(axis=-1)  # [b, H]
                ab = jnp.abs(diff).mean(axis=-1)
                if valid is not None:
                    v = valid.astype(jnp.float32)
                    mse = (sq * v).sum() / (v.sum() + 1e-8)
                    mae = (ab * v).sum() / (v.sum() + 1e-8)
                else:
                    mse = sq.mean()
                    mae = ab.mean()

                m["action/mse"] += float(mse)
                m["action/mae"] += float(mae)

            m["atomic/ce_t"] += float(ce_t)
            m["atomic/ce_r"] += float(ce_r)
            m["atomic/ce_g"] += float(ce_g)
            m["atomic/ce_d"] += float(ce_d)
            m["atomic/ce_total"] += float(ce_total)
            m["atomic/top1_t"] += float(acc1_t)
            m["atomic/top1_r"] += float(acc1_r)
            m["atomic/top1_g"] += float(acc1_g)
            m["atomic/top1_d"] += float(acc1_d)
            m["atomic/top3_t"] += float(acc3_t)
            m["atomic/top3_r"] += float(acc3_r)
            m["atomic/top3_g"] += float(acc3_g)
            m["atomic/top3_d"] += float(acc3_d)
            m["n_batches"] += 1

            if (i % max(1, log_interval)) == 0:
                cur_nb = max(1, i + 1)
                # 构造当前阶段的所有指标（均值）
                log_items = {}
                for k, v in m.items():
                    if k == "n_batches":
                        continue
                    try:
                        log_items[k] = float(v) / cur_nb
                    except Exception:
                        pass
                # 排序后打印，便于对齐
                msg = ", ".join(f"{k}={log_items[k]:.4f}" for k in sorted(log_items.keys()))
                logging.info(f"Step {i}: {msg}")

    nb = max(1, m["n_batches"])
    summary = {k: (v / nb if k != "n_batches" else v) for k, v in m.items()}

    print(json.dumps(summary, indent=2))
    # 以 logging 再打印一次完整汇总，便于统一采集
    try:
        msg = ", ".join(f"{k}={summary[k]:.4f}" if isinstance(summary[k], (int, float)) else f"{k}={summary[k]}" for k in sorted(summary.keys()))
        logging.info(f"Summary: {msg}")
    except Exception:
        logging.info(f"Summary: {summary}")
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # Visualization of accuracies
    if plot_path or show_plot:
        labels = ["t", "r", "g", "d"]
        top1 = [
            float(summary.get("atomic/top1_t", 0.0)),
            float(summary.get("atomic/top1_r", 0.0)),
            float(summary.get("atomic/top1_g", 0.0)),
            float(summary.get("atomic/top1_d", 0.0)),
        ]
        top3 = [
            float(summary.get("atomic/top3_t", 0.0)),
            float(summary.get("atomic/top3_r", 0.0)),
            float(summary.get("atomic/top3_g", 0.0)),
            float(summary.get("atomic/top3_d", 0.0)),
        ]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, top1, width, label="top1")
        ax.bar(x + width / 2, top3, width, label="top3")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Atomic token accuracy (per head)")
        ax.legend()
        plt.tight_layout()
        if plot_path:
            os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()


def main():
    p = argparse.ArgumentParser("Evaluate atomic classification and action prediction")
    p.add_argument("--config-name", type=str, default="pi0_atomic_bridge")
    p.add_argument("--params", type=str, required=True, help="Path to params directory")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--action-sample-steps", type=int, default=10)
    p.add_argument("--skip-norm-stats", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--plot-path", type=str, default=None)
    p.add_argument("--show-plot", action="store_true")
    p.add_argument("--fsdp-devices", type=int, default=4, help="Number of FSDP shards across devices")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--atomic-only", action="store_true")
    p.add_argument("--cast-bfloat16", action="store_true")
    args = p.parse_args()

    init_logging()

    evaluate(
        config_name=args.config_name,
        params_path=args.params,
        repo_id=args.repo_id,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        action_sample_steps=args.action_sample_steps,
        skip_norm_stats=args.skip_norm_stats,
        output_json=args.output_json,
        plot_path=args.plot_path,
        show_plot=args.show_plot,
        fsdp_devices=args.fsdp_devices,
        log_interval=args.log_interval,
        atomic_only=args.atomic_only,
        cast_bfloat16=args.cast_bfloat16,
    )


if __name__ == "__main__":
    main()


