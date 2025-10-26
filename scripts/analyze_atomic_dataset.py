import argparse
import dataclasses
import json
import os
import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tqdm_loggable.auto as tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data
from openpi.training.data_loader import create_torch_dataset, transform_dataset, AtomicWindowDataset


def _build_config(config_name: str, *, repo_id: str | None, root: str | None, batch_size: int) -> _config.TrainConfig:
    cfg = _config.get_config(config_name)
    if repo_id is not None or root is not None:
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
        num_workers=0,
        wandb_enabled=False,
    )
    return cfg


def analyze(
    *,
    config_name: str,
    repo_id: str | None,
    root: str | None,
    num_batches: int,
    output_json: str | None,
    output_png: str | None,
    log_interval: int,
    analyze_all: bool,
    batch_size: int,
):
    cfg = _build_config(config_name, repo_id=repo_id, root=root, batch_size=batch_size)

    # 若需要全量分析，预先计算窗口化后的数据集长度，推导需要的批次数
    if analyze_all:
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        base_ds = create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
        try:
            if len(base_ds) > 0 and "atomic.segment_start" in base_ds[0]:
                base_ds = AtomicWindowDataset(base_ds)
        except Exception:
            pass
        transformed = transform_dataset(base_ds, data_config, skip_norm_stats=False)
        try:
            ds_len = len(transformed)
        except Exception:
            ds_len = len(base_ds)
        if ds_len <= 0:
            raise RuntimeError("Dataset appears empty after windowing.")
        # torch DataLoader 在本项目中本地 batch_size = cfg.batch_size（单进程）
        num_batches = int(np.ceil(ds_len / float(cfg.batch_size)))
        logging.info("Analyze all: dataset_len=%d, batch_size=%d, num_batches=%d", ds_len, cfg.batch_size, num_batches)

    # 用训练同构的数据构建与变换，DataLoader 内部会自动应用 AtomicWindowDataset
    dl = _data.create_data_loader(cfg, shuffle=False, skip_norm_stats=False, num_batches=num_batches)
    data_iter = iter(dl)

    # 四个头的计数器
    cnt_t, cnt_r, cnt_g, cnt_d = Counter(), Counter(), Counter(), Counter()
    total = 0

    pbar = tqdm.tqdm(
        range(0, num_batches),
        initial=0,
        total=num_batches,
        dynamic_ncols=True,
    )

    for i in pbar:
        obs, _ = next(data_iter)
        labels = obs.atomic_tokens  # [B, 4]
        if labels is None:
            continue
        labels = np.asarray(labels)
        for j in range(labels.shape[0]):
            t, r, g, d = map(int, labels[j])
            cnt_t[t] += 1
            cnt_r[r] += 1
            cnt_g[g] += 1
            cnt_d[d] += 1
            total += 1

        if (i % max(1, log_interval)) == 0:
            # 打印阶段统计：已处理窗口数与每头已覆盖的类别数
            logging.info(
                "Step %d: windows=%d, unique(t,r,g,d)=(%d,%d,%d,%d)",
                i,
                total,
                len(cnt_t), len(cnt_r), len(cnt_g), len(cnt_d),
            )

    stats = {
        "total_windows": total,
        "translation": dict(cnt_t),
        "rotation": dict(cnt_r),
        "gripper": dict(cnt_g),
        "duration": dict(cnt_d),
    }

    # 保存 JSON
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # 可视化
    if output_png:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        heads = [("translation", cnt_t), ("rotation", cnt_r), ("gripper", cnt_g), ("duration", cnt_d)]
        for ax, (name, counter) in zip(axes.ravel(), heads):
            if len(counter) == 0:
                ax.set_title(f"{name} (empty)")
                continue
            xs = sorted(counter.keys())
            ys = [counter[x] for x in xs]
            ax.bar(xs, ys)
            ax.set_title(name)
            ax.set_xlabel("class index")
            ax.set_ylabel("count")
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
        plt.savefig(output_png, dpi=150, bbox_inches="tight")

    print(json.dumps(stats, indent=2))


def main():
    p = argparse.ArgumentParser("Analyze atomic token distribution on AtomicWindowDataset")
    p.add_argument("--config-name", type=str, default="pi0_atomic_bridge")
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--num-batches", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--all", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--output-png", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    analyze(
        config_name=args.config_name,
        repo_id=args.repo_id,
        root=args.root,
        num_batches=args.num_batches,
        output_json=args.output_json,
        output_png=args.output_png,
        log_interval=args.log_interval,
        analyze_all=args.all,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


