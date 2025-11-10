import argparse
import os
import sys
import json
import inspect
from typing import Iterable

try:
    from huggingface_hub import HfApi, HfFolder
except Exception as e:  # pragma: no cover
    print("[ERROR] huggingface_hub 未安装。请先执行: pip install -U huggingface_hub", file=sys.stderr)
    raise


def _is_lerobot_dataset_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


def _read_info_json(path: str) -> dict | None:
    info_p = os.path.join(path, "meta", "info.json")
    try:
        with open(info_p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _ensure_readme(local_root: str, repo_id: str, force: bool = False) -> None:
    readme_p = os.path.join(local_root, "README.md")
    if os.path.exists(readme_p) and not force:
        return
    info = _read_info_json(local_root) or {}
    title = info.get("repo_id", repo_id)
    lines = [
        f"# {title}",
        "",
        "LeRobot 格式的数据集。该仓库存放由本地生成的 episodes 与元数据。",
        "",
        "## 结构",
        "- meta/info.json: 数据集元信息",
        "- episodes/: 数据切分后的 parquet/图像等",
        "",
        "## 备注",
        "- 本 README 由上传脚本自动生成，可按需修改。",
    ]
    try:
        with open(readme_p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass


def _get_token(cli_token: str | None) -> str | None:
    token = (
        cli_token
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or HfFolder.get_token()
    )
    return token


def _normalize_patterns(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for v in values:
        if not v:
            continue
        # 支持以逗号分隔的一串模式
        parts = [p.strip() for p in str(v).split(",")]
        out.extend([p for p in parts if p])
    return out or None


def upload(
    local_root: str,
    repo_id: str,
    *,
    private: bool,
    token: str | None,
    branch: str,
    message: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    max_workers: int,
    dry_run: bool,
    ensure_card: bool,
):
    if not os.path.isdir(local_root):
        raise FileNotFoundError(f"本地目录不存在: {local_root}")
    if not _is_lerobot_dataset_root(local_root):
        print("[WARN] 目标目录看起来不是 LeRobot 数据集根目录（缺少 meta/info.json）。仍将按普通目录上传。", file=sys.stderr)

    if ensure_card:
        _ensure_readme(local_root, repo_id, force=False)

    token = _get_token(token)
    if not token:
        print("[ERROR] 未检测到 Hugging Face 访问令牌。请执行 `huggingface-cli login` 或设置环境变量 HUGGINGFACE_TOKEN/HF_TOKEN，或使用 --token 传入。", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)

    # 创建（或复用）数据集仓库
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    # 默认忽略模式，附加用户自定义
    default_ignore = [
        "**/.git/**",
        "**/__pycache__/**",
        "**/*.tmp",
        "**/*.lock",
        "**/tmp/**",
    ]
    if ignore_patterns:
        default_ignore.extend(ignore_patterns)
    ignore_patterns = default_ignore

    if dry_run:
        print("[DRY-RUN] 将上传目录:", local_root)
        print("[DRY-RUN] 目标仓库:", repo_id)
        print("[DRY-RUN] 分支:", branch)
        print("[DRY-RUN] 提交说明:", message)
        print("[DRY-RUN] 允许模式:", allow_patterns or ["<ALL>"])
        print("[DRY-RUN] 忽略模式:", ignore_patterns)
        return

    # 执行上传（兼容不同版本的 huggingface_hub）
    kwargs = dict(
        folder_path=local_root,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=".",
        commit_message=message,
        revision=branch,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    try:
        sig = inspect.signature(api.upload_folder)
        if "max_workers" in sig.parameters:
            kwargs["max_workers"] = max_workers
        elif "num_workers" in sig.parameters:
            kwargs["num_workers"] = max_workers
    except Exception:
        # 如果无法检测签名，直接尝试不带并发参数
        pass

    try:
        api.upload_folder(**kwargs)
    except TypeError:
        # 回退：移除可能不被支持的并发参数再试一次
        kwargs.pop("max_workers", None)
        kwargs.pop("num_workers", None)
        api.upload_folder(**kwargs)

    print(f"已上传到 https://huggingface.co/datasets/{repo_id} （分支: {branch}）")


def main():
    p = argparse.ArgumentParser("Upload local LeRobot dataset folder to Hugging Face Hub")
    p.add_argument("local_root", type=str, help="本地 LeRobot 数据集根目录（包含 meta/info.json）")
    p.add_argument("repo_id", type=str, help="目标数据集仓库 id，如 username/dataset-name")
    p.add_argument("--private", action="store_true", help="创建为私有仓库")
    p.add_argument("--token", type=str, default=None, help="Hugging Face 访问令牌（可用环境变量代替）")
    p.add_argument("--branch", type=str, default="main", help="目标分支/修订（默认 main）")
    p.add_argument("--message", type=str, default="Upload LeRobot dataset", help="提交说明")
    p.add_argument(
        "--allow-patterns",
        type=str,
        nargs="*",
        default=None,
        help="仅上传匹配的通配符模式（多值或逗号分隔）",
    )
    p.add_argument(
        "--ignore-patterns",
        type=str,
        nargs="*",
        default=None,
        help="忽略匹配的通配符模式（多值或逗号分隔）",
    )
    p.add_argument("--max-workers", type=int, default=8, help="并发上传 worker 数")
    p.add_argument("--dry-run", action="store_true", help="仅打印信息，不执行上传")
    p.add_argument("--no-card", action="store_true", help="不自动生成 README.md")
    args = p.parse_args()

    allow_patterns = _normalize_patterns(args.allow_patterns)
    ignore_patterns = _normalize_patterns(args.ignore_patterns)

    upload(
        local_root=args.local_root,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        branch=args.branch,
        message=args.message,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max(int(args.max_workers), 1),
        dry_run=bool(args.dry_run),
        ensure_card=not bool(args.no_card),
    )


if __name__ == "__main__":
    main()


