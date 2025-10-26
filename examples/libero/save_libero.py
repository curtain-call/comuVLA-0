# examples/libero/collect_libero_data.py
import collections
import dataclasses
import logging
import pathlib
import numpy as np
import tyro
import pickle
from PIL import Image

from openpi_client import image_tools
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class CollectArgs:
    """数据收集配置"""
    task_suite_name: str = "libero_spatial"
    num_test_tasks: int = 5
    resize_size: int = 224
    seed: int = 7
    output_dir: str = "libero_observation_data"


def collect_libero_observations(args: CollectArgs):
    """收集 Libero 环境的观测数据"""

    print(f"=== 收集 Libero 观测数据 ===")
    print(f"任务套件: {args.task_suite_name}")
    print(f"任务数量: {args.num_test_tasks}")

    # 创建输出目录
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 设置 Libero 环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    collected_data = []

    # 收集前几个任务的数据
    for task_id in range(min(args.num_test_tasks, task_suite.n_tasks)):
        print(f"\n{'=' * 50}")
        print(f"收集任务 {task_id + 1}/{min(args.num_test_tasks, task_suite.n_tasks)}")
        print(f"{'=' * 50}")

        try:
            # 获取任务
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            print(f"任务描述: {task_description}")

            # 重置环境并获取观测
            env.reset()
            obs = env.set_init_state(initial_states[0])

            # 获取图像（应用与训练时相同的预处理）
            agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            # 预处理图像
            agentview_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(agentview_img, args.resize_size, args.resize_size)
            )
            wrist_processed = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            # 获取状态信息
            robot_state = np.concatenate([
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ])

            # 收集数据
            task_data = {
                "task_id": task_id,
                "task_description": task_description,
                "task_suite": args.task_suite_name,
                "images": {
                    "agentview_raw": agentview_img,
                    "agentview_processed": agentview_processed,
                    "wrist_raw": wrist_img,
                    "wrist_processed": wrist_processed,
                },
                "robot_state": robot_state,
                "raw_obs": {
                    "robot0_eef_pos": obs["robot0_eef_pos"],
                    "robot0_eef_quat": obs["robot0_eef_quat"],
                    "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                },
                "metadata": {
                    "resize_size": args.resize_size,
                    "env_resolution": LIBERO_ENV_RESOLUTION,
                    "seed": args.seed,
                }
            }

            collected_data.append(task_data)

            # 保存单个任务的图像（用于验证）
            Image.fromarray(agentview_processed).save(output_dir / f"task_{task_id}_agentview.png")
            Image.fromarray(wrist_processed).save(output_dir / f"task_{task_id}_wrist.png")

            print(f"✅ 任务 {task_id} 数据收集完成")
            print(f"   - 任务描述: {task_description}")
            print(f"   - AgentView 图像形状: {agentview_processed.shape}")
            print(f"   - Wrist 图像形状: {wrist_processed.shape}")
            print(f"   - 机器人状态维度: {robot_state.shape}")

        except Exception as e:
            print(f"❌ 任务 {task_id} 数据收集失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存完整数据集
    data_file = output_dir / "libero_observations.pkl"
    with open(data_file, "wb") as f:
        pickle.dump(collected_data, f)

    # 保存数据摘要
    summary_file = output_dir / "data_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Libero 观测数据摘要\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"任务套件: {args.task_suite_name}\n")
        f.write(f"收集任务数: {len(collected_data)}\n")
        f.write(f"图像尺寸: {args.resize_size}x{args.resize_size}\n")
        f.write(f"环境分辨率: {LIBERO_ENV_RESOLUTION}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"\n任务列表:\n")

        for i, data in enumerate(collected_data):
            f.write(f"{i:2d}. {data['task_description']}\n")

    print(f"\n✅ 数据收集完成！")
    print(f"   - 收集了 {len(collected_data)} 个任务的数据")
    print(f"   - 数据文件: {data_file}")
    print(f"   - 摘要文件: {summary_file}")
    print(f"   - 图像文件保存在: {output_dir}")


def _get_libero_env(task, resolution, seed):
    """初始化 LIBERO 环境"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """四元数转轴角表示"""
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tyro.cli(collect_libero_observations)