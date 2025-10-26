import collections
import dataclasses
import logging
import math
import pathlib
import json, shutil
import time

import imageio
import numpy as np
import tyro
import tqdm
from PIL import Image, ImageDraw, ImageFont

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

out_root = pathlib.Path("/your/custom/dir")


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class KArgs:
    '''配置类'''
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1000  # 每个任务执行100次

    config: str = "pi0_libero"

    video_out_path: str = "/home/zhiyu/mzh/datasets/libero_failure_dataset_video"
    seed = 7

    # LeRobot 格式数据保存参数 - 只记录失败轨迹
    record_failure_trajectory: bool = True
    lerobot_repo_id: str = "maozihao/libero_failure_dataset"
    save_images: bool = True
    fps: int = 10
    push_to_hub: bool = False


def eval_libero_collect_failures(args: KArgs):
    """收集失败轨迹的评估函数"""
    np.random.seed(args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"每个任务将执行 {args.num_trials_per_task} 次，只记录失败轨迹")

    # 初始化 LeRobot 数据集（只用于失败轨迹）
    failure_dataset = None
    if args.record_failure_trajectory:
        # 清理现有数据集
        output_path = out_root / args.lerobot_repo_id
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)

        # 创建失败轨迹数据集
        failure_dataset = LeRobotDataset.create(
            repo_id=args.lerobot_repo_id,
            root="/home/zhiyu/mzh/datasets/libero_failure_dataset/",
            robot_type="panda",
            fps=args.fps,
            features={
                "image": {"dtype": "image",
                                             "shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3),
                                             "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image",
                                                   "shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3),
                                                   "names": ["height", "width", "channel"]},
                "observation.state": {"dtype": "float32", "shape": (8,),
                                      "names": {"motors": ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]}},
                "action": {"dtype": "float32", "shape": (7,),
                           "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}},
            },
            image_writer_threads=4,
            image_writer_processes=2,
        )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # 设置最大步数
    if args.task_suite_name == "libero_spatial":
        max_steps = 250
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # 统计变量
    task_failure_stats = {}  # 每个任务的失败统计
    total_episodes = 0
    total_failures = 0
    total_recorded_failures = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # 初始化任务统计
        task_failures = 0
        task_successes = 0
        task_recorded_failures = 0

        # 执行多次试验
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            t = 0

            # 用于记录失败轨迹的数据
            failure_trajectory_frames = []
            replay_images = []

            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for simulator to stabilize
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed images
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    origin_img = img.copy()
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(origin_img)

                    processed_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    processed_wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        # Compute new action chunk
                        element = {
                            "observation/image": processed_img,
                            "observation/wrist_image": processed_wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model
                        output = client.infer(element)
                        action_chunk = output["actions"]
                        decoded_text = output["decoded_text"]

                        # replay_images.append(origin_img)

                        assert (
                                len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # Execute action
                    action = action_plan.popleft()

                    # 准备当前帧数据（无论成功失败都先收集）
                    if args.record_failure_trajectory:
                        # 准备状态数据
                        observation_state = np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]).astype(np.float32)

                        # 确保状态向量是 8 维
                        if len(observation_state) == 7:
                            observation_state = np.append(observation_state, 0.0)

                        # 准备动作数据
                        action_array = np.array(action.tolist(), dtype=np.float32)

                        # 保存帧数据到临时列表
                        frame_i = len(failure_trajectory_frames)
                        global_i = total_episodes * 10 ** 6 + frame_i  # 或自增全局计数器，确保唯一
                        frame_data = {
                            "image": origin_img if args.save_images else np.zeros(
                                (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), dtype=np.uint8),
                            "wrist_image": wrist_img if args.save_images else np.zeros(
                                (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), dtype=np.uint8),
                            "observation.state": observation_state,
                            "action": action_array,
                            # 时间戳应该是什么? 暂时还用不到
                            "task": f"{task_description} [FAILURE]",
                        }
                        failure_trajectory_frames.append(frame_data)
                    # Step environment
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        # 成功了，不记录轨迹，清空临时数据
                        failure_trajectory_frames = []
                        replay_images = []
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            total_episodes += 1

            # 如果没有成功，记录为失败轨迹
            if not done:
                task_failures += 1
                total_failures += 1

                print('\n failure_trajectory_frames长度')
                print(len(failure_trajectory_frames))
                # print('\n')
                # print(failure_dataset)
                # print('\n')

                # 记录失败轨迹到数据集
                if failure_dataset is not None and len(failure_trajectory_frames) > 0:
                    print(failure_dataset.features)
                    # 将所有帧添加到数据集
                    for frame_data in failure_trajectory_frames:
                        failure_dataset.add_frame(frame_data)

                    # 保存失败的 episode
                    failure_dataset.save_episode()
                    task_recorded_failures += 1
                    total_recorded_failures += 1

                    print(len(replay_images))

                    # 保存失败视频
                    if replay_images:
                        print('执行到视频')
                        failure_video_path = pathlib.Path(
                            args.video_out_path) / f"task_{task_id}_failure_{task_failures}.mp4"
                        try:
                            imageio.mimwrite(failure_video_path, [np.asarray(x) for x in replay_images], fps=10)
                        except Exception as e:
                            logging.error(f"保存失败视频出错: {e} -> {failure_video_path}")

            # 定期打印进度
            if (episode_idx + 1) % 10 == 0:
                current_failure_rate = task_failures / (episode_idx + 1) * 100
                logging.info(f"Task {task_id} - 进度: {episode_idx + 1}/{args.num_trials_per_task}, "
                             f"失败率: {current_failure_rate:.1f}% ({task_failures} failures)")

        # 记录任务统计
        task_failure_stats[task_id] = {
            "task_description": task_description,
            "total_trials": args.num_trials_per_task,
            "failures": task_failures,
            "successes": task_successes,
            "recorded_failures": task_recorded_failures,
            "failure_rate": task_failures / args.num_trials_per_task * 100
        }

        logging.info(f"Task {task_id} 完成:")
        logging.info(f"  描述: {task_description}")
        logging.info(f"  总试验: {args.num_trials_per_task}")
        logging.info(f"  失败: {task_failures} 次 ({task_failures / args.num_trials_per_task * 100:.1f}%)")
        logging.info(f"  成功: {task_successes} 次 ({task_successes / args.num_trials_per_task * 100:.1f}%)")
        logging.info(f"  记录的失败轨迹: {task_recorded_failures}")

    # 完成数据集创建
    if failure_dataset:
        print("正在整合失败轨迹数据集...")
        failure_dataset.consolidate(run_compute_stats=False)

        # 可选：推送到 Hugging Face Hub
        if args.push_to_hub:
            print("正在推送失败轨迹数据集到 Hugging Face Hub...")
            failure_dataset.push_to_hub(
                tags=["libero", "panda", "failure", "inference"],
                private=False,
                push_videos=args.save_images,
                license="apache-2.0",
            )

        print(f"失败轨迹数据集已保存到: {out_root / args.lerobot_repo_id}")

    # 打印最终统计报告
    print("\n" + "=" * 80)
    print("失败轨迹收集完成 - 最终统计报告")
    print("=" * 80)
    print(f"总执行次数: {total_episodes}")
    print(f"总失败次数: {total_failures}")
    print(f"总成功次数: {total_episodes - total_failures}")
    print(f"整体失败率: {total_failures / total_episodes * 100:.1f}%")
    print(f"记录的失败轨迹: {total_recorded_failures}")
    print()

    print("各任务详细统计:")
    print("-" * 120)
    print(f"{'Task ID':<8} {'失败次数':<10} {'成功次数':<10} {'失败率':<10} {'记录轨迹':<10} {'任务描述':<50}")
    print("-" * 120)

    for task_id, stats in task_failure_stats.items():
        print(f"{task_id:<8} {stats['failures']:<10} {stats['successes']:<10} "
              f"{stats['failure_rate']:<9.1f}% {stats['recorded_failures']:<10} "
              f"{stats['task_description']:<50}")

    print("-" * 120)

    # 保存统计报告到文件
    stats_file = pathlib.Path(args.video_out_path) / "failure_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_episodes": total_episodes,
                "total_failures": total_failures,
                "total_successes": total_episodes - total_failures,
                "overall_failure_rate": total_failures / total_episodes * 100,
                "recorded_failure_trajectories": total_recorded_failures
            },
            "task_stats": task_failure_stats
        }, f, indent=2, ensure_ascii=False)

    print(f"详细统计已保存到: {stats_file}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite"""
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
    tyro.cli(eval_libero_collect_failures)
