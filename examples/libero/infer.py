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

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@dataclasses.dataclass
class KArgs:
    '''
    配置类
    '''
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = (
        "libero_spatial"
    )
    num_steps_wait: int = 10
    num_trials_per_task: int = 1

    config:str="pi0_libero"

    video_out_path: str = "data/libero-spatial-failure"
    seed = 7

    log_failed: bool = True
    failed_out_dir: str = "data/libero_failed"

    record_trajectory: bool = True
    lerobot_repo_id: str = "maozihao/libero_failure_dataset"  # LeRobot 数据集 ID
    save_images: bool = True
    fps: int = 10
    push_to_hub: bool = False


def eval_libero_with_lerobot_recording(args: KArgs):
    """使用 LeRobot 标准 API 记录轨迹的评估函数"""
    np.random.seed(args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # 初始化 LeRobot 数据集（使用官方 API）
    lerobot_dataset = None
    if args.record_trajectory:
        # 清理现有数据集
        output_path = HF_LEROBOT_HOME / args.lerobot_repo_id
        if output_path.exists():
            shutil.rmtree(output_path)
        
        # 创建 LeRobot 数据集，使用标准字段名
        lerobot_dataset = LeRobotDataset.create(
            repo_id=args.lerobot_repo_id,
            robot_type="panda",
            fps=args.fps,
            features={
                # 注意：使用 LeRobot 标准字段名
                "observation.images.image": {  # 主视角图像
                    "dtype": "image",
                    "shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.wrist_image": {  # 腕部图像
                    "dtype": "image", 
                    "shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {  # 机器人状态
                    "dtype": "float32",
                    "shape": (8,),  # 3位置 + 3轴角 + 1夹爪 + 1额外
                    "names": ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"],
                },
                "action": {  # 动作
                    "dtype": "float32",
                    "shape": (7,),  # 6DOF + 夹爪
                    "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                },
            },
            image_writer_threads=4,
            image_writer_processes=2,
        )

    # 设置最大步数
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
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

    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\n{task_description}")
            
            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            episode_success = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            
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

                        print(f"Step {t}: {decoded_text}")
                        replay_images.append(origin_img)

                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # Execute action
                    action = action_plan.popleft()
                    
                    # 记录到 LeRobot 数据集（使用官方 API）
                    if lerobot_dataset:
                        # 准备状态数据
                        observation_state = np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]).astype(np.float32)
                        
                        # 确保状态向量是 8 维
                        if len(observation_state) == 7:
                            observation_state = np.append(observation_state, 0.0)  # 添加额外维度
                        
                        # 准备动作数据
                        action_array = np.array(action.tolist(), dtype=np.float32)
                        
                        # 添加帧到数据集
                        frame_data = {
                            "observation.images.image": origin_img if args.save_images else np.zeros((LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), dtype=np.uint8),
                            "observation.images.wrist_image": wrist_img if args.save_images else np.zeros((LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3), dtype=np.uint8),
                            "observation.state": observation_state,
                            "action": action_array,
                        }
                        
                        lerobot_dataset.add_frame(frame_data)

                    # Step environment
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        episode_success = True
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # 保存 episode 到 LeRobot 数据集
            if lerobot_dataset:
                lerobot_dataset.save_episode(task=task_description)

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # 完成数据集创建
    if lerobot_dataset:
        print("正在整合数据集...")
        lerobot_dataset.consolidate(run_compute_stats=False)
        
        # 可选：推送到 Hugging Face Hub
        if args.push_to_hub:
            print("正在推送到 Hugging Face Hub...")
            lerobot_dataset.push_to_hub(
                tags=["libero", "panda", "inference"],
                private=False,
                push_videos=args.save_images,
                license="apache-2.0",
            )
        
        print(f"LeRobot 数据集已保存到: {HF_LEROBOT_HOME / args.lerobot_repo_id}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")



def eval_libero_with_text(args: KArgs):
    np.random.seed(args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # modify task_description for input prompt
        # task_description = f"How to determine whether task '{task_description}' succeeded or failed?"
        # task_description = f"What should the result of the task '{task_description}' based on your observation"
        # task_description = "What is on the table? \n"
        # task_description = "Caption: "

        # task_description = "What is on the table? \n"
        # task_description = "Question: What is on the table?\nAnswer:"

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\n{task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    origin_img = img.copy()
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        output = client.infer(element)
                        action_chunk = output["actions"]
                        decoded_text = output["decoded_text"]

                        # print(decoded_text[0][0])

                        # Save preprocessed image for replay video
                        # replay_image = draw_text_on_image(origin_img,decoded_text[0][0])
                        print(decoded_text)

                        replay_images.append(origin_img)

                        assert (
                                len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])


                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero-degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def draw_text_on_image(image, text, font_size=24):
    # image: numpy array (H, W, 3)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    # 文字位置
    margin = 10
    draw.text((margin, margin), text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_with_trajectory_recording)


