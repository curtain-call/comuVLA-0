import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class AlohaSimEnvironment(_environment.Environment):
    """An environment for an Aloha robot in simulation."""

    def __init__(self, task: str, obs_type: str = "pixels_agent_pos", seed: int = 0) -> None:
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        self._gym = gymnasium.make(task,
                                   obs_type=obs_type,
                                   max_episode_steps=500,

                                   )
        # observation_height = 480,
        # observation_width = 640,
        # visualization_height = 480,
        # visualization_width = 640

        self._last_obs = None
        self._done = True
        self._episode_reward = 0.0

        #  ================================================================

        self._success = False  # 新增：跟踪任务是否成功
        self._step_count = 0  # 新增：步数计数器

    @override
    def reset(self) -> None:
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False
        self._episode_reward = 0.0
        self._success = False  # 重置成功状态
        self._step_count = 0  # 重置步数计数器

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    @override
    def get_observation(self) -> dict:
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])

        # ============= debug ===========================================================
        # a = self._gym.step(action["actions"])
        # for item in a:
        #     print(item)
        # for k,v in info.items():
        #     print(f"{k}: {v}")

        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = terminated or truncated
        self._episode_reward = max(self._episode_reward, reward)

        # ===============================================================================
        if "is_success" in info and info["is_success"]:
            self._success = True
            print(f"The op is succeed")
        #
        if terminated:
            print(f"Terminated at step {self._step_count}")
        if truncated:
            print(f"Truncated at step {self._step_count}")
        # 更新终止条件
        self._step_count += 1  # 更新步数计数器
        # print(self._step_count)


    # ===================================================================================
    def was_successful(self) -> bool:
        """返回当前episode是否成功完成任务"""
        return self._success

    def get_progress(self) -> float:
        """返回当前episode的进度比例 (0.0-1.0)"""
        return min(1.0, self._step_count / self._max_episode_steps)
    # ===================================================================================

    def _convert_observation(self, gym_obs: dict) -> dict:
        img = gym_obs["pixels"]["top"]
        # print(img.shape)
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        # Convert axis order from [H, W, C] --> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return {
            "state": gym_obs["agent_pos"],
            "images": {"cam_high": img},
            "initial_images": {"cam_high": np.transpose(image_tools.convert_to_uint8(gym_obs["pixels"]["top"]), (2, 0, 1))}
        }
