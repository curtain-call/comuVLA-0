import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

libero_meta = lerobot_dataset.LeRobotDatasetMetadata(
    repo_id="lerobot/libero_spatial_image",
    root="/home/zhiyu/mzh/datasets/libero_spatial_image",
)

libero = lerobot_dataset.LeRobotDataset(
    repo_id="lerobot/libero_spatial_image",
    root="/home/zhiyu/mzh/datasets/libero_spatial_image",
    delta_timestamps={
        "action": [t / libero_meta.fps for t in range(50)]
    }
)

print(libero)