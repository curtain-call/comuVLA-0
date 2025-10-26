import os
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, Image

root1 = "/home/zhiyu/mzh/datasets/bridge2lerobot/data"

# correct schema
features = Features({
        **{
                f"observation.images.image_{i}": Image()
                for i in range(4)
            },
    "camera_present": Sequence(Value("bool"),length=4),
    "observation.state": Sequence(Value("float32"), length=7),
    "action": Sequence(Value("float32"), length=7),
    "timestamp": Value("float32"),
    "frame_index": Value("int64"),
    "episode_index": Value("int64"),
    "index": Value("int64"),
    "task_index": Value("int64"),
    "atomic.valid": Value("bool"),
            "atomic.segment_id": Value("int32"),
            "atomic.frame_in_segment": Value("int32"),
            "atomic.segment_start": Value("bool"),
            "atomic.cur_translation_idx": Value("int32"),
            "atomic.cur_rotation_idx": Value("int32"),
            "atomic.cur_gripper_idx": Value("int32"),
            "atomic.cur_duration_idx": Value("int32"),
            "atomic.raw_frame_index": Value("int32"),
})



def fix_file(path: str):
    # read old
    table = pq.read_table(path)

    # strip metadata
    schema = table.schema.remove_metadata()
    table = table.cast(schema)

    # make HF Dataset and recast
    ds = Dataset(table).cast(features)

    # overwrite in place
    tmp_path = path + ".tmp"
    ds.to_parquet(tmp_path)
    os.replace(tmp_path, path)

    print(f"fixed {path}")

# walk through all shards
for root, _, files in os.walk(root1):
    for fname in files:
        if fname.endswith(".parquet"):
            fix_file(os.path.join(root, fname))

print("\n all parquet shards in smol-libero2/data have been rewritten with Sequence schema ;0")