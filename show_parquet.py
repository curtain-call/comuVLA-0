import pyarrow.parquet as pq

path = "/home/zhiyu/mzh/datasets/libero_spatial_image/data/chunk-000/episode_000000.parquet"

file = pq.ParquetFile(path)

# table = file.read()

for i in range(file.num_row_groups):
    row_group = file.read_row_group(i)
    for j in range(row_group.num_rows):
        row = row_group[j]
        print(type(row))
