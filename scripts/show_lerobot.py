import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def visualize_parquet_images(parquet_path: str, frame_idx: int = 0, save_path: str = None):
    """从 parquet 文件中读取并可视化图像
    
    Args:
        parquet_path: parquet 文件路径
        frame_idx: 要可视化的帧索引（默认第0帧）
        save_path: 保存图像的路径（可选）
    """
    # 读取 parquet 文件
    parquet_file = pq.ParquetFile(parquet_path)
    
    # 查看 schema
    print("Schema:")
    print(parquet_file.schema)
    print(f"\n总帧数: {parquet_file.metadata.num_rows}")
    print(f"总列数: {parquet_file.metadata.num_columns}")
    
    # 读取数据
    table = parquet_file.read()
    df = table.to_pandas()
    
    # 找出所有图像列
    image_cols = [col for col in df.columns if 'image' in col.lower() and 'observation' in col]
    print(f"\n找到 {len(image_cols)} 个图像列: {image_cols}")
    
    if len(image_cols) == 0:
        print("未找到图像列！")
        return
    
    if frame_idx >= len(df):
        print(f"帧索引 {frame_idx} 超出范围（总帧数: {len(df)}）")
        return
    
    # 创建子图
    n_images = len(image_cols)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    if n_images == 1:
        axes = [axes]
    
    # 显示每个相机的图像
    for idx, col in enumerate(image_cols):
        img_data = df[col].iloc[frame_idx]
        
        # 处理不同的数据格式
        if img_data is None:
            print(f"{col}: 数据为 None")
            axes[idx].text(0.5, 0.5, 'No Image', ha='center', va='center')
            axes[idx].set_title(f"{col}\n(No Data)")
        else:
            # LeRobot Image 特征格式：{'path': '...', 'bytes': None}
            if isinstance(img_data, dict):
                if 'path' in img_data and img_data['path']:
                    # 图像存储在外部文件中
                    import os
                    img_path = img_data['path']
                    # 如果是相对路径，需要相对于 parquet 文件所在目录
                    if not os.path.isabs(img_path):
                        parquet_dir = os.path.dirname(os.path.abspath(parquet_path))
                        # 通常图像在 parquet 文件的上两级目录的 videos 文件夹中
                        dataset_root = os.path.dirname(os.path.dirname(parquet_dir))
                        img_path = os.path.join(dataset_root, img_path)
                    
                    if os.path.exists(img_path):
                        img_array = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
                        print(f"{col}: 从文件加载 {img_path}")
                    else:
                        print(f"{col}: 文件不存在 {img_path}")
                        axes[idx].text(0.5, 0.5, f'File not found\n{img_path}', 
                                     ha='center', va='center', fontsize=8)
                        axes[idx].set_title(f"{col}\n(File Missing)")
                        axes[idx].axis('off')
                        continue
                elif 'bytes' in img_data and img_data['bytes']:
                    # 图像以字节形式内嵌
                    from io import BytesIO
                    img_array = np.array(Image.open(BytesIO(img_data['bytes'])).convert('RGB'), dtype=np.uint8)
                    print(f"{col}: 从内嵌字节加载")
                else:
                    print(f"{col}: 字典格式但无有效数据 {img_data}")
                    axes[idx].text(0.5, 0.5, 'Invalid Dict', ha='center', va='center')
                    axes[idx].set_title(f"{col}\n(Invalid)")
                    axes[idx].axis('off')
                    continue
            # 直接是 numpy 数组或列表
            elif isinstance(img_data, (list, np.ndarray)):
                img_array = np.array(img_data, dtype=np.uint8)
                print(f"{col}: 直接数组格式")
            # PyArrow 类型
            elif hasattr(img_data, 'as_py'):
                img_array = np.array(img_data.as_py(), dtype=np.uint8)
                print(f"{col}: PyArrow 格式")
            else:
                print(f"{col}: 未知格式 {type(img_data)}")
                img_array = np.array(img_data, dtype=np.uint8)
            
            print(f"  shape={img_array.shape}, dtype={img_array.dtype}, "
                  f"min={img_array.min()}, max={img_array.max()}")
            
            # 显示图像
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                axes[idx].imshow(img_array)
            elif len(img_array.shape) == 2:
                axes[idx].imshow(img_array, cmap='gray')
            else:
                print(f"  警告: 无法显示形状为 {img_array.shape} 的图像")
                axes[idx].text(0.5, 0.5, f'Invalid shape\n{img_array.shape}', 
                             ha='center', va='center')
            
            axes[idx].set_title(f"{col}\n{img_array.shape}")
        
        axes[idx].axis('off')
    
    plt.suptitle(f'Frame {frame_idx} from {parquet_path}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图像已保存到: {save_path}")
    
    plt.show()
    
    # 显示其他数据信息
    print("\n其他列信息:")
    for col in df.columns:
        if 'image' not in col.lower():
            try:
                val = df[col].iloc[frame_idx]
                if isinstance(val, (list, np.ndarray)):
                    val_array = np.array(val)
                    print(f"  {col}: shape={val_array.shape}, value={val_array}")
                else:
                    print(f"  {col}: {val}")
            except Exception as e:
                print(f"  {col}: 无法读取 ({e})")


def main():
    parser = argparse.ArgumentParser(description="可视化 LeRobot parquet 文件中的图像")
    parser.add_argument("parquet_path", type=str, 
                       help="parquet 文件路径")
    parser.add_argument("--frame", type=int, default=0,
                       help="要可视化的帧索引（默认: 0）")
    parser.add_argument("--save", type=str, default=None,
                       help="保存图像的路径（可选）")
    
    args = parser.parse_args()
    
    visualize_parquet_images(args.parquet_path, args.frame, args.save)


if __name__ == "__main__":
    main()