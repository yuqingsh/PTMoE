import numpy as np
import os
import argparse
import time

from pointcept.datasets.nuscenes_norm import NuScenesNormDataset


def run_test(info_path, data_root, index, sweeps):
    """实例化数据集并加载一个样本进行基本检查"""

    print(f"--- Initializing Dataset ---")
    print(f"Info Path: {info_path}")
    print(f"Data Root: {data_root}")
    print(f"Sweeps: {sweeps}")
    print(f"Index: {index}")

    # 1. 初始化 Dataset
    try:
        # 使用 split='val' 来触发 pkl 加载 (需要对应的 val pkl 文件存在)
        # test_mode=False 确保调用 get_data 然后应用 transform (如果有)
        dataset = NuScenesNormDataset(
            split="val",  # 或者 'train'，取决于你的 info_path
            data_root=data_root,
            sweeps=sweeps,
            test_mode=False,
            # transform=None, # 暂时不加 transform，检查 get_data 的原始输出
            # 如果需要测试 transform，取消注释并传入配置
            # transform=[dict(type="ToTensor"), ...]
        )
        print(f"Dataset initialized. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"ERROR: Failed to initialize dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    if index >= len(dataset):
        print(f"ERROR: Index {index} out of bounds (0-{len(dataset)-1}).")
        return

    # 2. 加载指定索引的数据 (调用 __getitem__)
    print(f"\n--- Loading Sample {index} ---")
    start_time = time.time()
    try:
        data_dict = dataset[index]  # 使用 __getitem__
        load_time = time.time() - start_time
        print(f"Sample loaded successfully in {load_time:.4f} seconds.")
    except Exception as e:
        print(f"ERROR: Failed to load sample {index}: {e}")
        import traceback

        traceback.print_exc()
        return

    # 3. 基本检查
    print("\n--- Basic Checks ---")
    print(f"Keys in data_dict: {list(data_dict.keys())}")

    required_keys = ["coord", "strength", "view_direction", "segment", "name"]
    missing_keys = [key for key in required_keys if key not in data_dict]
    if missing_keys:
        print(f"ERROR: Missing required keys: {missing_keys}")
        return

    coord = data_dict["coord"]
    strength = data_dict["strength"]
    view_dir = data_dict["view_direction"]
    num_points = coord.shape[0]

    print(f"Sample Name (Token): {data_dict['name']}")
    print(f"Number of points (aggregated): {num_points}")
    print(f"Coord shape: {coord.shape}")  # Expected: (N, 3)
    print(f"Strength shape: {strength.shape}")  # Expected: (N, 1)
    print(f"View Dir shape: {view_dir.shape}")  # Expected: (N, 3)
    print(f"Segment shape: {data_dict['segment'].shape}")  # Expected: (N,)

    # 检查视线方向模长 (关键)
    if num_points > 0:
        # 注意：如果应用了 ToTensor 变换，这里会是 torch.Tensor
        # 如果没用 ToTensor，会是 NumPy 数组
        if isinstance(view_dir, np.ndarray):
            norms = np.linalg.norm(view_dir, axis=1)
            if np.allclose(norms, 1.0, atol=1e-5):
                print("View direction normalization check: PASSED (norms approx. 1.0)")
            else:
                print(
                    "View direction normalization check: WARNING (norms deviate from 1.0)"
                )
                print(f"  Norm Min/Max: {np.min(norms):.6f} / {np.max(norms):.6f}")
                if not np.allclose(norms, 1.0, atol=1e-4):
                    problem_indices = np.where(np.abs(norms - 1.0) > 1e-5)[0]
                    print(
                        f"Indices with problematic norms (first 5): {problem_indices[:5]}"
                    )
                    print(f"Problematic norms (first 5): {norms[problem_indices[:5]]}")
                    print(
                        f"Corresponding view directions (first 5):\n{view_dir[problem_indices[:5]]}"
                    )
                min_norm_idx = np.argmin(norms)  # 找到最小模长的索引
                min_norm_val = norms[min_norm_idx]
                print(f"\n--- Details for Minimum Norm Point ---")
                print(
                    f"Index of min norm point within aggregated array: {min_norm_idx}"
                )
                print(f"Minimum norm value: {min_norm_val:.6f}")
                print(f"View direction at min norm: {view_dir[min_norm_idx]}")
                print(f"Coordinate at min norm: {coord[min_norm_idx]}")
        # 如果是 Tensor, 需要用 PyTorch 的函数
        # elif torch.is_tensor(view_dir):
        #     norms = torch.linalg.norm(view_dir, dim=1)
        #     if torch.allclose(norms, torch.tensor(1.0, dtype=norms.dtype), atol=1e-5):
        #          print("View direction normalization check: PASSED (norms approx. 1.0)")
        #     else:
        #          print("View direction normalization check: WARNING (norms deviate from 1.0)")
        #          print(f"  Norm Min/Max: {torch.min(norms).item():.6f} / {torch.max(norms).item():.6f}")
        else:
            print("Cannot check norms for unknown data type:", type(view_dir))

    print("\n--- Test Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple test script for NuScenesNormDataset."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/share/home/u01015/ShenYuqing/data/nuScenes_pointcept_preprocessed",  # <--- 修改为你的路径
        help="Root directory of the nuScenes dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the sample to load and test.",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=10,  # <--- 确保与 .pkl 文件名匹配
        help="Number of sweeps used for the info file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",  # 使用哪个 split 的 info 文件
        choices=["train", "val", "test"],
        help="Dataset split to use (determines which info pkl to load).",
    )

    args = parser.parse_args()

    # 构建 info 文件路径
    info_path = os.path.join(
        args.data_root, "info", f"nuscenes_infos_{args.sweeps}sweeps_{args.split}.pkl"
    )

    if not os.path.exists(info_path):
        print(f"FATAL ERROR: Info file not found at {info_path}")
    elif not os.path.exists(args.data_root):
        print(f"FATAL ERROR: Data root not found at {args.data_root}")
    else:
        run_test(info_path, args.data_root, args.index, args.sweeps)
