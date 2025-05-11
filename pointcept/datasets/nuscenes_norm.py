"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
import pickle

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class NuScenesNormDataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        # Use self.data_list populated by get_data_list via super().__init__
        info = self.data_list[idx % len(self.data_list)]

        # --- Load ref frame ---
        ref_lidar_path = os.path.join(self.data_root, "raw", info["lidar_path"])
        ref_points = np.fromfile(
            str(ref_lidar_path), dtype=np.float32, count=-1
        ).reshape([-1, 5])
        ref_coord = ref_points[:, :3]
        ref_strength = ref_points[:, 3:4] / 255.0
        ref_view_direction = ref_coord / (
            np.linalg.norm(ref_coord, axis=1, keepdims=True) + 1e-8
        )

        all_coords = [ref_coord]
        all_strengths = [ref_strength]
        all_view_directions = [ref_view_direction]

        # --- Load and process sweeps ---
        for sweep_info in info["sweeps"]:
            sweep_lidar_path = os.path.join(
                self.data_root, "raw", sweep_info["lidar_path"]
            )
            # Basic check if sweep file exists
            if not os.path.exists(sweep_lidar_path):
                continue  # Skip if sweep file is missing

            sweep_points = np.fromfile(
                str(sweep_lidar_path), dtype=np.float32, count=-1
            ).reshape([-1, 5])
            sweep_coord = sweep_points[:, :3]
            sweep_strength = sweep_points[:, 3:4] / 255.0

            transform_matrix = sweep_info["transform_matrix"]
            if transform_matrix is not None:
                sweep_coord_hom = np.hstack(
                    (
                        sweep_coord,
                        np.ones((sweep_coord.shape[0], 1), dtype=sweep_coord.dtype),
                    )
                )
                sweep_coord_in_ref_hom = sweep_coord_hom @ transform_matrix.T
                sweep_coord_in_ref = sweep_coord_in_ref_hom[:, :3]
                sweep_origin_in_ref = transform_matrix[:3, 3]
                sweep_view_direction_raw = sweep_coord_in_ref - sweep_origin_in_ref
                sweep_view_direction = sweep_view_direction_raw / (
                    np.linalg.norm(sweep_view_direction_raw, axis=1, keepdims=True)
                    + 1e-8
                )

                all_coords.append(sweep_coord_in_ref)
                all_strengths.append(sweep_strength)
                all_view_directions.append(sweep_view_direction)

        # --- Aggregate ---
        coord = np.concatenate(all_coords, axis=0)
        strength = np.concatenate(all_strengths, axis=0)
        view_direction = np.concatenate(all_view_directions, axis=0)

        # --- Load labels (only for ref points) ---
        if "gt_segment_path" in info.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", info["gt_segment_path"]
            )
            if os.path.exists(gt_segment_path):
                ref_segment = np.fromfile(
                    str(gt_segment_path), dtype=np.uint8, count=-1
                ).reshape([-1])
                ref_segment = np.vectorize(self.learning_map.__getitem__)(
                    ref_segment
                ).astype(np.int64)
                num_ref_points = ref_coord.shape[0]
                num_total_points = coord.shape[0]
                segment = np.full(num_total_points, self.ignore_index, dtype=np.int64)
                if len(ref_segment) == num_ref_points:  # Basic check
                    segment[:num_ref_points] = ref_segment
                else:  # Fallback if lengths mismatch
                    segment = (
                        np.ones((coord.shape[0],), dtype=np.int64) * self.ignore_index
                    )
            else:  # Fallback if segment file missing
                segment = np.ones((coord.shape[0],), dtype=np.int64) * self.ignore_index
        else:
            segment = np.ones((coord.shape[0],), dtype=np.int64) * self.ignore_index

        # --- Return final dictionary ---
        data_dict = dict(
            coord=coord.astype(np.float32),
            strength=strength.astype(np.float32),
            view_direction=view_direction.astype(np.float32),  # Added view direction
            segment=segment,
            name=self.get_data_name(idx),  # Use inherited method
        )
        data_dict["index_valid_keys"] = [
            "coord",
            "strength",
            "view_direction",
            "segment",
        ]
        # n_total = coord.shape[0]
        # assert strength.shape == (n_total, 1), f"Strength shape mismatch: {strength.shape}"
        # assert view_direction.shape == (n_total, 3), f"View direction shape mismatch: {view_direction.shape}"
        # assert segment.shape == (n_total,), f"Segment shape mismatch: {segment.shape}"
        # assert isinstance(coord, np.ndarray)
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
