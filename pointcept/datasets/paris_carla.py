import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement

from .builder import DATASETS, build_dataset
from pointcept.utils.logger import get_root_logger
from pointcept.datasets.transform import Compose


@DATASETS.register_module()
class ParisCarlaDataset(Dataset):
    def __init__(
        self,
        data_root="data/paris-carla-3d",
        split="train",
        transform=None,
        test_mode=False,
        ignore_index=-1,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.loop = loop if not test_mode else 1

        self.data_list = self.get_data_list()

        logger = get_root_logger()
        logger.info("Totally {} x {} samples.".format(len(self.data_list), self.loop))

    def get_data_list(self):
        split_path = os.path.join(self.data_root, self.split)
        return glob.glob(os.path.join(split_path, "*.ply"))

    def load_ply_data(self, file_path):
        try:
            ply_data = PlyData.read(file_path)
            vertex = ply_data["vertex"]
        except Exception as e:
            logger = get_root_logger()
            logger.error(f"Error reading PLY file {file_path}: {e}")
            return None

        data_dict = {}

        try:
            x = vertex["x"]
            y = vertex["y"]
            z = vertex["z"]
            data_dict["coord"] = np.stack([x, y, z], axis=1).astype(np.float32)
        except KeyError as e:
            raise KeyError(
                f"Missing coordinate property '{e}' in PLY file: {file_path}."
            )
        except Exception as e:
            raise RuntimeError(f"Error loading coordinates from {file_path}: {e}")

        num_points = len(data_dict["coord"])

        try:
            r = vertex["red"]
            g = vertex["green"]
            b = vertex["blue"]
            data_dict["color"] = np.stack([r, g, b], axis=1).astype(np.float32)
        except KeyError:
            pass

        try:
            data_dict["strength"] = (
                vertex["intensity"].reshape(-1, 1).astype(np.float32)
            )
        except KeyError:
            pass

        try:
            data_dict["semantic"] = vertex["semantic"].astype(np.uint32)
        except KeyError:
            pass

        try:
            data_dict["instance"] = vertex["instance"].astype(np.uint32)
        except KeyError:
            pass

        data_dict["name"] = os.path.basename(file_path)
        data_dict["split"] = self.split

        return data_dict

    def __getitem__(self, idx):

        file_path = self.data_list[idx % len(self.data_list)]
        data_dict = self.load_ply_data(file_path=file_path)

        if data_dict is None:
            logger = get_root_logger()
            logger.warning(f"Failed to load data for index {idx}.")
            return {}

        if self.transform:
            data_dict = self.transform(data_dict=data_dict)

        return data_dict

    def __len__(self):
        return len(self.data_list) * self.loop
