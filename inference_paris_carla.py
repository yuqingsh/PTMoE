import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.utils.config import Config
from pointcept.utils.visualization import save_point_cloud
from pointcept.utils.comm import get_world_size
from pointcept.datasets.utils import point_collate_fn
import pointcept.datasets.paris_carla
from pointcept.utils.logger import get_root_logger
from torch.amp import autocast

from tensorboardX import SummaryWriter

data_config = dict(
    type="ParisCarlaDataset",
    split="Train",
    data_root="data/Paris-CARLA-3d",
    transform=[
        dict(
            type="RandomDropout",
            dropout_ratio=0.95,
            dropout_application_ratio=1.0,
        ),
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "name"),
            feat_keys=("coord", "strength"),
        ),
    ],
    test_mode=True,
)

model_config = dict(
    type="PT-v3m1",
    in_channels=4,
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
)

checkpoint = "exp/ptv3/pretrained_weight_downloaded/model_best.pth"
keywords = "backbone."
replacement = ""
idx = 300

if __name__ == "__main__":
    logger = get_root_logger()
    model = build_model(model_config).cuda()

    checkpoint = torch.load(
        checkpoint, map_location=lambda storage, loc: storage.cuda()
    )
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("module."):
            key = "module." + key  # xxx.xxx -> module.xxx.xxx
        if keywords in key:
            key = key.replace(keywords, replacement)
        if get_world_size() == 1:
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
        weight[key] = value
    load_state_info = model.load_state_dict(weight, strict=False)
    logger.info(f"Load state dict info: {load_state_info}")
    dataset = build_dataset(data_config)
    data = dataset[idx]
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda(non_blocking=True)

    output = model(data)
    feat = output.feat
    feat = feat - feat.mean(dim=-2, keepdim=True)
    u, s, v = torch.pca_lowrank(feat, center=False, q=3)
    projection = feat @ v
    min_val = projection.min(dim=0, keepdim=True)[0]
    max_val = projection.max(dim=0, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    pca_color = (projection - min_val) / div
    os.makedirs("test/vis_pca", exist_ok=True)
    save_point_cloud(
        coord=output.coord,
        color=pca_color,
        file_path=f"test/vis_pca/pca_color{idx}.ply",
    )
    vertices = output.coord.unsqueeze(0).clone().detach().cpu().numpy() / 3
    colors = pca_color.unsqueeze(0).clone().detach().cpu().numpy() * 255

    writer = SummaryWriter("test/vis_pca")
    writer.add_mesh(
        f"train/{data['name']}",
        vertices=vertices,
        colors=colors,
        global_step=1,
    )
