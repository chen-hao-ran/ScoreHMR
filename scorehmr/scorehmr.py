import os
import numpy as np
import torch
import pickle
import cv2

from scorehmr.configs import model_config
from scorehmr.models.model_utils import load_pare, load_diffusion_model
from scorehmr.utils import StandarizeImageFeatures, prepare_smpl_params

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
NUM_SAMPLES = 1
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

class ScoreHMR:
    def __init__(self, device):
        self.model_cfg = model_config()
        self.device = device
        self.pare = load_pare(self.model_cfg.SMPL).to(self.device)
        self.pare.eval()
        self.img_feat_standarizer = StandarizeImageFeatures(
            backbone=self.model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
            use_betas=False,
            device=self.device,
        )
        self.extra_args = {
            "keypoint_guidance": True,
            "temporal_guidance": True,
            "use_default_ckpt": True,
            "device": self.device,
        }
        self.diffusion_model = load_diffusion_model(self.model_cfg, **self.extra_args)

    def iterate(self, data):
        for track_idx, mv_batch in data:
            batch = mv_batch[0]
            batch_size = batch["keypoints_2d"].size(0)

            # Get PARE image features.
            with torch.no_grad():
                pare_out = self.pare(batch["img"], get_feats=True)
            cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
            cond_feats = self.img_feat_standarizer(cond_feats)  # normalize image features

            # Iterative refinement with ScoreHMR.
            print(f'=> Running ScoreHMR for tracklet {track_idx+1}/{len(data)}')
            with torch.no_grad():
                dm_out = self.diffusion_model.sample(
                    mv_batch, cond_feats, batch_size=batch_size * NUM_SAMPLES
                )
            pred_smpl_params = prepare_smpl_params(
                dm_out['x_0'],
                num_samples = NUM_SAMPLES,
                use_betas = False,
                pred_betas=batch["pred_betas"],
            )
            smpl_out = self.diffusion_model.smpl(**pred_smpl_params, pose2rot=False)


