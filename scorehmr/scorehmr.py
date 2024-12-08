import os
import numpy as np
import torch
import pickle
import cv2

from demo.utils import slice_dict, slice_dict_start_end

from scorehmr.configs import model_config
from scorehmr.models.model_utils import load_pare, load_diffusion_model
from scorehmr.utils import StandarizeImageFeatures, recursive_to, prepare_smpl_params
from scorehmr.utils.geometry import aa_to_rotmat

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
NUM_SAMPLES = 1
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

class ScoreHMR:
    def __init__(self, args, device, datasets, obs_data, num_tracks):
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
        self.datasets = datasets
        self.obs_data = obs_data
        self.num_tracks = num_tracks

    def iterate(self, args):
        for track_idx in range(self.num_tracks):
            # Get batch & extra_batch
            batch = self.get_batch(0, track_idx)
            extra_batch = []
            for view_idx in range(args.view_num):
                extra_batch.append(self.get_batch(view_idx, track_idx))
            batch_size = batch["keypoints_2d"].size(0)

            # Get mv_data
            mv_data = {}
            extri = []
            R_new_R0_inv = []
            with open('example_data/extra/test.pkl', 'rb') as f:
                annots = pickle.load(f)
            annot = annots[(int(args.seq_name) - 13) // 2]
            R0 = annot[0][0]['0']['extri'][:3, :3]
            R0_inv = np.linalg.inv(R0)
            for idx in range(args.view_num):
                R = annot[idx][0]['0']['extri'][:3, :3]
                R_tensor = torch.from_numpy(np.dot(R, R0_inv)).to(self.device).to(torch.float32).requires_grad_()
                R_new_R0_inv.append(R_tensor)
                extri.append(torch.from_numpy(annot[idx][0]['0']['extri']).to(self.device).to(torch.float32).requires_grad_())
            mv_data['extra_batch'] = extra_batch
            mv_data['R_new_R0_inv'] = R_new_R0_inv
            mv_data['extri'] = extri

            # Get PARE image features.
            with torch.no_grad():
                pare_out = self.pare(batch["img"], get_feats=True)
            cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
            cond_feats = self.img_feat_standarizer(cond_feats)  # normalize image features

            # Iterative refinement with ScoreHMR.
            print(f'=> Running ScoreHMR for tracklet {track_idx+1}/{self.num_tracks}')
            with torch.no_grad():
                dm_out = self.diffusion_model.sample(
                    batch, cond_feats, mv_data, batch_size=batch_size * NUM_SAMPLES
                )
            pred_smpl_params = prepare_smpl_params(
                dm_out['x_0'],
                num_samples = NUM_SAMPLES,
                use_betas = False,
                pred_betas=batch["pred_betas"],
            )
            smpl_out = self.diffusion_model.smpl(**pred_smpl_params, pose2rot=False)

            # Save smpl_out
            for frame in range(args.frame_num):
                # Betas
                betas = smpl_out.body_pose.cpu().numpy()[frame].reshape(1, -1)
                # Body pose
                body_pose_mat = smpl_out.body_pose.cpu().numpy()[frame]
                body_pose_vec = []
                for mat in body_pose_mat:
                    vec = cv2.Rodrigues(mat)[0].T.reshape(3)
                    body_pose_vec.append(vec)
                body_pose_vec = np.array(body_pose_vec)
                body_pose = body_pose_vec.reshape(-1)
                # Global pose
                global_orient_mat = smpl_out.global_orient.cpu().numpy()[frame].reshape(3, 3)
                global_orient_vec = cv2.Rodrigues(global_orient_mat)[0].T
                global_orient = global_orient_vec.reshape(-1)
                # Pose
                pose = np.concatenate((global_orient_vec, body_pose_vec), axis=0).reshape(-1)
                # Transl
                transl = dm_out['camera_translation'].cpu().numpy()[frame]

                person00 = {
                    'betas': betas,
                    'pose': pose,
                    'body_pose': body_pose,
                    'global_orient': global_orient,
                    'transl': transl,
                }
                data = {
                    'person00': person00,
                }
                output_dir = f'output/results/{args.seq_name}'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{frame:05d}.pkl')
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)

    def get_batch(self, view_idx, track_idx):
        # Get the the data for the current tracklet.
        batch = slice_dict(self.obs_data[view_idx], track_idx)
        start_idx, end_idx = batch["track_interval"]
        start_idx = start_idx.item()
        end_idx = end_idx.item()

        # Keep only the valid data of the tracklet.
        batch = slice_dict_start_end(batch, start_idx, end_idx)
        batch_size = batch["keypoints_2d"].size(0)
        batch["img_size"] = (
            torch.Tensor(self.datasets[view_idx].img_size)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(self.device)
        )
        batch["camera_center"] = batch["img_size"] / 2
        global_orient_rotmat = aa_to_rotmat(batch["init_root_orient"]).reshape(batch_size, -1, 3, 3)
        body_pose_rotmat = aa_to_rotmat(batch["init_body_pose"].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
        batch["pred_pose"] = torch.cat((global_orient_rotmat, body_pose_rotmat), dim=1)
        focal_length = 2398.95251 * torch.ones(
            batch_size,
            2,
            device=self.device,
            dtype=batch["keypoints_2d"].dtype,
        )
        batch["focal_length"] = focal_length
        # Todo: 修改t的读取路径
        import pickle
        with open('example_data/extra/pred_cam_full.pkl', 'rb') as f:
            pred_cam_full = pickle.load(f)
        batch["init_cam_t"] = torch.from_numpy(pred_cam_full).to(self.device).requires_grad_()
        batch["joints_2d"] = batch["keypoints_2d"][:, :, :2]
        batch["joints_conf"] = batch["keypoints_2d"][:, :, [2]]

        return batch
