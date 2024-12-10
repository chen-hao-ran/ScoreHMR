import os
import numpy as np
import cv2
import pickle
import torch
from demo.dataset import MultiPeopleDataset
from demo.utils import slice_dict_start_end, slice_dict
from torch.utils.data import DataLoader

from scorehmr.utils import recursive_to
from scorehmr.utils.geometry import aa_to_rotmat


def get_batch(view_idx, track_idx, obs_data, datasets, device):
    # Get the the data for the current tracklet.
    batch = slice_dict(obs_data[view_idx], track_idx)
    start_idx, end_idx = batch["track_interval"]
    start_idx = start_idx.item()
    end_idx = end_idx.item()

    # Keep only the valid data of the tracklet.
    batch = slice_dict_start_end(batch, start_idx, end_idx)
    batch_size = batch["keypoints_2d"].size(0)
    batch["img_size"] = (
        torch.Tensor(datasets[view_idx].img_size)
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(device)
    )
    batch["camera_center"] = batch["img_size"] / 2
    global_orient_rotmat = aa_to_rotmat(batch["init_root_orient"]).reshape(batch_size, -1, 3, 3)
    body_pose_rotmat = aa_to_rotmat(batch["init_body_pose"].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
    batch["pred_pose"] = torch.cat((global_orient_rotmat, body_pose_rotmat), dim=1)
    focal_length = 2398.95251 * torch.ones(
        batch_size,
        2,
        device=device,
        dtype=batch["keypoints_2d"].dtype,
    )
    batch["focal_length"] = focal_length
    # Todo: 修改t的读取路径
    import pickle
    with open('example_data/extra/pred_cam_full.pkl', 'rb') as f:
        pred_cam_full = pickle.load(f)
    batch["init_cam_t"] = torch.from_numpy(pred_cam_full).to(device).requires_grad_()
    batch["joints_2d"] = batch["keypoints_2d"][:, :, :2]
    batch["joints_conf"] = batch["keypoints_2d"][:, :, [2]]

    return batch

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR = 'demo_out/videos'
    view_num = 6
    seq_name = '0019'
    datasets = []
    obs_data = []
    data = {}
    for view_idx in range(view_num):
        data_sources = {
            "images": f"{OUT_DIR}/images/{seq_name}_{view_idx:02d}",
            "tracks": f"{OUT_DIR}/track_preds/{seq_name}_{view_idx:02d}",
            "shots": f"{OUT_DIR}/shot_idcs/{seq_name}_{view_idx:02d}.json",
        }
        dataset = MultiPeopleDataset(data_sources=data_sources, seq_name=seq_name, shot_idx=0)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        datasets.append(dataset)
        obs_data.append(recursive_to(next(iter(loader)), device))
        num_tracks = obs_data[0]["track_id"].size(0)
        data["num_tracks"] = num_tracks

    for track_idx in range(num_tracks):
        # Get batch & extra_batch
        batch = get_batch(0, track_idx, obs_data, datasets, device)
        extra_batch = []
        for view_idx in range(view_num):
            extra_batch.append(get_batch(view_idx, track_idx, obs_data, datasets, device))
        batch_size = batch["keypoints_2d"].size(0)

        # Get mv_data
        mv_data = {}
        extri = []
        R_new_R0_inv = []
        with open('example_data/extra/test.pkl', 'rb') as f:
            annots = pickle.load(f)
        annot = annots[(int(seq_name) - 13) // 2]
        R0 = annot[0][0]['0']['extri'][:3, :3]
        R0_inv = np.linalg.inv(R0)
        for idx in range(view_num):
            R = annot[idx][0]['0']['extri'][:3, :3]
            R_tensor = torch.from_numpy(np.dot(R, R0_inv)).to(device).to(torch.float32).requires_grad_()
            R_new_R0_inv.append(R_tensor)
            extri.append(torch.from_numpy(annot[idx][0]['0']['extri']).to(device).to(torch.float32).requires_grad_())
        mv_data['extra_batch'] = extra_batch
        mv_data['R_new_R0_inv'] = R_new_R0_inv
        mv_data['extri'] = extri

        data["batch"] = batch
        data["extra_batch"] = extra_batch
        data["mv_data"] = mv_data

    data["seq_name"] = "0013"
    output_dir = "example_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, seq_name + ".pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)




