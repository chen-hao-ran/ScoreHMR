import os
import cv2
import json
import argparse
import warnings

import numpy as np
from click.core import batch
from dill import pickle
from sympy.codegen.ast import float32
from torch.utils.data import DataLoader
from triton.language import dtype

from demo.utils import *
from demo.dataset import MultiPeopleDataset
from scorehmr.utils import *
from scorehmr.configs import model_config
from scorehmr.utils.geometry import aa_to_rotmat
from scorehmr.models.model_utils import load_diffusion_model, load_pare
from scorehmr.utils.mesh_renderer import MeshRenderer

warnings.filterwarnings('ignore')


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
NUM_SAMPLES = 1
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, default="example_data/videos/breakdancing.mp4", help="Path of the input video.")
    parser.add_argument("--out_folder", type=str, default="demo_out/videos", help="Path to save the output video.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="If set, overwrite the 4D-Human tracklets.")
    parser.add_argument('--save_mesh', action='store_true', default=False, help='If set, save meshes to disk.')
    parser.add_argument("--fps", type=int, default=30, help="Frame rate to save the output video.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    OUT_DIR = args.out_folder

    video_name = os.path.basename(args.input_video)
    filename, _ = os.path.splitext(video_name)
    img_dir = f'{OUT_DIR}/images/{filename}'

    # Extract the frames of the input video.
    if not os.path.isdir(img_dir):
        video_to_frames(path=args.input_video, out_dir=img_dir)

    # Detects shots, runs 4D-Humans tracking, detects 2D with ViTPose, and prepares all necessary files.
    process_seq(out_root=OUT_DIR, seq=args.input_video, img_dir=img_dir, overwrite=args.overwrite)

    # Get the number of shots in the video.
    shots_path = f'{OUT_DIR}/shot_idcs/{filename}.json'
    with open(shots_path, "r") as f:
        shots_dict = json.load(f)
    num_shots = max(shots_dict.values())


    # ----------------------------------------
    ### Prepare ScoreHMR ###

    # Load config.
    model_cfg = model_config()

    # Load PARE model.
    pare = load_pare(model_cfg.SMPL).to(device)
    pare.eval()

    img_feat_standarizer = StandarizeImageFeatures(
        backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
        use_betas=False,
        device=device,
    )

    # Load diffusion model.
    extra_args = {
        "keypoint_guidance": True,
        "temporal_guidance": True,
        "use_default_ckpt": True,
        "device": device,
    }
    diffusion_model = load_diffusion_model(model_cfg, **extra_args)

    # ----------------------------------------


    # Set up renderer.
    renderer = MeshRenderer(model_cfg, faces=diffusion_model.smpl.faces)


    ## Iterate over shots in the video ##

    for shot_idx in range(num_shots+1):
        data_sources = {
            "images": f"{OUT_DIR}/images/{filename}",
            "tracks": f"{OUT_DIR}/track_preds/{filename}",
            "shots": f"{OUT_DIR}/shot_idcs/{filename}.json",
        }

        # Create dataset.
        dataset = MultiPeopleDataset(data_sources=data_sources, seq_name=filename, shot_idx=shot_idx)

        # Ignore shots with no tracklets or not long enough tracklets.
        if len(dataset.track_ids) <= 0 or dataset.num_imgs <= 20:
            continue

        B = len(dataset)     # number of people
        T = dataset.seq_len  # number of frames
        loader = DataLoader(dataset, batch_size=B, shuffle=False)

        obs_data = recursive_to(next(iter(loader)), device)

        num_tracks = obs_data["track_id"].size(0)
        pred_cam_t_all = torch.zeros((B, T, 3))
        pred_vertices_all = torch.zeros((B, T, 6890, 3))


        ## Iterate over tracklets ##

        for track_idx in range(num_tracks):
            # Get the the data for the current tracklet.
            batch = slice_dict(obs_data, track_idx)
            start_idx, end_idx = batch["track_interval"]
            start_idx = start_idx.item()
            end_idx = end_idx.item()

            # Keep only the valid data of the tracklet.
            batch = slice_dict_start_end(batch, start_idx, end_idx)
            batch_size = batch["keypoints_2d"].size(0)
            batch["img_size"] = (
                torch.Tensor(dataset.img_size)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .to(device)
            )
            batch["camera_center"] = batch["img_size"] / 2
            global_orient_rotmat = aa_to_rotmat(batch["init_root_orient"]).reshape(batch_size, -1, 3, 3)
            body_pose_rotmat = aa_to_rotmat(batch["init_body_pose"].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
            batch["pred_pose"] = torch.cat((global_orient_rotmat, body_pose_rotmat), dim=1)
            # focal_length = model_cfg.EXTRA.FOCAL_LENGTH * torch.ones(
            #     batch_size,
            #     2,
            #     device=device,
            #     dtype=batch["keypoints_2d"].dtype,
            # )
            focal_length = 2398.95251 * torch.ones(
                batch_size,
                2,
                device=device,
                dtype=batch["keypoints_2d"].dtype,
            )
            batch["focal_length"] = focal_length

            # Get PARE image features.
            with torch.no_grad():
                pare_out = pare(batch["img"], get_feats=True)
            cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
            cond_feats = img_feat_standarizer(cond_feats) # normalize image features

            # batch["init_cam_t"] = batch["pred_cam_t"]
            import pickle
            with open('example_data/extra/pred_cam_full.pkl', 'rb') as f:
                pred_cam_full = pickle.load(f)
            batch["init_cam_t"] = torch.from_numpy(pred_cam_full).to(device)
            batch["joints_2d"] = batch["keypoints_2d"][:, :, :2]
            batch["joints_conf"] = batch["keypoints_2d"][:, :, [2]]

            # Save batch, cond_feats
            data = {
                'batch': batch,
                'cond_feats': cond_feats,
            }
            output_dir = f'example_data/extra/phalp_out/{filename[:4]}/Camera{int(filename[5:]):02d}'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'phalp_out.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)

            # Pre-process mv_data
            mv_data = {}
            extra_batch = []
            extri = []
            R_new_R0_inv = []
            processed_extri = []
            view_num = 6
            frame_num = 1200
            with open('example_data/extra/test.pkl', 'rb') as f:
                annots = pickle.load(f)
            annot = annots[(int(filename[:4]) - 13) // 2]
            R0 = annot[0][0]['0']['extri'][:3, :3]
            R0_inv = np.linalg.inv(R0)
            for idx in range(view_num):
                with open(f'example_data/extra/phalp_out/{filename[:4]}/Camera{idx:02d}/phalp_out.pkl', 'rb') as f:
                    phalp_out = pickle.load(f)
                extra_batch.append(phalp_out['batch'])
                R = annot[idx][0]['0']['extri'][:3, :3]
                R_tensor = torch.from_numpy(np.dot(R, R0_inv)).to(device).to(torch.float32).requires_grad_()
                R_new_R0_inv.append(R_tensor)
                extri.append(torch.from_numpy(annot[idx][0]['0']['extri']).to(device).to(torch.float32).requires_grad_())
            for idx in range(view_num):
                # Tmp get processed extri
                R0 = annot[0][0]['0']['extri'][:3, :3]
                R0_inv = np.linalg.inv(R0)
                t0 = annot[0][0]['0']['extri'][:3, 3]
                R_new = annot[idx][0]['0']['extri'][:3, :3]
                t_new = annot[idx][0]['0']['extri'][:3, 3]
                processed_R = np.dot(R_new, R0_inv)
                processed_t = t_new - np.dot(np.dot(R_new, R0_inv), t0)
                extri_new = np.zeros((4, 4))
                extri_new[:3, :3] = processed_R
                extri_new[:3, 3] = processed_t
                extri_new[3, 3] = 1.
                processed_extri.append(torch.from_numpy(extri_new).to(device).to(torch.float32))
            mv_data['extra_batch'] = extra_batch
            mv_data['R_new_R0_inv'] = R_new_R0_inv
            mv_data['extri'] = extri
            mv_data['processed_extri'] = processed_extri

            # Iterative refinement with ScoreHMR.
            print(f'=> Running ScoreHMR for tracklet {track_idx+1}/{num_tracks}')
            with torch.no_grad():
                dm_out = diffusion_model.sample(
                    batch, cond_feats, mv_data, batch_size=batch_size * NUM_SAMPLES
                )

            pred_smpl_params = prepare_smpl_params(
                dm_out['x_0'],
                num_samples = NUM_SAMPLES,
                use_betas = False,
                pred_betas=batch["pred_betas"],
            )
            smpl_out = diffusion_model.smpl(**pred_smpl_params, pose2rot=False)
            pred_cam_t_all[track_idx, start_idx:end_idx] = dm_out['camera_translation'].cpu()
            pred_vertices_all[track_idx, start_idx:end_idx] = smpl_out.vertices.cpu()

            # Save smpl_out
            for frame in range(frame_num):
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
                output_dir = f'output/results/{filename[:4]}'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{frame:05d}.pkl')
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)

                # Project vertices
                # vertices = smpl_out.vertices[frame].cpu().numpy()
                # vertices += dm_out['camera_translation'][frame].cpu().numpy()
                # intri = annots[0][0][0]['0']['intri']
                # v2d = np.dot(vertices, intri.T)
                # v2d[:, :2] = (v2d[:, :2] / v2d[:, 2:])
                # h, w = 1536, 2048
                # # image = np.zeros((h, w, 3), dtype=np.uint8)
                # image = cv2.imread(f'demo_out/videos/images/0013_00/{(frame+1):06d}.jpg')
                # for v in v2d:
                #     cv2.circle(image, (int(v[0]), int(v[1])), 3, (0, 0, 255), -1)
                # output_dir = 'output/check/vertices_proj'
                # os.makedirs(output_dir, exist_ok=True)
                # output_path = os.path.join(output_dir, f'{frame:05d}.png')
                # cv2.imwrite(output_path, image)
                #
                # # Save vertices
                # vertices = smpl_out.vertices[frame].cpu().numpy()
                # output_dir = 'output/check/vertices'
                # os.makedirs(output_dir, exist_ok=True)
                # output_path = os.path.join(output_dir, f'{frame:05d}.txt')
                # np.savetxt(output_path, vertices)

            # Save meshes as OBJ files.
            if args.save_mesh:
                verts = smpl_out.vertices.cpu().numpy()
                cam_t = dm_out['camera_translation'].cpu().numpy()
                person_id = str(batch['track_id'].item()).zfill(3)
                tmesh_path = f"{OUT_DIR}/mesh_output/{filename}/{person_id}"
                print(f'=> Saving mesh files for {person_id} in {tmesh_path}')
                os.makedirs(tmesh_path, exist_ok=True)
                for ii, (vvv, ttt) in enumerate(zip(verts, cam_t)):
                    tmesh = renderer.vertices_to_trimesh(vvv, ttt, LIGHT_BLUE)
                    frame_id = str(ii + start_idx + 1).zfill(6)
                    tmesh.export(f'{tmesh_path}/{frame_id}.obj')


        # # Save output video.
        # frame_list = create_visuals(
        #     renderer,
        #     pred_vertices_all.numpy(),
        #     pred_cam_t_all.numpy(),
        #     dataset.sel_img_paths,
        # )
        #
        # height, width, _ = frame_list[0].shape
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # os.makedirs(f"{OUT_DIR}", exist_ok=True)
        # video_writer = cv2.VideoWriter(
        #     f"{OUT_DIR}/{filename}_{shot_idx}.mp4",
        #     fourcc,
        #     args.fps,
        #     (width, height),
        # )
        # for frame in frame_list:
        #     video_writer.write(cv2.convertScaleAbs(frame))
        # video_writer.release()


if __name__ == "__main__":
    main()