import argparse
import torch
from demo.dataset import MultiPeopleDataset
from torch.utils.data import DataLoader

from scorehmr.scorehmr import ScoreHMR
from scorehmr.utils import recursive_to

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="demo_out/videos", help="Path to save the output video.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="If set, overwrite the 4D-Human tracklets.")
    parser.add_argument('--save_mesh', action='store_true', default=False, help='If set, save meshes to disk.')
    parser.add_argument("--fps", type=int, default=30, help="Frame rate to save the output video.")
    parser.add_argument("--frame_num", type=int, default=1200, help="")
    parser.add_argument("--view_num", type=int, default=6, help="")
    parser.add_argument("--seq_name", type=str, default="0013", help="")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process Dataset
    OUT_DIR = args.out_folder
    datasets = []
    obs_data = []
    for view_idx in range(args.view_num):
        data_sources = {
            "images": f"{OUT_DIR}/images/{args.seq_name}_{view_idx:02d}",
            "tracks": f"{OUT_DIR}/track_preds/{args.seq_name}_{view_idx:02d}",
            "shots": f"{OUT_DIR}/shot_idcs/{args.seq_name}_{view_idx:02d}.json",
        }
        dataset = MultiPeopleDataset(data_sources=data_sources, seq_name=args.seq_name, shot_idx=0)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        datasets.append(dataset)
        obs_data.append(recursive_to(next(iter(loader)), device))
        num_tracks = obs_data[0]["track_id"].size(0)

    # Run scorehmr
    scorehmr = ScoreHMR(args, device, datasets, obs_data, num_tracks)
    scorehmr.iterate(args)