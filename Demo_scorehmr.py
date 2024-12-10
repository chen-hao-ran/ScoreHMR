import torch
import pickle

from scorehmr.scorehmr import ScoreHMR

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get Data
    data_path = "example_data/0019.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Start time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Run scorehmr
    scorehmr = ScoreHMR(device)
    scorehmr.iterate(data["batch"], data["mv_data"], data["num_tracks"])

    # End time
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000
    print(f"Elapsed time: {elapsed_time} seconds")
