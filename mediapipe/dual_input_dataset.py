import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class DualInputPoseDataset(Dataset):
    def __init__(self, list_file, sequence_len=16, frame_size=(112, 112)):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                frame_path, keypoint_path, label = line.strip().split("\t")
                self.samples.append((frame_path, keypoint_path, int(label)))
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(frame_size),
            T.ToTensor(),
            T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, kp_path, label = self.samples[idx]
        frames = np.load(frame_path)  # [T, H, W, 3]
        frames = torch.stack([self.transform(frame) for frame in frames], dim=1)  # [C, T, H, W]
        keypoints = torch.tensor(np.load(kp_path), dtype=torch.float32)  # [T, 132]
        return frames, keypoints, label
