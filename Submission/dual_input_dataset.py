import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class DualInputPoseDataset(Dataset):
    def __init__(self, list_file, sequence_len=48, frame_size=(224, 224)):
        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                frame_path, kp_path, vel_path, label = line.strip().split("\t")
                self.samples.append((frame_path, kp_path, vel_path, int(label)))

        self.sequence_len = sequence_len
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(frame_size),
            T.ToTensor(),
            T.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, kp_path, vel_path, label = self.samples[idx]

        frames = np.load(frame_path)  # [T, H, W, 3]
        keypoints = torch.tensor(np.load(kp_path), dtype=torch.float32)  # [T, 132]
        velocities = torch.tensor(np.load(vel_path), dtype=torch.float32)  # [T, 132]

        T_actual = frames.shape[0]

        if T_actual < self.sequence_len:
            pad = self.sequence_len - T_actual
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
            keypoints = torch.cat([keypoints, keypoints[-1:].repeat(pad, 1)], dim=0)
            velocities = torch.cat([velocities, velocities[-1:].repeat(pad, 1)], dim=0)
        else:
            frames = frames[:self.sequence_len]
            keypoints = keypoints[:self.sequence_len]
            velocities = velocities[:self.sequence_len]

        frame_tensors = [self.transform(frame.astype(np.uint8)) for frame in frames]
        frames = torch.stack(frame_tensors, dim=1)  # [C, T, H, W]

        # Concatenate keypoints and velocities here:
        kp_vel = torch.cat([keypoints, velocities], dim=-1)  # [T, 264]

        return frames, kp_vel, label

