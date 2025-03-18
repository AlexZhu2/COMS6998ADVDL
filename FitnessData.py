import torch
import os
import cv2
import torchvision.transforms as transforms
import numpy as np

class FitnessData(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        if train:
            self.root_dir = self.root_dir + "-train"
        else:
            self.root_dir = self.root_dir + "-val"
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # mean and std adopted from ImageNet

            ])
        else:
            self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.classes = sorted([cls for cls in os.listdir(self.root_dir) if not cls.startswith(".")])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.video_files = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if os.path.isdir(cls_path):
                for file in os.listdir(cls_path):
                    if file.endswith((".mp4", ".avi", ".mov")):
                        self.video_files.append((os.path.join(cls_path, file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.video_files)
    
    def read_video_frames(self, video_path):
        video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frames = []
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            video.release()
            return None
        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)
        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        video.release()
        return frames if len(frames) == self.frames_per_clip else None
    
    def __getitem__(self, idx):
        attempts = 3 
        for _ in range(attempts):
            video_path, label = self.video_files[idx]
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found → {video_path}")
                idx = np.random.randint(0, len(self))
                continue  # Try another video
            frames = self.read_video_frames(video_path)

            if frames is not None:  # Successfully read
                if self.transform is not None:
                    frames = [self.transform(frame) for frame in frames]
                return torch.stack(frames).permute(1,0,2,3), label
            
            idx = np.random.randint(0, len(self))
        
        raise RuntimeError(f"Failed to load video after {attempts} attempts: {video_path}")