import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from functools import total_ordering
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import AutoModelForVideoClassification, AutoModel
from transformers import TimesformerConfig, TimesformerForVideoClassification

class FitnessData(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
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
        self.classes = [cls for cls in os.listdir(root_dir) if not cls.startswith(".")]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.video_files = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
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


def train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for videos, labels in train_loader:
            # videos (B, C, T, H, W) = (B, 3, 16, 224, 224)
            # need to become (B, T, C, H, W) = (B, 16, 3, 224, 224)
            videos = videos.permute(0, 2, 1, 3, 4)
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            # TimeSformer expects inputs in shape (B, C, T, H, W)
            outputs = model(videos).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%")

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)
        train_acc = 100. * correct / total
        val_acc = validate(model, val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/total:.4f} Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% ")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "timesformer_best_1.pth")
            print(f"New Best Model Saved with Accuracy: {best_acc:.2f}%")
    print(f"Finished Training. Best Validation Accuracy: {best_acc:.2f}%")

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.permute(0, 2, 1, 3, 4)
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos).logits
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.permute(0, 2, 1, 3, 4)
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos).logits
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_acc = 100. * correct / total
    print(f"Val Acc: {val_acc:.2f}%")

def measure_fps(model, data_loader, device, warmup=5):
    model.eval()
    it = iter(data_loader)
    # 预热若干 batch（不计时）
    for _ in range(warmup):
        videos, _ = next(it)
        videos = videos.to(device)
        _ = model(videos.permute(0,2,1,3,4)).logits

    total_frames = 0
    start = time.perf_counter()
    with torch.no_grad():
        for videos, _ in data_loader:
            B, C, T, H, W = videos.shape  # usually (B,3,16,224,224)
            videos = videos.to(device)
            _ = model(videos.permute(0,2,1,3,4)).logits
            total_frames += B * T
    end = time.perf_counter()

    elapsed = end - start
    fps = total_frames / elapsed
    print(f"[Inference] Elapsed: {elapsed:.2f}s | Frames: {total_frames} | FPS: {fps:.1f}")
    return fps

if __name__ == "__main__":
    # Create dataset and DataLoaders using the provided FitnessData class.
    train_dataset = FitnessData(root_dir="kaggle_data/workoutfitness-video-train",transform=None, frames_per_clip=16)
    val_dataset = FitnessData(root_dir="kaggle_data/workoutfitness-video-val",transform=None, frames_per_clip=16)
    test_dataset = FitnessData(root_dir="kaggle_data/workoutfitness-video-test",transform=None, frames_per_clip=16)

    batch_size = 3
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    finetune_use = "origin"

    if finetune_use == "origin":
        # 1) Load the architecture config (patch size, depth, attention type, etc.)
        config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
        # 2) Instantiate a fresh TimeSformer model with random weights
        model = TimesformerForVideoClassification(config)
    elif finetune_use == "in21k":
        # 1) Build a fresh TimeSformer (random init) with the same architecture config
        config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model = TimesformerForVideoClassification(config)

        # 2) Load only the ViT‐Base/16 ImageNet weights
        vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")  # or "...-224"
        # Extract its state_dict
        vit_dict = vit.state_dict()

        # 3) Load those into the video backbone:
        #    (model.base_model is the `TimesformerModel` that wraps the ViT layers)
        model.base_model.load_state_dict(vit_dict, strict=False)
    else:
        # Load a pre-trained TimeSformer model from Hugging Face.
        # Note: "facebook/timesformer-base-finetuned-k400" is pre-trained on Kinetics-400 (400 classes).
        # model = TimeSformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400").to(device)
        model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    # Modify the classification head to output 22 classes (matching your dataset)
    model.config.num_labels = 22
    # Replace the classifier with a new linear layer.
    model.classifier = nn.Linear(model.config.hidden_size, 22)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Train and validate the model.
    train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=100)


