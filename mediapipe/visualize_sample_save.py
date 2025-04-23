import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from dual_input_dataset import DualInputPoseDataset

# Output folder
OUT_DIR = "./sample_visualization"
os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
dataset = DualInputPoseDataset("processed_data/workoutfitness-video-train/data_list.txt")
clip, keypoints, label = dataset[0]  # Change index as needed

# Denormalize: [C, T, H, W] → [T, H, W, C]
clip_np = clip.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
clip_np = (clip_np * 0.225 + 0.45) * 255.0
clip_np = clip_np.clip(0, 255).astype(np.uint8)

# Keypoints: [T, 132] → [T, 33, 4]
keypoints = keypoints.numpy().reshape(-1, 33, 4)
H, W = clip_np.shape[1:3]

# Draw and save each frame
for t in range(len(clip_np)):
    frame = clip_np[t].copy()
    kps = keypoints[t]
    for i in range(33):
        x, y, _, v = kps[i]
        if v > 0.5:
            cv2.circle(frame, (int(x * W), int(y * H)), 2, (0, 255, 0), -1)

    # Convert to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(4, 4))
    plt.imshow(frame_rgb)
    plt.title(f"Frame {t} | Label: {label}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"frame_{t:03d}.png"))
    plt.close()

print(f"Saved frames with keypoints to: {OUT_DIR}/")
