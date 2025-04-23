import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from dual_input_r2plus1d import DualInputR2Plus1D

# -------- CONFIG --------
VIDEO_PATH = "Flat Dumbbell Bench Press.mp4"
GROUND_TRUTH_LABEL = "Bench Press"
LABELS = ["Barbell Bicep Curl", "Bench Press", "Chest Fly Machine", "Deadlift", "Decline Bench Press", "Hammer Curl", "Hip Thrust", "Incline Bench Press", "Lat Pulldown", "Lateral Raise", "Leg Extension", "Leg Raises", "Plank","Pull Up", "Push Up","Romanian Deadlift", "Russian Twist", "Shoulder Press", "Squat", "T Bar Row", "Tricep Dips", "Tricep Pushdown"]  # fill in your full label list
CHECKPOINT_PATH = "checkpoints/best_model.pth"
OUTPUT_PATH = "output_continuous_prediction.mp4"
SEQUENCE_LENGTH = 16
FRAME_SIZE = (112, 112)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- MODEL --------
model = DualInputR2Plus1D(num_classes=len(LABELS)).to(device)
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -------- MEDIAPIPE --------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

def extract_keypoints(frame):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        return np.array([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(33 * 4)

def draw_keypoints_and_skeleton(frame, keypoints):
    H, W, _ = frame.shape
    keypoints = np.array(keypoints).reshape(33, 4)
    for i, (x, y, _, v) in enumerate(keypoints):
        if v > 0.5:
            cx, cy = int(x * W), int(y * H)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
    for idx1, idx2 in mp_pose.POSE_CONNECTIONS:
        x1, y1, _, v1 = keypoints[idx1]
        x2, y2, _, v2 = keypoints[idx2]
        if v1 > 0.5 and v2 > 0.5:
            p1 = int(x1 * W), int(y1 * H)
            p2 = int(x2 * W), int(y2 * H)
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
    return frame

# -------- LOAD FRAMES --------
cap = cv2.VideoCapture(VIDEO_PATH)
raw_frames = []
keypoints_seq = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, FRAME_SIZE)
    kps = extract_keypoints(resized)
    keypoints_seq.append(kps)
    raw_frames.append(frame.copy())
cap.release()

# -------- SLIDING WINDOW --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.45] * 3, [0.225] * 3)
])

output_frames = []
for i in range(SEQUENCE_LENGTH - 1, len(raw_frames)):
    clip = [transform(cv2.resize(raw_frames[j], FRAME_SIZE)) for j in range(i - SEQUENCE_LENGTH + 1, i + 1)]
    kpts = [keypoints_seq[j] for j in range(i - SEQUENCE_LENGTH + 1, i + 1)]
    kpts = np.array(kpts)
    clip_tensor = torch.stack(clip, dim=1).unsqueeze(0).to(device)
    kpts_tensor = torch.tensor([kpts], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(clip_tensor, kpts_tensor)
        pred_idx = logits.argmax(dim=1).item()
        pred_label = LABELS[pred_idx]

    frame = draw_keypoints_and_skeleton(raw_frames[i], kpts[-1])
    cv2.putText(frame, f"GT: {GROUND_TRUTH_LABEL}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"PRED: {pred_label}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    output_frames.append(frame)

# -------- SAVE VIDEO --------
H, W = output_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (W, H))
for frame in output_frames:
    out.write(frame)
out.release()
print(f"✅ Saved to {OUTPUT_PATH}")