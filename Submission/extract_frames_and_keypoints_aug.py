import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import argparse
import random

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def extract_keypoints(frame):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        return np.array([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(33 * 4)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def extract_video_frames_and_kps(video_path, T=48, frame_size=(224, 224), augment_type=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, T).astype(int)
    frames, kps = [], []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            if augment_type == "flip":
                frame = cv2.flip(frame, 1)
            elif augment_type == "rotate":
                frame = rotate_image(frame, angle=random.uniform(-15, 15))
            resized = cv2.resize(frame, frame_size)
            keypoints = extract_keypoints(resized)
            frames.append(resized)
            kps.append(keypoints)
    cap.release()

    if len(frames) != T:
        return None, None, None

    frames = np.stack(frames)  # [T, H, W, 3]
    keypoints = np.stack(kps)  # [T, 132]

    # Compute keypoint velocities
    keypoint_velocities = np.diff(keypoints, axis=0)
    # Pad the first velocity to match length T
    first_velocity = keypoint_velocities[0:1]
    keypoint_velocities = np.vstack([first_velocity, keypoint_velocities])  # [T, 132]

    return frames, keypoints, keypoint_velocities


def process_and_save(video_path, out_dir, cls, fname, label, f):
    base_name = os.path.splitext(fname)[0]
    aug_types = [None, "flip", "rotate"]

    for aug in aug_types:
        frames, keypoints, velocities = extract_video_frames_and_kps(video_path, augment_type=aug)
        if frames is None:
            continue
        suffix = "" if aug is None else f"_{aug}"
        out_base = f"{cls}_{base_name}{suffix}".replace(" ", "_")
        frame_out = os.path.join(out_dir, f"{out_base}_frames.npy")
        kps_out = os.path.join(out_dir, f"{out_base}_kps.npy")
        vel_out = os.path.join(out_dir, f"{out_base}_vel.npy")

        np.save(frame_out, frames)
        np.save(kps_out, keypoints)
        np.save(vel_out, velocities)

        # Update list file
        f.write(f"{frame_out}\t{kps_out}\t{vel_out}\t{label}\n")


def main(args):
    input_dir = args.input_dir
    list_name = args.list_name
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    class_names = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    class_to_label = {cls: i for i, cls in enumerate(class_names)}

    list_path = os.path.join(out_dir, list_name)
    with open(list_path, "w") as f:
        for cls in class_names:
            label = class_to_label[cls]
            cls_dir = os.path.join(input_dir, cls)
            for fname in tqdm(os.listdir(cls_dir), desc=f"Processing {cls}"):
                if not fname.endswith((".mp4", ".avi")):
                    continue
                video_path = os.path.join(cls_dir, fname)
                process_and_save(video_path, out_dir, cls, fname, label, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", default="processed_data")
    parser.add_argument("--list_name", default="train_list.txt")
    args = parser.parse_args()
    main(args)
