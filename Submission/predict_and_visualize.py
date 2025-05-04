import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from dual_input_r2plus1d import DualInputR2Plus1D
from dual_input_r2plus1d_positional import DualInputR2Plus1DPositional

def put_label_text(frame, gt_label, pred_label):
    H, W, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # bigger font
    thickness = 3

    # Prepare text strings
    gt_text = f"GT: {gt_label}"
    pred_text = f"PRED: {pred_label}"

    # Get text sizes
    (gt_w, gt_h), _ = cv2.getTextSize(gt_text, font, font_scale, thickness)
    (pred_w, pred_h), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)

    # Compute positions: center horizontally, place near bottom
    x_gt = (W - gt_w) // 2
    x_pred = (W - pred_w) // 2
    y_gt = H - 70
    y_pred = H - 30

    # Optional: draw background rectangles
    cv2.rectangle(frame, (x_gt - 10, y_gt - gt_h - 10), (x_gt + gt_w + 10, y_gt + 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (x_pred - 10, y_pred - pred_h - 10), (x_pred + pred_w + 10, y_pred + 10), (0, 0, 0), -1)

    # Draw the text over the rectangles
    cv2.putText(frame, gt_text, (x_gt, y_gt), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, pred_text, (x_pred, y_pred), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return frame

def configure_model_inputs(model):
    if isinstance(model, DualInputR2Plus1DPositional):
        frame_size = (224, 224)
        sequence_length = 48
        keypoint_dim = 264
    elif model.__class__.__name__ == "DualInputR2Plus1D":
        frame_size = (224, 224)
        sequence_length = 48
        keypoint_dim = 132
    elif model.__class__.__name__ == "DualInputR2Plus1D":
        frame_size = (224, 224)
        sequence_length = 48
        keypoint_dim = 132
    else:
        frame_size = (224, 224)  # default
        sequence_length = 48
        keypoint_dim = 264
    return frame_size, sequence_length, keypoint_dim


# -------- CONFIG --------
test_vides = os.listdir("test/")
for video in test_vides:
    VIDEO_PATH = "test/{}".format(video)
    GROUND_TRUTH_LABEL = video.split(".")[0]
    LABELS = ["Barbell Bicep Curl", "Bench Press", "Chest Fly Machine", "Deadlift", "Decline Bench Press", "Hammer Curl", "Hip Thrust", "Incline Bench Press", "Lat Pulldown", "Lateral Raise", "Leg Extension", "Leg Raises", "Plank","Pull Up", "Push Up","Romanian Deadlift", "Russian Twist", "Shoulder Press", "Squat", "T Bar Row", "Tricep Pushdown", "Tricep Dips"]  # fill in your full label list
    CHECKPOINT_PATH = "checkpoints/best_model_resnet18_aug_velocity.pth"
    OUTPUT_PATH = "test_results/{}_aug_vel_prediction.mp4".format(GROUND_TRUTH_LABEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- MODEL --------
    model = DualInputR2Plus1D(num_classes=len(LABELS)).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    FRAME_SIZE, SEQUENCE_LENGTH, KEYPOINT_DIM = configure_model_inputs(model)
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

        # Calculate velocities
        velocities = np.diff(kpts, axis=0)
        first_velocity = velocities[0:1]
        velocities = np.vstack([first_velocity, velocities])

        # Concatenate keypoints and velocities
        if model.__class__.__name__ == "DualInputR2Plus1DPositional":
            kp_vel = np.concatenate([kpts, velocities], axis=-1)
            assert kp_vel.shape[-1] == KEYPOINT_DIM, f"Expected keypoint dim {KEYPOINT_DIM}, got {kp_vel.shape[-1]}"
        else:
            kp_vel = kpts

        clip_tensor = torch.stack(clip, dim=1).unsqueeze(0).to(device)
        kpts_tensor = torch.tensor([kp_vel], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(clip_tensor, kpts_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_label = LABELS[pred_idx]

        frame = draw_keypoints_and_skeleton(raw_frames[i], kpts[-1])
        frame = put_label_text(frame, GROUND_TRUTH_LABEL, pred_label)
        output_frames.append(frame)

    # -------- SAVE VIDEO --------
    H, W = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (W, H))
    for frame in output_frames:
        out.write(frame)
    out.release()
    print(f"✅ Saved to {OUTPUT_PATH}")