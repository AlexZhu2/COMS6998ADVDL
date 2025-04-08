import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import sys
import random

# Adjust sys.path as needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from deeppose.random_flip_transform import RandomHorizontalFlipWithKeypoints

class PennActionFrameDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        self.mode = mode
        self.target_size = (224, 224)  # (height, width)

        self.augment = mode == 'train'
        self.frame_stride = 10 if mode == 'train' else 1

        # Define image-only transforms (applied after joint transforms)
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations.
        ann_file = f'penn_action/labels_split/{mode}/all_{mode}_annotations.json'
        print(f"Loading annotations from {ann_file}")
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)

        if mode == 'train':
            random.shuffle(self.annotations)
        self.frames_dir = f'penn_action/frames_split/{mode}'

        # Build a frame-level index.
        num_samples_per_video = 10 if mode == 'train' else 1
        self.frame_index = []
        for seq_idx, seq in enumerate(self.annotations):
            for _ in range(num_samples_per_video):
                self.frame_index.append(seq_idx)

        # Define joint transform (only for training)
        if self.augment:
            self.joint_transform = RandomHorizontalFlipWithKeypoints(
                p=0.5,
                image_size=self.target_size,
                keypoint_pairs=np.array([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12]
                ])
            )
        else:
            self.joint_transform = None

    def crop_and_resize(self, frame, keypoints, bbox):
        """
        Crop the frame using the bounding box (bbox given as [x1, y1, x2, y2]),
        resize the cropped image to self.target_size, and normalize the keypoints.
        
        Normalized keypoints are computed as:
            normalized_x = (x - x1) / (x2 - x1)
            normalized_y = (y - y1) / (y2 - y1)
        
        Args:
            frame: full image (H, W, 3)
            keypoints: numpy array of shape [2, num_keypoints] (absolute pixel coordinates)
            bbox: numpy array, format [x1, y1, x2, y2]
        
        Returns:
            resized_crop: image cropped to the bbox and resized to target_size.
            keypoints_normalized: keypoints normalized to [0,1] relative to the crop.
        """
        # Get image dimensions.
        img_h, img_w = frame.shape[:2]

        # Extract and clamp bbox coordinates.
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(img_w, int(round(x2)))
        y2 = min(img_h, int(round(y2)))

        # If the bbox is invalid, default to the full image.
        if x2 <= x1 or y2 <= y1:
            # Print a warning and use full image instead.
            print("Warning: Invalid bounding box, using full image as fallback.")
            x1, y1, x2, y2 = 0, 0, img_w, img_h

        w = x2 - x1
        h = y2 - y1
        
        # Crop image.
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            # If, for any reason, the crop is still empty (shouldn't happen now),
            # fallback to the full image.
            print("Warning: Empty crop encountered, using full image as fallback.")
            crop = frame

            # Adjust keypoints: normalize against full image.
            x1, y1, w, h = 0, 0, img_w, img_h
        else:
            # Keypoints will be normalized relative to the crop.
            pass
        
        # Resize crop to target size.
        target_h, target_w = self.target_size
        resized_crop = cv2.resize(crop, (target_w, target_h))
        
        # Normalize keypoints relative to the crop.
        keypoints_norm = np.empty_like(keypoints)
        keypoints_norm[0] = (keypoints[0] - x1) / w
        keypoints_norm[1] = (keypoints[1] - y1) / h
        
        return resized_crop, keypoints_norm


    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        seq_idx = self.frame_index[idx]
        seq = self.annotations[seq_idx]
        seq_id = seq['sequence_id']

        # Randomly choose a frame from the sequence.
        frame_idx = np.random.randint(len(seq['frames']))
        frame_path = os.path.join(self.frames_dir, seq_id, f'{frame_idx+1:06d}.jpg')

        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        frame_ann = seq['frames'][frame_idx]
        keypoints = np.array(frame_ann['keypoints'], dtype=np.float32)  # Shape: [2, num_keypoints]
        bbox = np.array(frame_ann['bbox'], dtype=np.float32)            # Format: [x1, y1, x2, y2]
        
        # Crop image to bbox and normalize keypoints.
        image, keypoints = self.crop_and_resize(image, keypoints, bbox)
        
        # Apply joint transformation (e.g. flip) if augmenting.
        if self.joint_transform:
            image = Image.fromarray(image)
            image, keypoints = self.joint_transform(image, keypoints)
            image = np.array(image)
        
        # Apply image-only transforms.
        image = self.image_transform(Image.fromarray(image))
        
        # Return dictionary with only 'frame' and 'keypoints'
        sample = {
            'frame': image,
            'keypoints': torch.from_numpy(keypoints).float(),
            'sequence_id': seq_id,
            'frame_idx': frame_idx
        }
        return sample

# For testing the dataset module:
# Define the transform (used in dataset)
if __name__ == "__main__":
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    idx_to_joint = {
        1: 'head',
        2: 'left_shoulder',
        3: 'right_shoulder',
        4: 'left_elbow',
        5: 'right_elbow',
        6: 'left_wrist',
        7: 'right_wrist',
        8: 'left_hip',
        9: 'right_hip',
        10: 'left_knee',
        11: 'right_knee',
        12: 'left_ankle',
        13: 'right_ankle'
    }
    # Load a sample from the dataset.
    dataset = PennActionFrameDataset(mode='train', transform=transform)
    sample = dataset[694]
    
    print("Sample keys:", sample.keys())
    print("Keypoints shape:", sample['keypoints'].shape)
    print("Keypoints before denormalization:", sample['keypoints'])
    
    # Denormalize keypoints.
    # Assuming the target image size is 224x224, multiply keypoints (normalized to [0,1])
    # by 224 to get pixel coordinates.
    img_h, img_w = 224, 224
    denorm_keypoints = sample['keypoints'] * img_w  # shape remains [2, num_keypoints]
    print("Keypoints after denormalization:", denorm_keypoints)

    # Denormalize image:
    # 1. Convert the tensor to numpy array. The shape is [C, H, W].
    frame = sample['frame'].cpu().numpy()
    # 2. Transpose the array from [C, H, W] to [H, W, C].
    frame = np.transpose(frame, (1, 2, 0))
    
    # 3. Undo the normalization.
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    frame = frame * std + mean
    # 4. Convert to 0-255 pixel range.
    frame = (frame * 255).clip(0, 255).astype(np.uint8)
    
    # 5. Convert RGB to BGR for OpenCV.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw keypoints on the frame.
    num_keypoints = denorm_keypoints.shape[1]
    for i in range(num_keypoints):
        # denorm_keypoints is of shape [2, num_keypoints] with first row = x, second row = y.
        x = denorm_keypoints[0, i].item()
        y = denorm_keypoints[1, i].item()
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(frame, idx_to_joint[i+1], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Display the image using OpenCV.
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()