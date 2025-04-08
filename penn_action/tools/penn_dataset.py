import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class PennActionDataset(Dataset):
    def __init__(self, mode='train', transform=None, frame_num=16):
        """
        Penn Action Dataset loader
        Args:
            mode (str): Which mode to load ('train', 'val', or 'test')
            transform: Optional transform to apply to images
            frame_num (int): Number of frames to sample from each sequence
        """
        self.mode = mode
        self.frame_num = frame_num
        self.target_size = (224, 224)
        
        # Define default transforms if none provided
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load annotations
        ann_file = f'penn_action/labels_split/{mode}/all_{mode}_annotations.json'
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.frames_dir = f'penn_action/frames_split/{mode}'
    
    def resize_frame_and_labels(self, frame, keypoints, bbox):
        """Resize frame and adjust labels accordingly"""
        h, w = frame.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factors while preserving aspect ratio
        scale = min(target_w/w, target_h/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize frame
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Pad frame
        frame = cv2.copyMakeBorder(frame, pad_h, target_h-new_h-pad_h, 
                                 pad_w, target_w-new_w-pad_w, 
                                 cv2.BORDER_CONSTANT, value=0)
        
        # Scale and adjust keypoints
        keypoints[0] = keypoints[0] * scale + pad_w  # x coordinates
        keypoints[1] = keypoints[1] * scale + pad_h  # y coordinates
        
        # Scale and adjust bbox
        bbox[0] = bbox[0] * scale + pad_w  # x
        bbox[1] = bbox[1] * scale + pad_h  # y
        bbox[2] = bbox[2] * scale          # width
        bbox[3] = bbox[3] * scale          # height
        
        return frame, keypoints, bbox
        
    def transform_frame(self, frame):
        """Apply transforms to a single frame"""
        frame = Image.fromarray(frame)
        frame = self.transform(frame)
        return frame
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Get sequence folder
        seq_id = ann['sequence_id']
        seq_dir = os.path.join(self.frames_dir, seq_id)
        
        # Get total frames in sequence
        num_frames = ann['num_frames']
        
        # Sample frame indices uniformly
        if num_frames >= self.frame_num:
            frame_indices = np.linspace(0, num_frames-1, self.frame_num, dtype=int)
        else:
            frame_indices = np.pad(
                np.arange(num_frames),
                (0, self.frame_num - num_frames),
                mode='edge'
            )
        
        # Load frames
        frames = []
        keypoints = []
        bboxes = []
        
        for frame_idx in frame_indices:
            # Load image
            frame_path = os.path.join(seq_dir, f'{frame_idx+1:06d}.jpg')
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Get annotations for this frame
            frame_ann = ann['frames'][frame_idx]
            kpts = np.array(frame_ann['keypoints'])
            bbox = np.array(frame_ann['bbox'])
            
            # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h] format
            x1, y1, x2, y2 = bbox
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])  # Convert to width/height format
            
            # Print original values for debugging
            if frame_idx < 5:  # Only print first 5 frames
                print(f"Frame {frame_idx+1} original - Image size: ({w}, {h})")
                print(f"  Original bbox [x1,y1,x2,y2]: {frame_ann['bbox']}")
                print(f"  Converted bbox [x,y,w,h]: {bbox}")
            
            # Resize frame and adjust labels
            image, kpts, bbox = self.resize_frame_and_labels(image, kpts, bbox)
            
            if frame_idx < 5:  # Only print first 5 frames
                print(f"  Transformed bbox: {bbox}")
            
            # Apply remaining transforms
            if self.transform:
                image = self.transform_frame(image)
            
            frames.append(image)
            keypoints.append(kpts)
            bboxes.append(bbox)
        
        # Stack frames into tensor [T, C, H, W]
        frames = torch.stack(frames)
        keypoints = np.array(keypoints)  # [T, 2, K] where K is number of keypoints
        bboxes = np.array(bboxes)      # [T, 4]
        
        sample = {
            'frames': frames,
            'keypoints': keypoints,
            'bboxes': bboxes,
            'sequence_id': seq_id
        }
        
        return sample
    
if __name__ == '__main__':
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PennActionDataset(mode='train', transform=transform)
    print(len(dataset))
    sample = dataset[980]
    print(sample['frames'].shape)
    print(sample['keypoints'].shape)
    print(sample['bboxes'].shape)
    
    # Visualize first 5 frames with keypoints and bounding boxes
    import matplotlib.pyplot as plt
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frames = sample['frames'][:5] * std + mean  # First 5 frames
    frames = frames.permute(0, 2, 3, 1).numpy()  # [5, H, W, C]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(frames[i])
        
        # Draw keypoints
        kpts = sample['keypoints'][i]
        for x, y in zip(kpts[0], kpts[1]):
            if x > 1 and y > 1:  # Only draw visible keypoints
                axes[i].plot(x, y, 'ro', markersize=3)
        
        # Draw bounding box
        bbox = sample['bboxes'][i]
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, color='g')
        axes[i].add_patch(rect)
        
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}')
    
    plt.tight_layout()
    plt.show()
