import os
import random
import shutil
import json

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split frames into train, validation and test sets.
    
    Args:
        data_dir: Directory containing the frame folders
        train_ratio: Ratio of training data (default 0.7) 
        val_ratio: Ratio of validation data (default 0.15)
        test_ratio: Ratio of test data (default 0.15)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all frame folders
    frame_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    random.shuffle(frame_folders)
    
    # Calculate split sizes
    total = len(frame_folders)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    # Split into train/val/test
    train_folders = frame_folders[:train_size]
    val_folders = frame_folders[train_size:train_size + val_size]
    test_folders = frame_folders[train_size + val_size:]
    
    # Create split directories if they don't exist
    splits = {
        'train': train_folders,
        'val': val_folders,
        'test': test_folders
    }
    
    # Create main split directory
    split_base_dir = 'penn_action/frames_split'
    os.makedirs(split_base_dir, exist_ok=True)
    
    # Create train/val/test subdirectories
    for split_name in splits.keys():
        split_dir = os.path.join(split_base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Copy frame folders to respective split directories
        for folder in splits[split_name]:
            src = os.path.join(data_dir, folder)
            dst = os.path.join(split_dir, folder)
            shutil.copytree(src, dst)
        
    print(f"Dataset split complete:")
    print(f"Train set: {len(train_folders)} sequences")
    print(f"Validation set: {len(val_folders)} sequences")
    print(f"Test set: {len(test_folders)} sequences")

if __name__ == '__main__':
    data_dir = 'penn_action/frames'  # Change this to your frames directory
    split_dataset(data_dir)
