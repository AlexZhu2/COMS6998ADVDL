import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # Create train, val, test directories
    base_path = Path(source_dir).parent
    for split in ['train', 'val', 'test']:
        split_path = base_path / f'{source_dir}-{split}'
        if split_path.exists():
            shutil.rmtree(split_path)
        split_path.mkdir(exist_ok=True)

    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')]

    for class_name in class_dirs:
        # Create class subdirectories in each split
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(f'{source_dir}-{split}', class_name), exist_ok=True)

        # Get all videos for this class
        videos = [v for v in os.listdir(os.path.join(source_dir, class_name)) 
                 if v.endswith(('.mp4', '.avi', '.mov'))]
        random.shuffle(videos)

        # Ensure at least 1 sample per class in val and test sets
        n_videos = len(videos)
        if n_videos < 3:
            raise ValueError(f"Class '{class_name}' has less than 3 videos. At least 3 are needed for splitting!")

        # Assign at least one sample to val and test sets
        n_val = max(1, int(val_ratio * n_videos))
        n_test = max(1, int(test_ratio * n_videos))
        n_train = n_videos - n_val - n_test  # Remaining goes to train

        # Split videos
        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]

        # Copy videos to respective directories
        for video, split_dir in [
            (train_videos, f'{source_dir}-train'),
            (val_videos, f'{source_dir}-val'),
            (test_videos, f'{source_dir}-test')
        ]:
            for v in video:
                src = os.path.join(source_dir, class_name, v)
                dst = os.path.join(split_dir, class_name, v)
                shutil.copy2(src, dst)

        print(f"Class '{class_name}' split complete:")
        print(f"Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")

if __name__ == "__main__":
    split_dataset('workoutfitness-video')
    print("Dataset splitting complete!")
