import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

class MPIIDataset(Dataset):
    def __init__(self, json_path, base_dir='mpii_data/mpii_split', mode='train'):
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val', or 'test'"
        self.mode = mode
        self.root_dir = os.path.join(base_dir, mode)
        self.img_size = (224, 224)

        # Load annotations
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        # Only keep entries where image exists in the corresponding folder
        valid_filenames = set(os.listdir(self.root_dir))
        self.samples = [
            entry for entry in all_data
            if entry["filename"] in valid_filenames
        ]

        # Image transformation
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample["filename"])
        image = Image.open(img_path).convert('RGB')

        # Original image size
        orig_w, orig_h = image.size

        # Resize image
        image = self.transform(image)

        # Compute scaling factors
        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h

        # Transform joint coordinates
        joints = torch.zeros((16, 2))
        for i in range(16):
            x, y = sample["joint_pos"][str(i)]
            joints[i][0] = x * scale_x
            joints[i][1] = y * scale_y

        return image, joints


# if __name__ == '__main__':
#     # Load dataset (example: validation set)
#     dataset = MPIIDataset(json_path='./mpii_data/data.json', mode='val')

#     # Get one sample
#     img_tensor, joints = dataset[0]  # (3, 224, 224), (16, 2)

#     # Convert image tensor to NumPy array for plotting
#     img_np = img_tensor.permute(1, 2, 0).numpy()  # (224, 224, 3)

#     # Plot the image
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img_np)
#     plt.scatter(joints[:, 0], joints[:, 1], c='red', s=20)
#     for i, (x, y) in enumerate(joints):
#         plt.text(x + 2, y, str(i), color='white', fontsize=8)
#     plt.title("Joint Visualization")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()