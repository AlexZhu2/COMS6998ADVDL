import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config
from utils import save_debug_image

def generate_heatmaps(keypoints, visibility, heatmap_size, sigma=10):
    """Generate Gaussian heatmaps for keypoints with correct coordinate mapping"""
    num_keypoints = len(Config.SELECTED_KEYPOINTS)
    heatmaps = torch.zeros((num_keypoints, heatmap_size[0], heatmap_size[1]))
    
    for i, ((x_img, y_img), vis) in enumerate(zip(keypoints, visibility)):
        # Skip non-visible and occluded keypoints
        if vis < 2:
            continue
            
        # Convert image coordinates to heatmap coordinates
        x = (x_img / Config.IMAGE_SIZE[1]) * heatmap_size[1]
        y = (y_img / Config.IMAGE_SIZE[0]) * heatmap_size[0]
        
        # Create coordinate grid
        yy, xx = torch.meshgrid(
            torch.arange(heatmap_size[0], dtype=torch.float32),
            torch.arange(heatmap_size[1], dtype=torch.float32)
        )
        
        # 2D Gaussian distribution
        gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmaps[i] = gaussian
    
    return heatmaps


class CocoKeypointsDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, is_train=True):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.is_train = is_train
        self.target_size = Config.IMAGE_SIZE
        self.heatmap_size = self.target_size
        
        # Normalization transform
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # Get person category ID
        self.person_ids = self.coco.getCatIds(catNms=['person'])
        
        # Get image IDs containing at least one person with keypoints
        self.image_ids = self.coco.getImgIds(catIds=self.person_ids)
        
        # Verify dataset integrity
        self._filter_invalid_entries()
        
    def _filter_invalid_entries(self):
        """Only keep images with one large visible person with sufficient keypoints"""
        valid_image_ids = []

        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.person_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            img_info = self.coco.loadImgs(img_id)[0]

            # Filter to persons with sufficient keypoints
            person_anns = [
                ann for ann in anns
                if ann.get("num_keypoints", 0) >= Config.MIN_KEYPOINTS
                and ann["category_id"] == 1
            ]

            if len(person_anns) != 1:
                continue  # Skip if not exactly one valid person

            # Check if person occupies enough of the image
            bbox = person_anns[0]['bbox']  # [x, y, w, h]
            person_area = bbox[2] * bbox[3]
            image_area = img_info['width'] * img_info['height']
            area_ratio = person_area / image_area

            if area_ratio >= Config.AREA_RATIO_THRESHOLD:
                valid_image_ids.append(img_id)

        self.image_ids = valid_image_ids                    

        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(
            self.root_dir,
            'train2014' if self.is_train else 'val2014',
            img_info['file_name']
        )
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.person_ids)
        annotations = self.coco.loadAnns(ann_ids)

        # Select the valid person annotation
        person_ann = None
        for ann in annotations:
            if ann["num_keypoints"] >= Config.MIN_KEYPOINTS:
                person_ann = ann
                break
        if person_ann is None:
            return self.__getitem__((idx + 1) % len(self))  # Skip bad samples

        # Process keypoints
        keypoints = np.array(person_ann['keypoints']).reshape(-1, 3)
        keypoints = keypoints[Config.SELECTED_KEYPOINTS]
        visibility = keypoints[:, 2].copy()
        keypoints = keypoints[:, :2].astype(np.float32)

        # Resize image
        image = cv2.resize(image, self.target_size)

        # Scale keypoints to resized image
        keypoints[:, 0] = keypoints[:, 0] * (self.target_size[1] / orig_w)
        keypoints[:, 1] = keypoints[:, 1] * (self.target_size[0] / orig_h)

        # Clip to bounds
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, self.target_size[1] - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, self.target_size[0] - 1)

        # Normalize keypoints to [0, 1]
        keypoints = keypoints / np.array([self.target_size[1], self.target_size[0]])
        keypoints = torch.from_numpy(keypoints).float().view(-1)

        # Convert image
        image = transforms.ToTensor()(image)
        image = self.normalize(image)

        return image, keypoints

    
    def verify_heatmaps(self, index=0, save_dir="debug_samples"):
        """Debug method to visualize heatmap generation"""
        image, heatmaps, keypoints, visibility = self[index]
        
        # Convert tensor to numpy
        image = image.numpy().transpose(1, 2, 0)
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw keypoints
        denorm_keypoints = (keypoints.view(-1, 2) * torch.tensor([self.target_size[1], self.target_size[0]])).numpy()
        for i, (x, y) in enumerate(denorm_keypoints):
            if visibility[i] >= 2:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Visualize heatmaps
        heatmap_img = np.max(heatmaps.numpy(), axis=0)
        heatmap_img = cv2.normalize(heatmap_img, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_img = cv2.applyColorMap(heatmap_img.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_img = cv2.resize(heatmap_img, self.target_size)
        
        combined = np.concatenate([image, heatmap_img], axis=1)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"debug_{index}.jpg"), combined)
        
def get_data_loaders():
    """Create data loaders with proper filtering"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CocoKeypointsDataset(
        root_dir=Config.ROOT_DIR,
        annotation_file=os.path.join(Config.ROOT_DIR, Config.TRAIN_ANNOTATION),
        transform=transform,
        is_train=True
    )
    
    val_dataset = CocoKeypointsDataset(
        root_dir=Config.ROOT_DIR,
        annotation_file=os.path.join(Config.ROOT_DIR, Config.VAL_ANNOTATION),
        transform=transform,
        is_train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    
    return train_loader, val_loader