import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from config import Config
from torchvision.utils import make_grid
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def save_debug_image(image_tensor, keypoints, heatmaps=None, save_path="debug_samples"):
    """
    Enhanced debug image saver with optional heatmap visualization
    Args:
        image_tensor: Normalized image tensor (C,H,W)
        keypoints: Normalized keypoints (N,2)
        visibility: Visibility flags (N,)
        heatmaps: Optional heatmaps tensor (N,H,W)
        save_path: Directory to save images
    """
    os.makedirs(save_path, exist_ok=True)
    filename = f"sample_{len(os.listdir(save_path)):04d}.jpg"
    filepath = os.path.join(save_path, filename)
    
    # Convert image tensor to OpenCV format
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Denormalize keypoints
    h, w = image.shape[:2]
    keypoints = keypoints.view(-1, 2) * torch.tensor([w, h])
    keypoints = keypoints.numpy().astype(int)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        color = (0,0,255)
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(image, str(i), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255,255,255), 2)
    
    # Add heatmaps if provided
    if heatmaps is not None:
        heatmap_img = visualize_heatmaps(heatmaps, image.shape[:2])
        image = np.concatenate([image, heatmap_img], axis=1)
    
    cv2.imwrite(filepath, image)
    print(filepath)
    return filepath

def visualize_heatmaps(heatmaps, img_size, alpha=0.5):
    """
    Create a single heatmap visualization from multiple keypoint heatmaps
    Args:
        heatmaps: Tensor of shape (N,H,W)
        img_size: Target output size (h,w)
        alpha: Transparency for blending
    """
    # Sum all heatmaps and normalize
    combined_hm = torch.sum(heatmaps, dim=0)
    combined_hm = (combined_hm - combined_hm.min()) / (combined_hm.max() - combined_hm.min() + 1e-6)
    combined_hm = combined_hm.cpu().numpy()
    
    # Convert to color heatmap
    heatmap_img = cv2.applyColorMap((combined_hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_img = cv2.resize(heatmap_img, (img_size[1], img_size[0]))
    return heatmap_img

def verify_dataset_transformations(dataset, num_samples=5, save_dir="debug_samples"):
    """
    Verify dataset transformations with side-by-side original/augmented samples
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Verifying transformations on {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if len(sample) == 2:
            orig_img, orig_kps = sample
            orig_vis = None
            orig_heat = None
        elif len(sample) == 3:
            orig_img, orig_kps, orig_vis = sample
            orig_heat = None
        elif len(sample) == 4:
            orig_img, orig_heat, orig_kps, orig_vis = sample
        else:
            raise ValueError("Unexpected dataset return format.")

        save_debug_image(orig_img, orig_kps, orig_vis, orig_heat,
                         os.path.join(save_dir, f"sample_{i}_orig.jpg"))

        print(f"\nSample {i} Keypoints:")
        kps = orig_kps.view(-1, 2).cpu().numpy()
        for j, (x, y) in enumerate(kps):
            status = 'Visible' if orig_vis is not None and orig_vis[j] == 2 else 'Not Visible'
            print(f"KP{j}: ({x:.3f}, {y:.3f}) - {status}")

def evaluate_model(model, data_loader, device, save_dir="eval_results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_targets = []
    all_visibilities = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if len(batch) == 2:
                images, target_kps = batch
                target_heat = None
                vis = None
            elif len(batch) == 3:
                images, target_kps, vis = batch
                target_heat = None
            elif len(batch) == 4:
                images, target_heat, target_kps, vis = batch
            else:
                raise ValueError("Unexpected batch structure.")

            images = images.to(device)
            pred_output = model(images)

            if isinstance(pred_output, tuple):
                pred_heat, pred_kps = pred_output
            else:
                pred_kps = pred_output
                pred_heat = None

            # Save visualization
            for i in range(min(4, images.size(0))):
                save_debug_image(
                    images[i].cpu(),
                    pred_kps[i].cpu(),
                    vis[i][i].cpu() if vis is not None else None,
                    pred_heat[i].cpu() if pred_heat is not None else None,
                    os.path.join(save_dir, f"batch{batch_idx}_sample{i}.jpg")
                )

            if pred_heat is not None and target_heat is not None and vis is not None:
                all_preds.append(pred_heat.cpu())
                all_targets.append(target_heat.cpu())
                all_visibilities.append(vis.cpu())

    if all_preds:
        pred_heat = torch.cat(all_preds)
        target_heat = torch.cat(all_targets)
        visibility = torch.cat(all_visibilities)

        metrics = {
            'mse': calculate_mse(pred_heat, target_heat, visibility),
            'pck': calculate_pck(pred_heat, target_heat, visibility),
            'ap': calculate_average_precision(pred_heat, target_heat, visibility)
        }
    else:
        metrics = {
            'mse': None,
            'pck': None,
            'ap': None
        }

    return metrics


def calculate_mse(pred_heat, target_heat, visibility, threshold=0.5):
    """Mean Squared Error for visible keypoints"""
    mask = (visibility == 2).float()  # Only visible keypoints
    error = (pred_heat - target_heat)**2
    return (error * mask).sum() / (mask.sum() + 1e-6)

def calculate_pck(pred_heat, target_heat, visibility, threshold=0.2):
    """Percentage of Correct Keypoints within threshold"""
    # Get predicted and ground truth coordinates
    pred_coords = heatmaps_to_coords(pred_heat)
    target_coords = heatmaps_to_coords(target_heat)
    
    # Calculate distances
    distances = torch.norm(pred_coords - target_coords, dim=-1)
    correct = (distances < threshold).float()
    
    # Only count visible keypoints
    mask = (visibility == 2).float()
    return (correct * mask).sum() / (mask.sum() + 1e-6)

def heatmaps_to_coords(heatmaps):
    """Convert heatmaps to normalized coordinates using soft-argmax"""
    batch_size, num_kps, h, w = heatmaps.shape
    device = heatmaps.device
    
    # Create grid
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w))
    yy = yy.float().to(device) / h  # [0,1]
    xx = xx.float().to(device) / w  # [0,1]
    
    # Soft-argmax
    heatmaps = heatmaps.view(batch_size, num_kps, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)
    heatmaps = heatmaps.view(batch_size, num_kps, h, w)
    
    # Expected coordinates
    coords_y = (heatmaps * yy).sum(dim=(2,3))
    coords_x = (heatmaps * xx).sum(dim=(2,3))
    
    return torch.stack([coords_x, coords_y], dim=-1)  # [B, K, 2]

def calculate_average_precision(pred_heat, target_heat, visibility, threshold=0.5):
    """Calculate Average Precision for keypoint detection"""
    pred_coords = heatmaps_to_coords(pred_heat)
    target_coords = heatmaps_to_coords(target_heat)
    
    # Calculate distances
    distances = torch.norm(pred_coords - target_coords, dim=-1)
    correct = (distances < threshold).float()
    
    # Only count visible keypoints
    mask = (visibility == 2).float()
    return (correct * mask).sum() / (mask.sum() + 1e-6)

def plot_training_curves(train_history, val_history, save_path="training_curves.png"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()