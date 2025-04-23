import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
from dataset import get_data_loaders
from model import build_model

from utils import save_debug_image
def visualize_predictions(model, data_loader, device, num_samples=10, save_dir="predictions"):
    """
    Visualize model predictions on validation data
    Args:
        model: Loaded model
        data_loader: Validation data loader
        device: CUDA device
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, target_kps) in enumerate(data_loader):
            if batch_idx * images.size(0) >= num_samples:
                break
                
            images = images.to(device)
            print("Beginning Inference...")
            pred_kps = model(images)
            print("Finished Inference")
            
            # Convert tensors to CPU for visualization
            images = images.cpu()
            pred_kps = pred_kps.cpu()
            target_kps = target_kps.cpu()
            
            # Save visualizations for each sample in batch
            for i in range(images.size(0)):
                if batch_idx * images.size(0) + i >= num_samples:
                    break
                    
                # Create composite image showing:
                # 1. Original image with ground truth keypoints
                # 2. Original image with predicted keypoints
                
                # Get all components
                img = images[i]
                gt_kps = target_kps[i]
                pr_kps = pred_kps[i]
                
                # Save ground truth
                gt_path = os.path.join(save_dir, f"sample_{batch_idx}_{i}_gt.jpg")
                save_debug_image(image_tensor=img, keypoints=gt_kps, save_path=gt_path)
                
                # Save prediction
                pred_path = os.path.join(save_dir, f"sample_{batch_idx}_{i}_pred.jpg")
                save_debug_image(image_tensor=img, keypoints=pr_kps, save_path=pred_path)

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = build_model().to(device)
    checkpoint = torch.load("./deeppose/checkpoints/best_model_resnet18.pth", map_location=device)
    model.load_state_dict(checkpoint)
    print("Model loaded from checkpoint")
    
    # Get data loaders
    _, val_loader = get_data_loaders()
    
    # Visualize predictions
    visualize_predictions(model, val_loader, device, num_samples=20, save_dir="./deeppose/predictions")
    print("Visualizations saved to 'predictions' directory")