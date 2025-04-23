import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_data_loaders
from model import build_model
from config import Config
import os
from utils import verify_dataset_transformations
from tqdm import tqdm  # Import tqdm for progress bars
import time
import torch.nn.functional as F
def train():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('debug_samples', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)  # For saving training logs
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    
    # Verify dataset transformations
    print("Verifying dataset transformations...")
    verify_dataset_transformations(train_loader.dataset, num_samples=5, save_dir="debug_samples")
    
    # Initialize model, loss, optimizer
    model = build_model(Config.BACKBONE, Config.NUM_KEYPOINTS).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), 
                         lr=Config.LEARNING_RATE, 
                         weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', 
                                patience=Config.PATIENCE, 
                                factor=0.1)
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    def heatmap_loss(pred, target, visibility, alpha=0.5):
        """Improved loss that penalizes false positives"""
        # Visible keypoint loss
        vis_mask = (visibility >= 1).float().unsqueeze(-1).unsqueeze(-1)
        pos_loss = F.mse_loss(pred * vis_mask, target * vis_mask)
        
        # Background suppression loss (new)
        bg_mask = (visibility < 1).float().unsqueeze(-1).unsqueeze(-1)
        bg_loss = F.mse_loss(pred * bg_mask, torch.zeros_like(pred) * bg_mask)
        
        return alpha * pos_loss + (1-alpha) * bg_loss
    
    def coord_loss(pred, target, visibility):
        """L1 loss for coordinate prediction"""
        # Only count visible keypoints
        vis_mask = (visibility >= 1).float().unsqueeze(-1)
        target = target.view(target.size(0), -1, 2)  # Reshape to [batch_size, num_keypoints, 2]
        return F.l1_loss(pred * vis_mask, target * vis_mask)
    
    def combined_loss(pred_heatmaps, pred_coords, target_heatmaps, target_coords, visibility):
        heatmap_loss = heatmap_loss(pred_heatmaps, target_heatmaps, visibility)
        coord_loss = coord_loss(pred_coords, target_coords, visibility)
        return heatmap_loss*100 + coord_loss
    
    # Training loop with tqdm
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        
        # Initialize tqdm progress bar for training
        train_pbar = tqdm(train_loader, 
                         desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]',
                         leave=False)
        
        for images, target_heatmaps, keypoints, visibility in train_pbar:
            images = images.to(Config.DEVICE)
            target_heatmaps = target_heatmaps.to(Config.DEVICE)
            visibility = visibility.to(Config.DEVICE)
            keypoints = keypoints.to(Config.DEVICE)

            pred_heatmaps, pred_coords = model(images)
            
            # Calculate loss
            loss = heatmap_loss(pred_heatmaps, target_heatmaps, visibility)
            loss += 0.3*coord_loss(pred_coords, keypoints, visibility)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss += loss.item() * images.size(0)
        
        train_pbar.close()
        
        # Validation loop with tqdm
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, 
                       desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]',
                       leave=False)
        
        with torch.no_grad():
            for images, target_heatmaps, _, visibility in val_pbar:
                images = images.to(Config.DEVICE)
                target_heatmaps = target_heatmaps.to(Config.DEVICE)
                visibility = visibility.to(Config.DEVICE)
                
                pred_heatmaps, _ = model(images)
                loss = heatmap_loss(pred_heatmaps, target_heatmaps, visibility)
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        val_pbar.close()
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{Config.NUM_EPOCHS} | '
              f'Time: {epoch_time:.1f}s | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f'New best model saved to {Config.CHECKPOINT_PATH}')
    
    # Save training logs
    torch.save({
        'train_history': train_history,
        'val_history': val_history,
        'config': vars(Config)
    }, 'training_logs/training_stats.pt')
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {Config.CHECKPOINT_PATH}")
    print(f"Training logs saved to: training_logs/")

if __name__ == '__main__':
    train()