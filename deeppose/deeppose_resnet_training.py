import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
import sys
import os
from tqdm import tqdm

# Add parent directory to sys.path so we can import the model and dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deeppose_resnet_model import DeepPoseResNetWithCascade  # Ensure this file contains the revised DeepPose model.
from penn_action.tools.penn_dataset_frame import PennActionFrameDataset  # Ensure this file returns 'bbox' and 'keypoints' correctly.

def train_one_epoch(model, dataloader, optimizer, criterion_kp, criterion_bbox, device, alpha=0.1, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc='Training', dynamic_ncols=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['frame'].to(device)
        gt_keypoints = batch['keypoints'].to(device)  # Expected shape: [B, 2, num_keypoints]
        gt_bbox = batch['bbox'].to(device)            # Expected shape: [B, 4]
        
        # Permute keypoints to match the model output shape: [B, num_keypoints, 2]
        gt_keypoints = gt_keypoints.permute(0, 2, 1)
        
        pred_bbox, pred_keypoints_stage1, pred_keypoints_stage2 = model(images)
        
        # Compute losses: stage1 and stage2 keypoints are both supervised.
        loss_kp_stage1 = criterion_kp(pred_keypoints_stage1, gt_keypoints)
        loss_kp_stage2 = criterion_kp(pred_keypoints_stage2, gt_keypoints)
        loss_bbox = criterion_bbox(pred_bbox, gt_bbox)
        
        # Total loss with a weighting factor for bounding box loss.
        loss = loss_kp_stage1 + alpha * loss_bbox + loss_kp_stage2
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'loss_kp': f'{loss_kp_stage1.item():.4f}',
            'loss_bbox': f'{loss_bbox.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion_kp, criterion_bbox, device, alpha=0.1):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', dynamic_ncols=True):
            images = batch['frame'].to(device)
            gt_keypoints = batch['keypoints'].to(device)
            gt_bbox = batch['bbox'].to(device)
            
            gt_keypoints = gt_keypoints.permute(0, 2, 1)
            
            pred_bbox, pred_keypoints_stage1, pred_keypoints_stage2 = model(images)
            
            loss_kp_stage1 = criterion_kp(pred_keypoints_stage1, gt_keypoints)
            loss_kp_stage2 = criterion_kp(pred_keypoints_stage2, gt_keypoints)
            loss_bbox = criterion_bbox(pred_bbox, gt_bbox)
            loss = loss_kp_stage1 + alpha * loss_bbox + loss_kp_stage2
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define image transform (you could add more data augmentation here if desired)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    # Create training and validation datasets
    train_dataset = PennActionFrameDataset(mode='train', transform=transform)
    val_dataset = PennActionFrameDataset(mode='val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    model = DeepPoseResNetWithCascade(num_keypoints=13).to(device)
    
    # Use MSELoss for both keypoint and bounding box regressions.
    criterion_kp = nn.MSELoss()
    criterion_bbox = nn.MSELoss()
    
    # Use Adam optimizer with weight decay for L2 regularization.
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler: reduce LR on plateau.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_kp, criterion_bbox, device)
        val_loss = validate(model, val_loader, criterion_kp, criterion_bbox, device)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Step scheduler based on validation loss.
        scheduler.step(val_loss)
        
        # Early stopping check.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'deeppose_best.pth')
            print("Saved best model.")
        else:
            epochs_no_improve += 1
        
        # Save current epoch model.
        torch.save(model.state_dict(), f'deeppose_epoch{epoch+1}.pth')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

if __name__ == '__main__':
    main()
