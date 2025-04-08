import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deeppose_resnet_model import DeepPoseResNet
from penn_action.tools.penn_dataset_frame import PennActionFrameDataset


def vis_image_with_preds(
    image_tensor,
    pred_keypoints,
    pred_bbox,
    seq_id,
    frame_idx,
    gt_keypoints=None,
    gt_bbox=None,
    save_dir=None
):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import torch

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype('uint8')

    img_h, img_w = 224, 224
    pred_keypoints = pred_keypoints * np.array([img_w, img_h])
    pred_bbox = pred_bbox * np.array([img_w, img_h, img_w, img_h])

    if gt_keypoints is not None:
        gt_keypoints = gt_keypoints * np.array([img_w, img_h])
    if gt_bbox is not None:
        gt_bbox = gt_bbox * np.array([img_w, img_h, img_w, img_h])

    # Plot
    plt.figure(figsize=(5, 5))
    plt.imshow(image)

    # Draw predicted keypoints
    for x, y in pred_keypoints:
        plt.plot(x, y, 'ro', markersize=3, label='Pred KP')

    # Draw predicted bbox
    x, y, w, h = pred_bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, color='r', linewidth=2, label='Pred BBox')
    plt.gca().add_patch(rect)

    # Draw ground truth keypoints (green)
    if gt_keypoints is not None:
        for x, y in gt_keypoints:
            plt.plot(x, y, 'go', markersize=3, label='GT KP')

    # Draw ground truth bbox (green)
    if gt_bbox is not None:
        x, y, w, h = gt_bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, color='g', linewidth=2, label='GT BBox')
        plt.gca().add_patch(rect)

    plt.axis('off')
    plt.title(f"{seq_id} - Frame {frame_idx+1}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{seq_id}_{frame_idx+1:04d}.png")
        plt.savefig(out_path, bbox_inches='tight')
    plt.close()



@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion_kp, criterion_bbox, device, alpha=0.1, visualize=False, save_dir=None):
    model.eval()
    total_loss = 0.0
    saved_count = 0
    max_visuals = 10

    for i, batch in enumerate(tqdm(dataloader, desc='Validation', dynamic_ncols=True)):
        images = batch['frame'].to(device)
        gt_keypoints = batch['keypoints'].to(device)
        gt_bbox = batch['bbox'].to(device)
        seq_ids = batch['sequence_id']
        frame_idxs = batch['frame_idx']

        gt_keypoints = gt_keypoints.permute(0, 2, 1)  # [B, 13, 2]

        pred_bbox, pred_keypoints = model(images)

        loss_kp = criterion_kp(pred_keypoints, gt_keypoints)
        loss_bbox = criterion_bbox(pred_bbox, gt_bbox)
        loss = loss_kp + alpha * loss_bbox
        total_loss += loss.item()

        if visualize and saved_count < max_visuals:
            for b in range(images.size(0)):
                if saved_count >= max_visuals:
                    break
                vis_image_with_preds(
                    images[b].cpu(),
                    pred_keypoints[b].cpu().numpy(),
                    pred_bbox[b].cpu().numpy(),
                    seq_ids[b],
                    frame_idxs[b],
                    gt_keypoints=gt_keypoints[b].cpu().numpy(),
                    gt_bbox=gt_bbox[b].cpu().numpy(),
                    save_dir=save_dir
                )

                saved_count += 1

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    val_dataset = PennActionFrameDataset(mode='train', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = DeepPoseResNet(num_keypoints=13).to(device)

    # Load pretrained weights
    checkpoint_path = "./deeppose/deeppose_epoch100.pth"  # <-- change to your checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")

    # Define losses
    criterion_kp = torch.nn.MSELoss()
    criterion_bbox = torch.nn.MSELoss()

    # Run validation
    val_loss = validate_one_epoch(
        model, val_loader,
        criterion_kp, criterion_bbox,
        device, alpha=0.1,
        visualize=True,
        save_dir="./deeppose/val_vis"
    )

    print(f"\nFinal Validation Loss: {val_loss:.6f}")


if __name__ == '__main__':
    main()
