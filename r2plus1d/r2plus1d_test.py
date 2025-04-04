import torch
import torchvision.models.video as models
from torch.utils.data import DataLoader
from FitnessData import FitnessData
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, dataset, num_samples=3, keyframe_interval=4):
    """
    Visualize video classification predictions in a compact layout.

    - Displays `num_samples` videos.
    - Shows keyframes instead of all frames.
    - Colors the title green if correct, red if incorrect.
    - Adds prediction confidence score.

    Args:
        model (torch.nn.Module): Trained video classification model.
        dataset (Dataset): Video dataset.
        num_samples (int): Number of videos to visualize.
        keyframe_interval (int): Interval for selecting keyframes.
    """
    model.eval()
    
    keyframes = range(0, dataset.frames_per_clip, keyframe_interval)  # Select keyframes
    fig, axes = plt.subplots(num_samples, len(keyframes), figsize=(12, 3*num_samples))  # Compact layout

    with torch.no_grad():
        for i in range(num_samples):
            frames, label = dataset[np.random.randint(0, len(dataset))]

            # Add batch dimension (1, C, T, H, W)
            frames = frames.unsqueeze(0)

            # Get model prediction
            output = model(frames)
            probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[0, pred].item()  # Get confidence score

            print(f"Predicted: {dataset.idx_to_class[pred]} (Conf: {confidence:.2%}) | True: {dataset.idx_to_class[label]}")

            # Unnormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            unnormalize = lambda img: img * std + mean
            unnormalized_frames = [unnormalize(frame).clamp(0, 1) for frame in frames.squeeze(0).permute(1, 0, 2, 3)]

            # Plot selected keyframes
            for j, k in enumerate(keyframes):
                axes[i, j].imshow(unnormalized_frames[k].permute(1, 2, 0))  # Convert (C, H, W) → (H, W, C)
                axes[i, j].axis('off')

            # Set title for the first frame in each row
            color = "green" if label == pred else "red"
            axes[i, 0].set_title(f'True: {dataset.idx_to_class[label]}\nPred: {dataset.idx_to_class[pred]} ({confidence:.2%})',
                                 fontsize=10, color=color)

    plt.tight_layout()
    plt.show()

# Load Model
num_classes = 22  
model = models.r2plus1d_18(weights=None)  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('r2plus1d_18_best.pth', map_location=torch.device('cpu')))
model.eval()

# Load Test Dataset
test_dataset = FitnessData('workoutfitness-video', train=False, frames_per_clip=16)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run Compact Visualization
visualize_predictions(model, test_dataset, num_samples=3, keyframe_interval=4)
