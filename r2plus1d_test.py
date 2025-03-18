import torch
import torchvision.models.video as models
from torch.utils.data import DataLoader
from FitnessData import FitnessData
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, dataset, num_samples=3):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, dataset.frames_per_clip, figsize=(20, 4*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            frames, label = dataset[np.random.randint(0, len(dataset))]
            
            # Add batch dimension (1, C, T, H, W)
            frames = frames.unsqueeze(0)
            
            # Get model prediction
            output = model(frames)
            pred = torch.argmax(output, dim=1).item()
            
            with torch.no_grad():
                probs = torch.softmax(output, dim=1)  # Convert logits to probabilities
                pred = torch.argmax(output, dim=1).item()

            print(f"Predicted Class: {pred} ({dataset.idx_to_class[pred]})")
            print(f"True Class: {label} ({dataset.idx_to_class[label]})")
            print("Softmax Probabilities:", probs.numpy())

            
            # Unnormalize for visualization
            unnormalize = lambda img: img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            unnormalized_frames = [unnormalize(frame).clamp(0, 1) for frame in frames.squeeze(0).permute(1, 0, 2, 3)]
            
            # Plot all frames in the sequence
            for j in range(dataset.frames_per_clip):
                axes[i, j].imshow(unnormalized_frames[j].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                axes[i, j].axis('off')

            # Set labels for the first frame in each row
            axes[i, 0].set_title(f'True: {dataset.idx_to_class[label]} | Pred: {dataset.idx_to_class[pred]}', fontsize=12, color='green' if label == pred else 'red')

    plt.tight_layout()
    plt.show()

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            outputs = model(videos)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total  # Return accuracy percentage

# Load Model
num_classes = 22  # Update based on dataset
model = models.r2plus1d_18(weights=None)  
print("Before loading weights: ", model.fc)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('r2plus1d_18_best.pth', map_location=torch.device('cpu')))
print("After loading weights: ", model.fc)
model.eval()

# Load Test Dataset
test_dataset = FitnessData('workoutfitness-video', train=False, frames_per_clip=16)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(test_dataset.class_to_idx)
print(test_dataset.idx_to_class)

# # Run Improved Visualization
visualize_predictions(model, test_dataset)
# validate(model, test_loader)
