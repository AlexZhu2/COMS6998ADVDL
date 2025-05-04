import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dual_input_dataset import DualInputPoseDataset
from dual_input_r2plus1d_positional import DualInputR2Plus1DPositional
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# ---------------------------- Config ----------------------------
EPOCHS = 10
BATCH_SIZE = 2
NUM_CLASSES = 22
CHECKPOINT_PATH = "checkpoints/best_model_resnet18_aug_velocity_positional.pth"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- Data ----------------------------
train_dataset = DualInputPoseDataset("processed_data/workoutfitness-train/train_list.txt", sequence_len=48)
val_dataset = DualInputPoseDataset("processed_data/workoutfitness-val/val_list.txt", sequence_len=48)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------- Model ----------------------------
model = DualInputR2Plus1DPositional(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
best_val_acc = 0.0

# ------------------------ Validation --------------------------
def evaluate(model, dataloader):
    model.eval()
    total, correct, loss_total = 0, 0, 0
    with torch.no_grad():
        for clips, keypoints, labels in dataloader:
            clips, keypoints, labels = clips.to(device), keypoints.to(device), labels.to(device)
            outputs = model(clips, keypoints)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            loss_total += loss.item() * clips.size(0)
            total += labels.size(0)
    return correct / total, loss_total / total

train_losses = []
val_losses = []
# ------------------------- Training Loop -----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch")

    for clips, keypoints, labels in loop:
        clips, kp_vel, labels = clips.to(device), keypoints.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips, kp_vel)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix({
            "TrainLoss": total_loss / total,
            "TrainAcc": f"{(correct / total) * 100:.2f}%"
        })

    # Validation
    train_acc = correct / total
    train_loss = total_loss / total
    val_acc, val_loss = evaluate(model, val_loader)
    print(f"\nEpoch {epoch+1}: TrainAcc={train_acc:.2%}, ValAcc={val_acc:.2%}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": best_val_acc
        }, CHECKPOINT_PATH)
        print(f"✅ Saved new best model (val acc: {best_val_acc:.2%})")
    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve (Positional)')
plt.legend()
plt.grid(True)
os.makedirs('loss_curves', exist_ok=True)
plt.savefig('loss_curves/loss_curve_positional.png')
plt.close()

print("✅ Saved loss curve to loss_curves/loss_curve_positional.png")
