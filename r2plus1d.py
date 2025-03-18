import torch
import torchvision.models.video as models
import torch.optim as optim
from tqdm import tqdm
from FitnessData import FitnessData
from torch.utils.data import DataLoader

model = models.r2plus1d_18(pretrained=True)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=22)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for videos, labels in loop:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss / total, acc=100. * correct / total)

        val_acc = validate(model, val_loader)
        print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "r2plus1d_18_best.pth")
            print(f"✅ New Best Model Saved with Accuracy: {best_acc:.2f}%")

    print(f"Finished Training. Best Validation Accuracy: {best_acc:.2f}%")

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total  # Return accuracy percentage

train_dataset = FitnessData(root_dir="workoutfitness-video", train=True, transform=None, frames_per_clip=16)
val_dataset = FitnessData(root_dir="workoutfitness-video", train=False, transform=None, frames_per_clip=16)

batch_size = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

model.load_state_dict(torch.load("r2plus1d_18_best.pth"))
print("Loaded Best Model for Final Testing")
final_acc = validate(model, val_loader)
print(f"Final Model Accuracy: {final_acc:.2f}%")
