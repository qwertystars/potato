# ====================================================
# train.py - Train Potato Leaf Disease Classifier
# ====================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import sys
import subprocess
from leaf_preprocess import preprocess_leaf

# ====================================================
# 1. Path Validating
# ====================================================
# Train / Test directories (already prepared by dataset.py)
base_dir = os.path.abspath(".")   # repo root
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
    print("⚠️ train/ and test/ folders not found.")
    print("👉 Running dataset.py to prepare dataset...")
    try:
        subprocess.run([sys.executable, "dataset_gen.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to run dataset.py. Exiting.")
        sys.exit(1)

if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
    print("Failed to make dataset...")
    sys.exit(1)

print("Train folder:", train_dir)
print("Test folder:", test_dir)


# ====================================================
# 2. Data Preprocessing
# ====================================================
transform = transforms.Compose([
transforms.Lambda(preprocess_leaf),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# ====================================================
# 3. Model (ResNet18 Pretrained)
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# ====================================================
# 4. Loss & Optimizer
# ====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====================================================
# 5. Training Loop
# ====================================================
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")

# ====================================================
# 6. Save Model
# ====================================================
os.makedirs("models", exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "class_names": class_names
}, "models/potato_model.pth")

print("Model saved at models/potato_model.pth")
