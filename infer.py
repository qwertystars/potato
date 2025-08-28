# ====================================================
# infer.py - Streamlit UI for Farmers
# ====================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ====================================================
# 1. Load Model
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "models/potato_model.pth",
    map_location=device,
    weights_only=True
)

# ✅ Your 6 target classes
class_names = ['Bacteria', 'Fungi', 'Healthy', 'Pest', 'Phytopthora', 'Virus']

# Build model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# ====================================================
# 2. Image Transform
# ====================================================
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====================================================
# 3. Streamlit UI
# ====================================================
st.title("🌿 Potato Leaf Disease Detector")
st.write("Upload a potato leaf image to check for **diseases**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width='stretch')

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    st.success(f"✅ Prediction: **{label}**")
