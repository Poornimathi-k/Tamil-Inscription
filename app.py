import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import random

# ===== Set Streamlit Page Config (must be first) =====
st.set_page_config(layout="wide")

# ===== Model Definition =====
class TamilCharCNN(nn.Module):
    def __init__(self):
        super(TamilCharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===== Load Trained Model =====
@st.cache_resource
def load_model():
    model = TamilCharCNN()
    model.load_state_dict(torch.load("tamil_CNN+RNN+GNN.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ===== Transform for Upload Image =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== Main App Title =====
st.markdown("<h1 style='text-align: center; color: green;'>Tamil Inscription Letter Processing</h1>", unsafe_allow_html=True)

# ===== Create 3 Columns =====
col1, col2, col3 = st.columns([1, 1, 1])

# === Column 1: Upload Image ===
col1.markdown("### Upload Image")
uploaded_file = col1.file_uploader("", type=["png", "jpg", "jpeg"])
if uploaded_file:
    ancient_image = Image.open(uploaded_file).convert("L")
    col1.image(ancient_image, width=200)

# === Column 2: Process Button ===
col2.markdown("###         Process")
process_clicked = col2.button("Process Image", type="primary")

# === Column 3: Show Modern Output ===
col3.markdown("### Recognized Letter")
if uploaded_file and process_clicked:
    input_tensor = transform(ancient_image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_class = torch.max(output, 1)
        predicted_label = pred_class.item()

    modern_path = f"Modern characters/{predicted_label}"
    if os.path.exists(modern_path):
        modern_img = Image.open(os.path.join(modern_path, random.choice(os.listdir(modern_path))))
        col3.image(modern_img, width=200)
    else:
        col3.warning("No modern character found.")
