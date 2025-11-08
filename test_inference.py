# ============================================================
# 🚀 Inference Script for Bacteria Detector
# ============================================================

import torch
import cv2
import os
import matplotlib.pyplot as plt
from model import BacteriaDetector
from utils import *

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Load Trained Model ---
model = BacteriaDetector(num_classes=1).to(device)
model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
model.eval()
print("✅ Loaded best model successfully!")

# --- Test Folder ---
test_folder = "test/images/"
save_folder = "results/"
os.makedirs(save_folder, exist_ok=True)

# --- Inference ---
for img_name in os.listdir(test_folder):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(test_folder, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (640, 640))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)

    # 🧠 Print model raw output (before decoding)
    print(f"🔍 {img_name} | Model Output Shapes: {[p.shape for p in preds]}")

    # For now, just show image — decoding comes next
    plt.imshow(image_rgb)
    plt.title(f"Prediction for {img_name}")
    plt.axis("off")
    plt.show()
