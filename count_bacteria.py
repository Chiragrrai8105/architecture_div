# ============================================================
# 🧫 Count Bacteria in a Petri Dish Image using Trained Model
# (macOS MPS compatible – fixed IndexError in NMS)
# ============================================================

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import BacteriaDetector
from utils import *
import os

# ============================================================
# 🔹 Set your image path here 👇
# ============================================================
IMAGE_PATH = "/Users/ashithrai/Documents/projects/new_div/test/images/-BD-3-R3-E-coli-001_001-2_png.rf.f4758368539c5c22f9e566ae5f0074f5.jpg"

# ============================================================
# 🔹 Configuration
# ============================================================
CONF_THRESH = 0.55
NMS_THRESH = 0.3


IMG_SIZE = 640

# ============================================================
# 🔹 Device & Model Setup
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

model = BacteriaDetector(num_classes=1).to(device)
model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
model.eval()
print("✅ Loaded best model successfully!")

# ============================================================
# 🔹 Safe PyTorch NMS (fixed scalar indexing issue)
# ============================================================
def nms_py(boxes, scores, iou_thresh=0.4):
    """Simple Non-Max Suppression that works on all devices (CPU/MPS)."""
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item() if order.dim() == 0 else order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        inter_w = torch.clamp(xx2 - xx1, min=0)
        inter_h = torch.clamp(yy2 - yy1, min=0)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = (iou <= iou_thresh).nonzero(as_tuple=False).flatten()
        if inds.numel() == 0:
            break
        order = order[inds + 1]
    return torch.tensor(keep, dtype=torch.long)

# ============================================================
# 🔹 Post-process YOLO-like predictions
# ============================================================
def post_process(preds, conf_thresh=0.5, nms_thresh=0.4):
    boxes = []
    for p in preds:
        p = p.permute(0, 2, 3, 1).reshape(-1, p.shape[1])
        conf = torch.sigmoid(p[:, 0])
        xywh = torch.sigmoid(p[:, 1:5])
        mask = conf > conf_thresh
        conf = conf[mask]
        xywh = xywh[mask]

        for i in range(len(conf)):
            cx, cy, w, h = xywh[i]
            x1 = (cx - w / 2) * IMG_SIZE
            y1 = (cy - h / 2) * IMG_SIZE
            x2 = (cx + w / 2) * IMG_SIZE
            y2 = (cy + h / 2) * IMG_SIZE
            boxes.append([x1.item(), y1.item(), x2.item(), y2.item(), conf[i].item()])

    if len(boxes) == 0:
        return []

    boxes = torch.tensor(boxes)
    keep = nms_py(boxes[:, :4], boxes[:, 4], nms_thresh)
    return boxes[keep].cpu().numpy()

# ============================================================
# 🔹 Inference
# ============================================================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found at: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(tensor)

detections = post_process(preds, CONF_THRESH, NMS_THRESH)
count = len(detections)

print(f"\n🧫 Image: {os.path.basename(IMAGE_PATH)}")
print(f"🧠 Total Bacteria Detected: {count}\n")

# ============================================================
# 🔹 Draw Bounding Boxes and Save
# ============================================================
for (x1, y1, x2, y2, conf) in detections:
    cv2.rectangle(resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(resized, f"{conf:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

save_folder = "results/"
os.makedirs(save_folder, exist_ok=True)
output_path = os.path.join(save_folder, os.path.basename(IMAGE_PATH))
cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

print(f"✅ Annotated image saved to: {output_path}")

plt.imshow(resized)
plt.title(f"Bacteria Count: {count}")
plt.axis("off")
plt.show()
