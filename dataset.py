import os
import torch
from torch.utils.data import Dataset
import cv2

class BacteriaDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(
            self.label_dir,
            image_name.replace('.jpg', '.txt').replace('.png', '.txt')
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        if os.path.exists(label_path):
            for line in open(label_path):
                parts = line.strip().split()

                # Skip blank or malformed lines
                if len(parts) < 5:
                    continue  

                # Take only the first 5 values (ignore extras if present)
                try:
                    c, x, y, bw, bh = map(float, parts[:5])
                except ValueError:
                    # If something can't be converted to float, skip it
                    continue

                boxes.append([c, x * w, y * h, bw * w, bh * h])


        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))
        image = cv2.resize(image, (640, 640))
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
        return image, boxes
