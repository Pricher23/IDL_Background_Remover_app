import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import glob

# picks GPU automatically if available, otherwise falls back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
PROJECT = r"D:\Uni\Info\Sem2\IDL"

print(f"Training on: {DEVICE}")

class SegDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # some augmentation so it doesn't overfit to collected lighting
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # NEAREST so mask edges don't get blurry after resize
        self.mask_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_tf(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.float32)
        # binarize to 0/1 and add channel dim
        mask = (mask > 0).float().unsqueeze(0)
        return self.img_tf(img), mask

def collect_pairs():
    pairs = []

    old_imgs = sorted(glob.glob(os.path.join(PROJECT, "dataset", "frames", "*.jpg")) +
                      glob.glob(os.path.join(PROJECT, "dataset", "frames", "*.png")))
    for ip in old_imgs:
        name = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".png", ".jpg"]:
            mp = os.path.join(PROJECT, "dataset", "masks", name + ext)
            if os.path.exists(mp):
                pairs.append((ip, mp))
                break

    kg_imgs = sorted(glob.glob(os.path.join(PROJECT, "dataset", "people_segmentation", "images", "*.jpg")) +
                     glob.glob(os.path.join(PROJECT, "dataset", "people_segmentation", "images", "*.png")))
    for ip in kg_imgs:
        name = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".png", ".jpg"]:
            mp = os.path.join(PROJECT, "dataset", "people_segmentation", "masks", name + ext)
            if os.path.exists(mp):
                pairs.append((ip, mp))
                break

    print(f"Total pairs found: {len(pairs)}")
    return pairs

def build_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # swap out the classifier to output a flat mask instead of class scores
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, IMG_SIZE * IMG_SIZE),
        nn.Sigmoid(),
    )
    weights_path = os.path.join(PROJECT, "model_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("Loaded existing weights. Continuing fine-tune")
    else:
        print("No existing weights. Training from scratch")
    return model.to(DEVICE)

def train():
    pairs = collect_pairs()
    if len(pairs) == 0:
        print("ERROR: No image/mask pairs found. Check dataset folder.")
        return

    # 90/10 train-val split
    split = int(0.9 * len(pairs))
    np.random.shuffle(pairs)
    train_dl = DataLoader(SegDataset(pairs[:split]), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(SegDataset(pairs[split:]), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    # started with plain Adam but AdamW worked a bit better (less overfitting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # tried StepLR first, cosine seems smoother for small datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            # model outputs flat vector, reshape to match mask dimensions
            out = model(imgs).view(-1, 1, IMG_SIZE, IMG_SIZE)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs).view(-1, 1, IMG_SIZE, IMG_SIZE)
                val_loss += criterion(out, masks).item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(PROJECT, "model_weights.pth"))
            print(f"  ✓ Saved best model (val={val_loss:.4f})")

    print("\nDone! Best val loss:", best_val)

if __name__ == "__main__":
    train()
