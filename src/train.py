# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import cv2
from tqdm import tqdm

from config import phase2_instance
from model import get_model

# --- Custom Dataset ---
class SIPaKMeDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for idx, cls in enumerate(phase2_instance.classes):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.exists(cls_path): continue
            for file in os.listdir(cls_path):
                if file.lower().endswith(".bmp"):
                    self.samples.append((os.path.join(cls_path, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

def get_train_transforms():
    return A.Compose([
        A.Resize(phase2_instance.img_size, phase2_instance.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.15),
        A.Normalize(mean=phase2_instance.mean, std=phase2_instance.std),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(phase2_instance.img_size, phase2_instance.img_size),
        A.Normalize(mean=phase2_instance.mean, std=phase2_instance.std),
        ToTensorV2()
    ])

def train(train_dir, val_dir, output_path="weights/mobilevit_s.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Data Loading
    train_ds = SIPaKMeDDataset(train_dir, get_train_transforms())
    val_ds = SIPaKMeDDataset(val_dir, get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=phase2_instance.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=phase2_instance.batch_size, shuffle=False, num_workers=2)

    # Class Weights for Macro Recall Optimization
    labels = [y for _, y in train_ds.samples]
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float).to(device)
    
    print(f"Class Weights: {class_weights}")

    # Model & Loss
    model = get_model(phase2_instance.NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=phase2_instance.label_smoothing)
    
    # 1. Warmup Phase (Head Only)
    print("--- Phase 1: Head Warmup ---")
    for param in model.parameters(): param.requires_grad = False
    for param in model.head.parameters(): param.requires_grad = True
    
    optimizer = optim.AdamW(model.head.parameters(), lr=phase2_instance.learning_rate)
    
    for epoch in range(5):
        train_epoch(model, train_loader, criterion, optimizer, device)
        acc = validate(model, val_loader, device)
        print(f"Warmup Epoch {epoch+1} | Val Acc: {acc:.4f}")

    # 2. Fine Tuning Phase (Full Model)
    print("--- Phase 2: Full Fine-tuning ---")
    for param in model.parameters(): param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=phase2_instance.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_acc = 0.0
    for epoch in range(20):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "classes": phase2_instance.classes}, output_path)
            print(f"Saved Best Model ({best_acc:.4f})")

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    TRAIN_DIR = "./data/SIPAKMED/Training"
    VAL_DIR = "./data/SIPAKMED/Test"
    train(TRAIN_DIR, VAL_DIR)