import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.vit import VIT
from data.dataset import GalaxyZooDataset
from configs.config import (
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH, IN_CHANNELS,
    IMG_DIR, CSV_PATH
)

EPOCHS     = 30
LR         = 1e-4
BATCH_SIZE = 64
NUM_CLASSES = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH,
                                transform=transform, labeled=True)

# use only 20% of data for training — same low-label regime as the SSL comparison
all_indices = np.arange(len(full_dataset))
train_idx, test_idx = train_test_split(all_indices, test_size=0.8, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
test_dataset  = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# ── model: ViT + classification head ─────────────────────────────────────────
class SupervisedViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE,
                           embed_dim=EMBED_DIM, in_channels=IN_CHANNELS,
                           num_heads=NUM_HEADS, depth=DEPTH)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        return self.head(self.encoder(x))  # encoder returns CLS token (B, embed_dim)

model     = SupervisedViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── training loop ─────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        # convert 37-col labels → 3-class
        class_label = labels[:, :3].argmax(dim=1).to(device)

        logits = model(images)
        loss   = criterion(logits, class_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 30 == 0:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Batch: {batch_idx}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

# ── evaluation ────────────────────────────────────────────────────────────────
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images      = images.to(device)
        class_label = labels[:, :3].argmax(dim=1)
        logits      = model(images)
        preds       = logits.argmax(dim=1).cpu()
        all_preds.append(preds.numpy())
        all_labels.append(class_label.numpy())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
acc = accuracy_score(all_labels, all_preds)
print(f"\nSupervised ViT Accuracy (20% labels): {acc * 100:.2f}%")
print(f"\n── Comparison ──────────────────────────────")
print(f"SSL pretrain + linear probe (100% labels): 81.00%")
print(f"SSL pretrain + kNN          (100% labels): 78.73%")
print(f"Supervised ViT from scratch ( 20% labels): {acc * 100:.2f}%")
