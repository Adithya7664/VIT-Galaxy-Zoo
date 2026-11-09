import os
import zipfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# EXTRACT IMAGES
# ─────────────────────────────────────────────
EXTRACT_DIR = "/kaggle/working/images_training_rev1/"
ZIP_PATH    = "/kaggle/input/galaxy-zoo-the-galaxy-challenge/images_training_rev1.zip"

if not os.path.exists(EXTRACT_DIR):
    print("Extracting training images...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall("/kaggle/working/")
    print("Extraction done.")
else:
    print("Images already extracted.")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = 96
PATCH_SIZE  = 16
EMBED_DIM   = 192
NUM_HEADS   = 3
DEPTH       = 6
IN_CHANNELS = 3
EPOCHS      = 30
LR          = 1e-4
BATCH_SIZE  = 64
NUM_CLASSES = 3

IMG_DIR  = "/kaggle/working/images_training_rev1/"
CSV_PATH = "/kaggle/input/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv"

# ─────────────────────────────────────────────
# PATCH EMBEDDING
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                               kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# ─────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio * embed_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# VIT
# ─────────────────────────────────────────────
class VIT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, in_channels, num_heads, depth):
        super().__init__()
        self.patch_tokens         = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches          = self.patch_tokens.num_patches
        self.cls_token            = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.transformer_blocks   = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm                 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_patch = self.patch_tokens(x)
        cls     = self.cls_token.expand(x.shape[0], -1, -1)
        x       = torch.cat([cls, x_patch], dim=1)
        x       = x + self.positional_embedding
        for block in self.transformer_blocks:
            x = block(x)
        return self.norm(x)[:, 0]

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class GalaxyZooDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.transform = transform
        df        = pd.read_csv(csv_path)
        prob_cols = [c for c in df.columns if c != 'GalaxyID']
        self.ids  = df['GalaxyID'].tolist()
        self.labels      = torch.tensor(df[prob_cols].values.astype('float32'))
        self.id_to_label = {gid: i for i, gid in enumerate(self.ids)}
        self.img_paths   = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.jpg')
            and int(os.path.splitext(f)[0]) in self.id_to_label
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        image     = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        galaxy_id = int(os.path.splitext(os.path.basename(img_path))[0])
        label     = self.labels[self.id_to_label[galaxy_id]]
        return image, label

# ─────────────────────────────────────────────
# SUPERVISED VIT
# ─────────────────────────────────────────────
class SupervisedViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE,
                           embed_dim=EMBED_DIM, in_channels=IN_CHANNELS,
                           num_heads=NUM_HEADS, depth=DEPTH)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        return self.head(self.encoder(x))

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH, transform=transform)

# 20% for training — low-label regime to compare against SSL
all_indices = np.arange(len(full_dataset))
train_idx, test_idx = train_test_split(all_indices, test_size=0.8, random_state=42)

train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(Subset(full_dataset, test_idx),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

model     = SupervisedViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images      = images.to(device)
        class_label = labels[:, :3].argmax(dim=1).to(device)
        loss        = criterion(model(images), class_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 30 == 0:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Batch: {batch_idx}")
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(train_loader):.4f}")

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images      = images.to(device)
        class_label = labels[:, :3].argmax(dim=1)
        preds       = model(images).argmax(dim=1).cpu()
        all_preds.append(preds.numpy())
        all_labels.append(class_label.numpy())

acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))

print(f"\n── Results ─────────────────────────────────")
print(f"SSL pretrain + linear probe (100% labels): 81.00%")
print(f"SSL pretrain + kNN          (100% labels): 78.73%")
print(f"Supervised ViT from scratch ( 20% labels): {acc * 100:.2f}%")
