import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.vit import VIT
from models.mae import MAE
from data.dataset import GalaxyZooDataset
from configs.config import (
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH, IN_CHANNELS,
    DECODER_DIM, DECODER_DEPTH,
    IMG_DIR, CSV_PATH
)

CHECKPOINT = "C:/Users/adith/Downloads/mae_epoch70.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── build model and load checkpoint ──────────────────────────────────────────
encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              in_channels=IN_CHANNELS, num_heads=NUM_HEADS, depth=DEPTH)
model = MAE(vit_encoder=encoder, patch_size=PATCH_SIZE,
            num_patches=(IMG_SIZE // PATCH_SIZE) ** 2,
            encoder_embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
            decoder_depth=DECODER_DEPTH)

state_dict = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state_dict)
model.to(device)

# freeze every parameter — we only want embeddings, not to train anything
for p in model.parameters():
    p.requires_grad = False

model.eval()
print("Checkpoint loaded and encoder frozen.")

# ── labeled dataset ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset    = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH,
                               transform=transform, labeled=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
print(f"Labeled dataset: {len(dataset)} images")

# ── extract embeddings ────────────────────────────────────────────────────────
# labels are 37 probability columns; take argmax of first 3 (Class1.1/1.2/1.3)
# → 0=smooth, 1=features/disk, 2=star/artifact
all_embeddings = []
all_labels     = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)

        # run through encoder only (skip MAE decoder)
        x = model.encoder.patch_tokens(images)
        for block in model.encoder.transformer_blocks:
            x = block(x)
        x = model.encoder.norm(x)     # (B, num_patches, embed_dim)
        embedding = x.mean(dim=1)     # average pool → (B, embed_dim)

        all_embeddings.append(embedding.cpu().numpy())

        # convert 37 probs → 3-class label via argmax of first 3 columns
        class_probs = labels[:, :3]
        class_label = class_probs.argmax(dim=1)
        all_labels.append(class_label.numpy())

embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 192)
labels_arr = np.concatenate(all_labels, axis=0)       # (N,)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Class distribution: {np.bincount(labels_arr)}")

# ── train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels_arr, test_size=0.2, random_state=42, stratify=labels_arr
)

# ── linear probe ─────────────────────────────────────────────────────────────
print("\nTraining linear probe...")
linear = LogisticRegression(max_iter=1000, C=1.0)
linear.fit(X_train, y_train)
linear_acc = accuracy_score(y_test, linear.predict(X_test))
print(f"Linear Probe Accuracy: {linear_acc * 100:.2f}%")

# ── kNN ───────────────────────────────────────────────────────────────────────
print("\nRunning kNN...")
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(f"kNN Accuracy (k=20): {knn_acc * 100:.2f}%")
