import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import umap
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vit import VIT
from models.mae import MAE
from data.dataset import GalaxyZooDataset
from configs.config import (
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH, IN_CHANNELS,
    DECODER_DIM, DECODER_DEPTH, IMG_DIR, CSV_PATH
)

CHECKPOINT = "C:/Users/adith/Downloads/mae_epoch70.pth"
OUT_PATH   = "C:/Users/adith/Documents/VIT-Galaxyzoo/eval/outputs/embeddings.npz"
NUM_SAMPLES = 5000
device = torch.device('cpu')

encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              in_channels=IN_CHANNELS, num_heads=NUM_HEADS, depth=DEPTH)
model = MAE(vit_encoder=encoder, patch_size=PATCH_SIZE,
            num_patches=(IMG_SIZE // PATCH_SIZE) ** 2,
            encoder_embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
            decoder_depth=DECODER_DEPTH)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
for p in model.parameters():
    p.requires_grad = False

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset    = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH,
                               transform=transform, labeled=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

embeddings, labels_list = [], []
count = 0

with torch.no_grad():
    for images, labels in dataloader:
        if count >= NUM_SAMPLES:
            break
        x = model.encoder.patch_tokens(images)
        for block in model.encoder.transformer_blocks:
            x = block(x)
        x   = model.encoder.norm(x)
        emb = x.mean(dim=1).numpy()
        lbl = labels[:, :3].argmax(dim=1).numpy()
        embeddings.append(emb)
        labels_list.append(lbl)
        count += len(images)
        print(f"{count}/{NUM_SAMPLES}")

embeddings = np.concatenate(embeddings)[:NUM_SAMPLES]
labels_arr = np.concatenate(labels_list)[:NUM_SAMPLES]

print("Running UMAP...")
reducer  = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_2d  = reducer.fit_transform(embeddings)

np.savez(OUT_PATH, umap_2d=umap_2d, labels=labels_arr, embeddings=embeddings)
print(f"Saved to {OUT_PATH}")
