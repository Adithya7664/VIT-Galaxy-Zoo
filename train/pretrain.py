import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vit import VIT
from models.mae import MAE
from data.dataset import GalaxyZooDataset
from configs.config import (
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH, IN_CHANNELS,
    MASK_RATIO, DECODER_DIM, DECODER_DEPTH,
    BATCH_SIZE, EPOCHS, LR,
    IMG_DIR, CSV_PATH
)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# dataset + dataloader
dataset = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH, transform=transform, labeled=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
print(f"Dataset size: {len(dataset)} images")

# model
encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              in_channels=IN_CHANNELS, num_heads=NUM_HEADS, depth=DEPTH)
model = MAE(vit_encoder=encoder, patch_size=PATCH_SIZE,
            num_patches=(IMG_SIZE // PATCH_SIZE) ** 2,
            encoder_embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
            decoder_depth=DECODER_DEPTH).to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)
        loss, _, _ = model(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if batch_idx%50 == 0:
            print(f"Epoch :{epoch} | Val_loss :{loss.item(): .4f} batch_idx:{batch_idx}")
    avg_loss = total_loss/len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"mae_epoch{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")

        