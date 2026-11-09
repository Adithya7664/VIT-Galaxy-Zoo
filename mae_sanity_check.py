import sys
sys.path.append(r'C:\Users\adith\Documents\VIT-Galaxyzoo\models')

import torch
from models.vit import VIT
from models.mae import MAE

encoder = VIT(image_size=96, patch_size=16, embed_dim=192, in_channels=3, num_heads=3, depth=6)
model = MAE(encoder, patch_size=16, num_patches=36, encoder_embed_dim=192)

x = torch.randn(2, 3, 96, 96)
loss, pred, mask = model(x)

print(loss.shape)   # scalar — ()
print(pred.shape)   # (2, 36, 768)  — 16*16*3 = 768
print(mask.shape)   # (2, 36)