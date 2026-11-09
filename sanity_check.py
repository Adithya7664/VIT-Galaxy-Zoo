from models.vit import VIT
import torch

model = VIT(image_size=96, patch_size=16, embed_dim=192, in_channels=3, num_heads=3, depth=6)
x = torch.randn(2, 3, 96, 96)  # batch of 2 images
out = model(x)
print(out.shape)  # should be (2, 192)