import os
import zipfile
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────────────────────────────────
# EXTRACT IMAGES (Kaggle stores them zipped)
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
IMG_SIZE       = 96
PATCH_SIZE     = 16
EMBED_DIM      = 192
NUM_HEADS      = 3
DEPTH          = 6
IN_CHANNELS    = 3
MASK_RATIO     = 0.75
DECODER_DIM    = 96
DECODER_DEPTH  = 4
DECODER_HEADS  = 3
BATCH_SIZE     = 64
EPOCHS         = 75
LR             = 1.5e-4

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
# VIT ENCODER
# ─────────────────────────────────────────────
class VIT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, in_channels, num_heads, depth):
        super().__init__()
        self.patch_tokens       = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches        = self.patch_tokens.num_patches
        self.cls_token          = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm               = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_patch = self.patch_tokens(x)
        cls     = self.cls_token.expand(x.shape[0], -1, -1)
        x       = torch.cat([cls, x_patch], dim=1)
        x       = x + self.positional_embedding
        for block in self.transformer_blocks:
            x = block(x)
        return self.norm(x)[:, 0]

# ─────────────────────────────────────────────
# MAE
# ─────────────────────────────────────────────
class MAE(nn.Module):
    def __init__(self, vit_encoder, patch_size, num_patches, encoder_embed_dim,
                 in_channels=3, decoder_dim=96, decoder_depth=4, decoder_heads=3):
        super().__init__()
        self.encoder                  = vit_encoder
        self.patch_size               = patch_size
        self.num_patches              = num_patches
        self.masked_token             = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.enc_to_dec               = nn.Linear(encoder_embed_dim, decoder_dim)
        self.decoder_positional_embedding = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.decoder_blocks           = nn.ModuleList([TransformerBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.decoder_norm             = nn.LayerNorm(decoder_dim)
        self.decoder_pred             = nn.Linear(decoder_dim, patch_size ** 2 * in_channels)

    def forward(self, x, mask_ratio=0.75):
        B = x.shape[0]

        x_patch = self.encoder.patch_tokens(x)

        noise       = torch.rand(B, self.num_patches, device=x.device)
        shuffle_ids = torch.argsort(noise, dim=1)
        restore_ids = torch.argsort(shuffle_ids, dim=1)

        num_visible = int(self.num_patches * (1 - mask_ratio))
        visible_ids = shuffle_ids[:, :num_visible]
        masked_ids  = shuffle_ids[:, num_visible:]

        visible_patches = torch.gather(
            x_patch, dim=1,
            index=visible_ids.unsqueeze(-1).expand(-1, -1, x_patch.shape[-1])
        )

        x_enc = visible_patches
        for block in self.encoder.transformer_blocks:
            x_enc = block(x_enc)
        x_enc = self.encoder.norm(x_enc)
        x_enc = self.enc_to_dec(x_enc)

        mask_tokens = self.masked_token.expand(B, self.num_patches - num_visible, -1)
        x_full = torch.cat([x_enc, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, dim=1,
            index=restore_ids.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )
        x_full = x_full + self.decoder_positional_embedding

        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full  = self.decoder_norm(x_full)
        x_pred  = self.decoder_pred(x_full)

        target = self.patchify(x)

        mask = torch.zeros(B, self.num_patches, device=x.device)
        mask.scatter_(1, masked_ids, 1)

        loss = (x_pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss, x_pred, mask

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        h = w = H // p
        x = x.reshape(B, C, h, p, w, p)
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(B, h * w, p * p * C)
        return x

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class GalaxyZooDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, labeled=False):
        self.img_dir  = img_dir
        self.transform = transform
        self.labeled  = labeled

        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.jpg')
        ])

        if labeled:
            df        = pd.read_csv(csv_path)
            prob_cols = [c for c in df.columns if c != 'GalaxyID']
            self.ids  = df['GalaxyID'].tolist()
            self.labels     = torch.tensor(df[prob_cols].values.astype('float32'))
            self.id_to_label = {gid: i for i, gid in enumerate(self.ids)}
            self.img_paths  = [
                p for p in self.img_paths
                if int(os.path.splitext(os.path.basename(p))[0]) in self.id_to_label
            ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image    = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labeled:
            galaxy_id = int(os.path.splitext(os.path.basename(img_path))[0])
            label     = self.labels[self.id_to_label[galaxy_id]]
            return image, label

        return image

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

dataset    = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH, transform=transform, labeled=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
print(f"Dataset size: {len(dataset)} images")

encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              in_channels=IN_CHANNELS, num_heads=NUM_HEADS, depth=DEPTH)
model   = MAE(vit_encoder=encoder, patch_size=PATCH_SIZE,
              num_patches=(IMG_SIZE // PATCH_SIZE) ** 2,
              encoder_embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
              decoder_depth=DECODER_DEPTH).to(device)

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
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch: {epoch+1} | Loss: {loss.item():.4f} | Batch: {batch_idx}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"/kaggle/working/mae_epoch{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")
