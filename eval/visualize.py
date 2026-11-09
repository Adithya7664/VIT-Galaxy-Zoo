import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import umap

from models.vit import VIT
from models.mae import MAE
from data.dataset import GalaxyZooDataset
from configs.config import (
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH, IN_CHANNELS,
    DECODER_DIM, DECODER_DEPTH, IMG_DIR, CSV_PATH
)

CHECKPOINT  = "C:/Users/adith/Downloads/mae_epoch70.pth"
OUTPUT_DIR  = "C:/Users/adith/Documents/VIT-Galaxyzoo/eval/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ── load model ────────────────────────────────────────────────────────────────
encoder = VIT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              in_channels=IN_CHANNELS, num_heads=NUM_HEADS, depth=DEPTH)
model = MAE(vit_encoder=encoder, patch_size=PATCH_SIZE,
            num_patches=(IMG_SIZE // PATCH_SIZE) ** 2,
            encoder_embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM,
            decoder_depth=DECODER_DEPTH)

model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print("Model loaded.")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor * std + mean).clamp(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ATTENTION MAPS
# ─────────────────────────────────────────────────────────────────────────────
def plot_attention_maps(num_images=4):
    # hook to capture attention weights from last transformer block
    attn_weights = {}
    def hook_fn(module, input, output):
        # output is (attn_output, attn_weights) when need_weights=True
        attn_weights['last'] = output[1].detach()

    last_block = model.encoder.transformer_blocks[-1]
    handle = last_block.attn.register_forward_hook(hook_fn)

    # patch the attn call to return weights
    original_forward = last_block.attn.forward
    def attn_with_weights(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True
        return original_forward(*args, **kwargs)
    last_block.attn.forward = attn_with_weights

    img_files = sorted(os.listdir(IMG_DIR))[:num_images]
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 4))

    for i, fname in enumerate(img_files):
        img_path = os.path.join(IMG_DIR, fname)
        img_pil  = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_t    = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            x = model.encoder.patch_tokens(img_t)
            cls = model.encoder.cls_token.expand(1, -1, -1)
            x   = torch.cat([cls, x], dim=1)
            x   = x + model.encoder.positional_embedding
            for block in model.encoder.transformer_blocks:
                x = block(x)

        # attn_weights['last']: (1, num_patches+1, num_patches+1)
        attn = attn_weights['last'][0]          # (N+1, N+1)
        cls_attn = attn[0, 1:]                  # CLS row, drop self-attn → (num_patches,)
        h = w = IMG_SIZE // PATCH_SIZE          # 6x6 grid
        attn_map = cls_attn.reshape(h, w).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # upsample to image size
        attn_img = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        attn_np  = np.array(attn_img) / 255.0

        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_pil)
        axes[i, 1].imshow(attn_np, alpha=0.6, cmap='inferno')
        axes[i, 1].set_title("Attention Map")
        axes[i, 1].axis('off')

    # restore original forward
    last_block.attn.forward = original_forward
    handle.remove()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "attention_maps.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Attention maps saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MAE RECONSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_reconstructions(num_images=4):
    img_files = sorted(os.listdir(IMG_DIR))[:num_images]
    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 4))

    for i, fname in enumerate(img_files):
        img_path = os.path.join(IMG_DIR, fname)
        img_t    = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

        with torch.no_grad():
            loss, pred, mask = model(img_t, mask_ratio=0.75)

        # unpatchify prediction → image
        p    = PATCH_SIZE
        h = w = IMG_SIZE // p
        pred_img = pred[0].reshape(h, w, p, p, 3)
        pred_img = pred_img.permute(4, 0, 2, 1, 3).reshape(3, IMG_SIZE, IMG_SIZE)
        pred_img = denormalize(pred_img.cpu())

        # build masked image (visible patches kept, masked patches zeroed)
        target   = model.patchify(img_t)[0]          # (num_patches, patch_dim)
        mask_1d  = mask[0].bool()                     # (num_patches,)
        masked   = target.clone()
        masked[mask_1d] = 0
        masked_img = masked.reshape(h, w, p, p, 3)
        masked_img = masked_img.permute(4, 0, 2, 1, 3).reshape(3, IMG_SIZE, IMG_SIZE)
        masked_img = denormalize(masked_img.cpu())

        orig = denormalize(img_t[0].cpu())

        for col, (title, arr) in enumerate([
            ("Original",      orig.permute(1,2,0).numpy()),
            ("Masked Input",  masked_img.permute(1,2,0).numpy()),
            ("Reconstructed", pred_img.permute(1,2,0).numpy()),
        ]):
            axes[i, col].imshow(arr.clip(0, 1))
            axes[i, col].set_title(title)
            axes[i, col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "reconstructions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Reconstructions saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. UMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_umap(num_samples=5000):
    dataset    = GalaxyZooDataset(img_dir=IMG_DIR, csv_path=CSV_PATH,
                                   transform=transform, labeled=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    embeddings, labels_list = [], []
    count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if count >= num_samples:
                break
            images = images.to(device)
            x = model.encoder.patch_tokens(images)
            for block in model.encoder.transformer_blocks:
                x = block(x)
            x = model.encoder.norm(x)
            emb = x.mean(dim=1).cpu().numpy()
            lbl = labels[:, :3].argmax(dim=1).numpy()
            embeddings.append(emb)
            labels_list.append(lbl)
            count += len(images)

    embeddings = np.concatenate(embeddings)[:num_samples]
    labels_arr = np.concatenate(labels_list)[:num_samples]

    print(f"Running UMAP on {len(embeddings)} embeddings...")
    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    class_names  = ['Smooth', 'Features/Disk', 'Star/Artifact']
    colors       = ['#e74c3c', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, (name, color) in enumerate(zip(class_names, colors)):
        mask = labels_arr == cls_idx
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=color, label=name, alpha=0.5, s=5)

    ax.set_title("UMAP of MAE Encoder Embeddings\n(no labels used during pretraining)", fontsize=13)
    ax.legend(markerscale=4)
    ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "umap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"UMAP saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating attention maps...")
    plot_attention_maps(num_images=4)

    print("Generating reconstructions...")
    plot_reconstructions(num_images=4)

    print("Generating UMAP...")
    plot_umap(num_samples=5000)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}")
