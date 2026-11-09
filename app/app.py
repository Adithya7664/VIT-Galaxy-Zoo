import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from torchvision import transforms
import streamlit as st
import umap
import io

from models.vit import VIT
from models.mae import MAE
from configs.config import (
    IMG_SIZE, PATCH_SIZE, EMBED_DIM, NUM_HEADS, DEPTH, IN_CHANNELS,
    DECODER_DIM, DECODER_DEPTH
)

CHECKPOINT     = "C:/Users/adith/Downloads/mae_epoch70.pth"
PRECOMP_EMBEDS = "C:/Users/adith/Documents/VIT-Galaxyzoo/eval/outputs/embeddings.npz"
device = torch.device('cpu')

# ── load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
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
    return model

# ── load precomputed UMAP embeddings (cached) ─────────────────────────────────
@st.cache_data
def load_umap_data():
    if not os.path.exists(PRECOMP_EMBEDS):
        return None, None, None
    data = np.load(PRECOMP_EMBEDS)
    return data['umap_2d'], data['labels'], data['embeddings']

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

def get_embedding(model, img_t):
    with torch.no_grad():
        x = model.encoder.patch_tokens(img_t)
        for block in model.encoder.transformer_blocks:
            x = block(x)
        x = model.encoder.norm(x)
        return x.mean(dim=1)  # (1, embed_dim)

def get_attention(model, img_t):
    attn_out = {}
    last_block = model.encoder.transformer_blocks[-1]

    original_forward = last_block.attn.forward
    def attn_with_weights(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True
        return original_forward(*args, **kwargs)
    last_block.attn.forward = attn_with_weights

    handle = last_block.attn.register_forward_hook(
        lambda m, i, o: attn_out.update({'w': o[1].detach()})
    )

    with torch.no_grad():
        x   = model.encoder.patch_tokens(img_t)
        cls = model.encoder.cls_token.expand(1, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + model.encoder.positional_embedding
        for block in model.encoder.transformer_blocks:
            x = block(x)

    last_block.attn.forward = original_forward
    handle.remove()

    attn   = attn_out['w'][0]       # (N+1, N+1)
    cls_attn = attn[0, 1:]          # (num_patches,)
    h = w  = IMG_SIZE // PATCH_SIZE
    attn_map = cls_attn.reshape(h, w).numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)

def get_reconstruction(model, img_t):
    with torch.no_grad():
        _, pred, mask = model(img_t, mask_ratio=0.75)

    p    = PATCH_SIZE
    h = w = IMG_SIZE // p

    pred_img = pred[0].reshape(h, w, p, p, 3)
    pred_img = pred_img.permute(4,0,2,1,3).reshape(3, IMG_SIZE, IMG_SIZE)
    pred_img = denormalize(pred_img)

    target   = model.patchify(img_t)[0]
    mask_1d  = mask[0].bool()
    masked   = target.clone()
    masked[mask_1d] = 0
    masked_img = masked.reshape(h, w, p, p, 3)
    masked_img = masked_img.permute(4,0,2,1,3).reshape(3, IMG_SIZE, IMG_SIZE)
    masked_img = denormalize(masked_img)

    return (masked_img.permute(1,2,0).numpy().clip(0,1),
            pred_img.permute(1,2,0).numpy().clip(0,1))

# ── streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Galaxy MAE Explorer", layout="wide")
st.title("Galaxy MAE Explorer")
st.markdown("Upload a galaxy image to see attention maps, MAE reconstruction, and embedding position.")

model = load_model()

uploaded = st.file_uploader("Upload a galaxy image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded).convert('RGB')
    img_t   = transform(img_pil).unsqueeze(0)

    col0, col1, col2, col3 = st.columns(4)

    # ── original image ─────────────────────────────────────────────────────────
    with col0:
        st.subheader("Original")
        st.image(img_pil.resize((IMG_SIZE, IMG_SIZE)))
        st.caption("Uploaded galaxy image")

    # ── attention map ──────────────────────────────────────────────────────────
    with col1:
        st.subheader("Attention Map")
        attn_map = get_attention(model, img_t)
        orig_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(orig_resized)
        ax.imshow(np.array(attn_map)/255.0, alpha=0.6, cmap='inferno')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        st.image(buf)
        st.caption("Where the model focuses attention")

    # ── reconstruction ─────────────────────────────────────────────────────────
    with col2:
        st.subheader("MAE Reconstruction")
        masked_np, recon_np = get_reconstruction(model, img_t)

        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].imshow(masked_np)
        axes[0].set_title("Masked (25% visible)")
        axes[0].axis('off')
        axes[1].imshow(recon_np)
        axes[1].set_title("Reconstructed")
        axes[1].axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        st.image(buf)
        st.caption("Model reconstructs masked patches from context")

    # ── umap embedding position ────────────────────────────────────────────────
    with col3:
        st.subheader("Embedding Position")
        umap_2d, umap_labels, all_embeddings = load_umap_data()

        emb = get_embedding(model, img_t).numpy()

        if umap_2d is not None:
            # find nearest neighbor in embedding space and use its UMAP coords
            dists     = np.linalg.norm(all_embeddings - emb, axis=1)
            nn_idx    = np.argmin(dists)
            new_point = umap_2d[nn_idx:nn_idx+1]  # reuse existing 2D coord

            class_names = ['Smooth', 'Features/Disk', 'Star/Artifact']
            colors      = ['#e74c3c', '#3498db', '#2ecc71']

            fig, ax = plt.subplots(figsize=(5,5))
            for cls_idx, (name, color) in enumerate(zip(class_names, colors)):
                mask = umap_labels == cls_idx
                ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                           c=color, label=name, alpha=0.3, s=3)
            ax.scatter(new_point[0,0], new_point[0,1],
                       c='yellow', s=150, marker='*', zorder=5, label='Your galaxy')
            ax.legend(markerscale=3, fontsize=8)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close()
            st.image(buf)
            st.caption("Where your galaxy lands in embedding space")
        else:
            st.info("Run eval/save_embeddings.py first to enable UMAP positioning.")
            st.write(f"Raw embedding (first 8 dims): {emb[0, :8].round(3)}")
