import torch
import torch.nn as nn
try:
    from models.vit import VIT, Transformer_Encoder
except ModuleNotFoundError:
    from vit import VIT, Transformer_Encoder

class MAE(nn.Module):
    def __init__(self, vit_encoder:VIT, patch_size, num_patches, encoder_embed_dim, in_channels=3, decoder_dim=96, decoder_depth=4, decoder_heads=3):
        super().__init__()
        self.masked_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.enc_to_dec = nn.Linear(encoder_embed_dim, decoder_dim)
        self.decoder_positional_embedding = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList([Transformer_Encoder(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channels)
        self.encoder = vit_encoder
        self.patch_size = patch_size
        self.num_patches = num_patches
    
    def forward(self, x, mask_ratio=0.75):
        B = x.shape[0]

        # step 1: patch embeddings (no CLS token)
        x_patch = self.encoder.patch_tokens(x)  # (B, num_patches, encoder_dim)

        # step 2: random masking
        # torch.rand gives random scores per patch; argsort gives a random shuffle of indices
        noise = torch.rand(B, self.num_patches, device=x.device)
        shuffle_ids = torch.argsort(noise, dim=1)               # (B, num_patches) — shuffled indices
        restore_ids = torch.argsort(shuffle_ids, dim=1)         # to unshuffle later

        num_visible = int(self.num_patches * (1 - mask_ratio))
        visible_ids = shuffle_ids[:, :num_visible]              # first num_visible are "kept"
        masked_ids  = shuffle_ids[:, num_visible:]              # rest are masked

        # gather only the visible patches — advanced indexing across batch
        visible_patches = torch.gather(
            x_patch, dim=1,
            index=visible_ids.unsqueeze(-1).expand(-1, -1, x_patch.shape[-1])
        )  # (B, num_visible, encoder_dim)

        # step 3: encode visible patches through encoder transformer blocks + norm
        x_enc = visible_patches
        for block in self.encoder.transformer_blocks:
            x_enc = block(x_enc)
        x_enc = self.encoder.norm(x_enc)  # (B, num_visible, encoder_dim)

        # step 4: project encoder dim → decoder dim
        x_enc = self.enc_to_dec(x_enc)  # (B, num_visible, decoder_dim)

        # step 5: build full decoder input — place encoded visible + mask tokens at correct positions
        mask_tokens = self.masked_token.expand(B, self.num_patches - num_visible, -1)
        # concatenate visible + mask tokens, then unshuffle to original patch order
        x_full = torch.cat([x_enc, mask_tokens], dim=1)  # (B, num_patches, decoder_dim)
        x_full = torch.gather(
            x_full, dim=1,
            index=restore_ids.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )  # (B, num_patches, decoder_dim) — back in original spatial order

        # add decoder positional embedding
        x_full = x_full + self.decoder_positional_embedding

        # step 6: decode
        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)
        x_pred = self.decoder_pred(x_full)  # (B, num_patches, patch_size^2 * 3)

        # step 7: compute MSE loss on masked patches only
        # first get the target: patchify the original image
        target = self.patchify(x)  # (B, num_patches, patch_size^2 * 3)

        # loss only on masked positions
        mask = torch.zeros(B, self.num_patches, device=x.device)
        mask.scatter_(1, masked_ids, 1)  # 1 = masked, 0 = visible

        loss = (x_pred - target) ** 2           # (B, num_patches, patch_dim)
        loss = loss.mean(dim=-1)                 # (B, num_patches)
        loss = (loss * mask).sum() / mask.sum()  # mean over masked patches only

        return loss, x_pred, mask

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        h = w = H // p
        x = x.reshape(B, C, h, p, w, p)
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(B, h * w, p * p * C)
        return x