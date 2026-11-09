import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
    def forward(self, x:torch.tensor)->torch.tensor:
        x_res = self.conv1(x)
        x_res = x_res.flatten(2)
        x_res = x_res.transpose(1, 2)
        return x_res        
    
class Transformer_Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim)
        )
        
    def forward(self, x:torch.tensor)->torch.tensor:
        x_prenorm = self.norm1(x)
        x_attn, _ = self.attn(x_prenorm, x_prenorm, x_prenorm)
        x = x + x_attn
        x_norm2 = self.norm2(x)
        x_mlp = self.mlp(x_norm2)
        x = x + x_mlp
        return x

class VIT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, in_channels, num_heads, depth):
        super().__init__()
        self.patch_tokens = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_tokens.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches+1, embed_dim))
        self.transformer_blocks = nn.ModuleList([Transformer_Encoder(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x:torch.tensor)->torch.tensor:
        x_patch_embed = self.patch_tokens(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x_patch_embed], dim=1)
        x_pos_embed = self.positional_embedding
        x = x + x_pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        x_final = self.norm(x)
        return x_final[:, 0]
    
