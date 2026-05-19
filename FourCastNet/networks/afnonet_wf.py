#reference: https://github.com/NVlabs/AFNO-transformer

from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
# from numpy.lib.arraypad import pad
from numpy import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # here first dim is 2 for real and imag part
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        double_skip=True,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        # x = self.drop_path(x)
        x = x + residual
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Image Sequence to Patch Embedding """
    def __init__(self, img_size=(5, 296, 296), patch_size=(1, 8, 8), in_chans=4, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input x shape: (B, C, T, H, W)
        x = self.proj(x)      # Output: (B, embed_dim, T_p, H_p, W_p)
        x = x.flatten(2)       # Output: (B, embed_dim, num_tubelets)
        x = x.transpose(1, 2)  # Output: (B, num_tubelets, embed_dim)
        return x

class AFNONet_Seq2Seq(nn.Module):
    """
    AFNONet adapted for 3D spatiotemporal data.
    
    MODIFIED: This version is now sequence-to-sequence.
    It takes T_in frames and predicts T_out frames, where T_out == T_in.
    (T_in is determined by context_len)
    """
    def __init__(
        self,
        img_size=(296, 296),
        patch_size=(8, 8),
        in_chans=4,
        out_chans=1,
        context_len=5, 
        temporal_patch_size=1, 
        embed_dim=768,
        depth=12,
        mlp_ratio=4.,
        num_blocks=8
    ):
        super().__init__()
        self.patch_size_3d = (temporal_patch_size, patch_size[0], patch_size[1])
        self.img_size_3d = (context_len, img_size[0], img_size[1])
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed3D(
            img_size=self.img_size_3d,
            patch_size=self.patch_size_3d,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        self.h_patches = img_size[0] // patch_size[0]
        self.w_patches = img_size[1] // patch_size[1]
        self.t_patches = context_len // temporal_patch_size
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, num_blocks=num_blocks) 
            for _ in range(depth)
        ])

        # We need to predict P_t * P_h * P_w pixels for every token.
        p_t = self.patch_size_3d[0] # temporal_patch_size
        p_h = self.patch_size_3d[1] # patch_height
        p_w = self.patch_size_3d[2] # patch_width
        
        self.head = nn.Linear(embed_dim, out_chans * p_t * p_h * p_w, bias=False)
        self.alpha = nn.Parameter(torch.full((1,), 0.25)) # Initialize with default 0.25
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input: (B, T_in, C_in, H, W) -> e.g., (32, 5, 4, 296, 296)
        x = x.permute(0, 2, 1, 3, 4)
        # x shape: (B, C_in, T_in, H, W) -> (32, 4, 5, 296, 296)

        x = self.patch_embed(x) + self.pos_embed
        # x shape: (B, num_tubelets, D) -> (32, 27380, 32)
        
        B = x.shape[0]
        
        # Reshape for AFNO blocks
        x = x.reshape(B, self.t_patches, self.h_patches, self.w_patches, self.embed_dim)
        # x shape: (B, 5, 74, 74, 32)
        x = rearrange(x, 'b t h w d -> b (t h) w d')
        # x shape: (B, 370, 74, 32)

        for blk in self.blocks:
            x = blk(x)
        # x shape: (B, 370, 74, 32)
                
        # 1. Apply head to the output of the blocks directly.
        #    The Linear layer will apply to the last dim (embed_dim).
        #    (B, 370, 74, 32) -> (B, 370, 74, 16)
        x = self.head(x) 
        # x = 0.7 * F.sigmoid(x)
        x = F.prelu(x, self.alpha)
        
        # 2. Reshape back to spatiotemporal grid (un-merging t and h)
        x = rearrange(x, 'b (t h) w d -> b t h w d', t=self.t_patches, h=self.h_patches)
        # x shape: (B, T_p, H_p, W_p, C_out*P*P) -> (32, 5, 74, 74, 16)

        # 3. Unpatch to get the final output *sequence*
        x = rearrange(
            x,
            "b t h w (p_t p_h p_w c_out) -> b (t p_t) c_out (h p_h) (w p_w)",
            p_t=self.patch_size_3d[0],
            p_h=self.patch_size_3d[1],
            p_w=self.patch_size_3d[2],
            h=self.h_patches,
            w=self.w_patches,
            t=self.t_patches
        )
        # Final shape: (B, T_out, C_out, H, W) -> (32, 5, 1, 296, 296)
        
        
        return x


class AFNONet_Residual(nn.Module):
    """
    A wrapper for AFNONet_Seq2Seq that implements residual learning.
    It adds the input 'source_map' (baseline physics) to the model's output.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        # x shape: (B, T_in, C_in, H, W)
        
        # 1. Extract the Baseline Physics (Source Map)
        # Your description says 'source_map' is the first channel (index 0).
        # We slice it to keep dimensions: (B, T_in, 1, H, W)
        baseline = x[:, :, 0:1, :, :] 
        
        # 2. Run the Backbone Model
        # This predicts the 'correction' term (wind effects, etc.)
        # output shape: (B, T_out, 1, H, W)
        correction = self.backbone(x)
        
        # 3. Add the Residual Connection
        # Final Prediction = Baseline + Correction
        # Note: If T_out != T_in (due to temporal_patch_size > 1), this addition
        # might fail or require slicing the baseline. 
        # With temporal_patch_size=1, T_out == T_in, so it works perfectly.
        return baseline + correction



if __name__ == "__main__":
    model = AFNONet(
        img_size=(296, 296), 
        patch_size=(4,4), 
        in_chans=4, 
        out_chans=1
    )
    sample = torch.randn(1, 4, 296, 296)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))
