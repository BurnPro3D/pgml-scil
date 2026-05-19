import torch
import torch.nn as nn
from transformers import VivitConfig, VivitModel


class TubeletReconstruction(nn.Module):
    def __init__(
        self, 
        image_size=224,
        embed_dim=768,
        tubelet_size=[2, 16, 16],
        window_size=32,
        frames_to_predict=1,
        out_channels=1,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.t, self.h, self.w = tubelet_size
        self.window_size = window_size
        self.H, self.W = image_size, image_size
        self.out_channels = out_channels
        self.frames_to_predict = frames_to_predict

        # calculate number of tubelets in each dimension
        self.num_h = self.H // self.h  # 18
        self.num_w = self.W // self.w  # 18
        self.num_t = self.window_size // self.t # 2

        # verify total number of tubelets / tokens matches
        # assert self.num_h * self.num_w * self.num_t == 3136, "Number of tubelets should be 3136=14x14x16"    Image size 224.

        # project each token to its tubelet dimension
        tubelet_dim = self.t * self.h * self.w * self.out_channels
        self.token_to_tubelet = nn.Sequential(
            nn.LayerNorm(self.embed_dim),  # Added normalization
            nn.Linear(self.embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, tubelet_dim),
        )

        # Add temporal projection layers
        # self.temporal_projection = nn.ModuleDict({
        #     str(i): nn.Sequential(
        #         # nn.Conv3d(1, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        #         # nn.LayerNorm([32, seq_length, image_size, image_size]),
        #         # nn.ReLU(inplace=True),
        #         nn.Conv3d(self.seq_length, 1, kernel_size=(seq_length - i + 1, 1, 1))
        #     ) for i in range(1, 6)  # Create projections for 1 to 5 future frames
        # })

        self.temporal_projection = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(window_size - frames_to_predict + 1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )

    def forward(self, x):
        B = x.shape[0]

        # print('TubeletReconstruction input shape:', x.shape)  # (B, 3136, 768)

        # project each token to tubelet dimensions
        tubelets = self.token_to_tubelet(x)    # (B, 3136, 768) -> (B, 3136, 512)
        # print('After token_to_tubelet shape:', tubelets.shape)

        # reshape into spatial-temporal grid of tubelets
        tubelets = tubelets.view(B, self.num_t, self.num_h, self.num_w,  -1)    # (B, 3136, 512) -> (B, 14, 14, 16, 512)
        # print('After reshaping to grid shape:', tubelets.shape)

        # reshape each tubelet into its spatial-temporal dimensions
        # (B, 14, 14, 16, 512) -> (B, 14, 14, 16, 2, 16, 16, 1)
        tubelets = tubelets.view(
            B,  self.num_t,self.num_h, self.num_w, self.t, self.h, self.w, self.out_channels
        )
        # print('After rearranging dimensions shape:', tubelets.shape)


        # rearrange dimensions to form final output
        # (B, 14, 14, 16, 2, 16, 16, 1) -> (B, 32, 1, 224, 224)
        x = tubelets.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
        # print('After permuting shape:', x.shape)
        x = x.view(B, self.window_size, self.out_channels, self.H, self.W)
        # print('After final reshaping shape:', x.shape)

        ####### add code to project (B, 32, 1, H, W) to (B, num_frames_to_predict, 1, H, W) ########
        x = x.transpose(1, 2)    # (B, 32, 1, H, W) -> (B, 1, 32, H, W)
        # print('Before temporal projection shape:', x.shape)
        x = self.temporal_projection(x)    # (B, 1, 32, H, W) -> (B, 1, frames_to_predict, H, W)
        # print('After temporal projection shape:', x.shape)
        x = x.transpose(1, 2)    # (B, out_channels, frames_to_predict, H, W) -> (B, frames_to_predict, out_channels, H, W)
        # print('After transposing back (output) shape:', x.shape)
        # x = x.squeeze(2)    # (B, frames_to_predict, 1, H, W) -> (B, frames_to_predict, H, W)

        return x


class DensityPrediction(nn.Module):

    def __init__(self, config, device='cuda'):
        super().__init__()

        # Initialize models
        vivit_config = VivitConfig(
            image_size=config.image_size,
            num_frames=config.context_len,
            tubelet_size=config.tubelet_size,
            hidden_size=config.embed_dim,
            num_channels=config.in_channels,
            num_hidden_layers=config.num_blocks,
            num_attention_heads=config.num_attn_heads,
        )

        if config.pretrained_path:
            print(f"Loading pretrained ViViT model from {config.pretrained_path}")
            self.vivit_model = VivitModel(vivit_config).from_pretrained(
                config.pretrained_path, 
                ignore_mismatched_sizes=True
            )
        else:
            print("Using randomly initialized ViViT model")
            self.vivit_model = VivitModel(vivit_config)

        self.vivit_model = self.vivit_model.to(device)

        # Enable memory efficient attention if available
        if hasattr(self.vivit_model.config, 'use_memory_efficient_attention'):
            self.vivit_model.config.use_memory_efficient_attention = True

        self.tubelet_model = TubeletReconstruction(
            embed_dim=config.embed_dim,
            tubelet_size=config.tubelet_size,
            window_size=config.context_len,
            frames_to_predict=config.num_pred_frames,
            image_size=config.image_size,
            out_channels=config.out_channels,
            dropout=config.dropout,
        ).to(device)


    def forward(self, x):
        out = self.vivit_model(x)
        # print("ViViT output keys:", out.keys())
        # print("ViViT last hidden state shape:", out['last_hidden_state'].shape)

        # out = out['pooler_output']
        # print("ViViT pooler output shape:", out.shape)

        out = out['last_hidden_state'][:, 1:]
        out = self.tubelet_model(out)
        return out