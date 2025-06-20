import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, out_size):
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return self.block(x)

class PatchDecoder(nn.Module):
    def __init__(self, emb_dim, out_dim=64):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(emb_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up1 = UpBlock(512, 256)   # -> 64x64
        self.up2 = UpBlock(256, 128)   # -> 128x128
        self.up3 = UpBlock(128, 64)    # -> 256x256
        self.up4 = UpBlock(64, out_dim)  # -> 512x512

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "Input must be square number of patches"

        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        x = self.initial_conv(x)
        x = self.up1(x, (64, 64))
        x = self.up2(x, (128, 128))
        x = self.up3(x, (256, 256))
        x = self.up4(x, (512, 512))
        return x

# ------------------------------
# ▶ Example run

if __name__ == "__main__":
    B = 1
    embed_dim = 64
    out_channels = 32
    num_tokens = 625  # 25x25

    # Random input token sequence
    x = torch.randn(B, num_tokens, embed_dim)

    # Initialize decoder
    decoder = PatchDecoder(emb_dim=embed_dim)

    # Run
    out = decoder(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # Expected: [1, 64, 512, 512]
