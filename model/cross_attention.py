import torch
import torch.nn as nn
from torch.nn import functional as F

class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, H, W, C = x1.size()  # (Batch, Tokens, Embedding dim)
        x1, x2 = x1.view(B, -1, C), x2.view(B, -1, C)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels:(i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels:(i + 1) * head_value_channels, :]
            context = value.transpose(1, 2) @ key # dk*dv
            attended_value = query @ context.transpose(1, 2)   # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, C, H, W)
        reprojected_value = self.reprojection(aggregated_values).permute(0, 2, 3, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttention(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=2):
        super().__init__()
        self.linear = nn.Linear(in_dim*2, in_dim)

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Cross_Attention(key_dim, value_dim, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))

        self.mlp = skip_ffn((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x1.size()
        norm_1 = self.norm1(self.linear(x1))         # (768)-->(384)
        norm_2 = self.norm1(x2)                      # (384)

        attn = self.attn(norm_1, norm_2)

        residual = x1
        tx = residual + attn
        mx = tx
        # mx = tx + self.mlp(self.norm2(tx))
        return mx

class skip_ffn(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x)) + self.fc1(x)))
        out = self.fc2(ax)
        return out

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.size()
        tx = x.permute(0, 3, 1, 2)      # (B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2).view(B, H, W, -1)