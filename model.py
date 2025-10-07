"""
The pytorch implementation of the champion model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.proj = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        x = self.conv(x)          # (B,128,28,28)
        x = self.proj(x)          # (B,64,28,28)
        return x                  # feature maps


class RNNBranch(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)

    def forward(self, featmaps):
        B, C, H, W = featmaps.shape
        seq = featmaps.view(B, C, H*W).permute(0, 2, 1)  # (B,T=H*W,C)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]  # (B, 2*hidden_dim)
        return last


class MKLBranch(nn.Module):
    """Approximates Multiple Kernel Learning as a learnable dense projection"""
    def __init__(self, in_dim=64*28*28, out_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, out_dim),
            nn.ReLU()
        )

    def forward(self, featmaps):
        B, C, H, W = featmaps.shape
        flat = featmaps.view(B, -1)
        out = self.fc(flat)
        return out


class HybridCNN_RNN_MKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.rnn_branch = RNNBranch(input_dim=64)
        self.mkl_branch = MKLBranch(in_dim=64*28*28, out_dim=256)

        combined_dim = 256 + 256  # RNN(2*128) + MKL(256)
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_out = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.cnn(x)
        rnn_out = self.rnn_branch(feats)  # (B,256)
        mkl_out = self.mkl_branch(feats)  # (B,256)
        fused = torch.cat([rnn_out, mkl_out], dim=1)
        refined = self.mlp(fused)
        out = self.fc_out(refined)
        return out
