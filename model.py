import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== CNN FEATURE EXTRACTOR =====
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                  # -> 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)                   # -> 28x28
        )
        self.proj = nn.Conv2d(128, 64, 1)     # compress channels

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)                      # (B,64,28,28)
        return x

# ===== RNN BRANCH =====
class RNNBranch(nn.Module):
    def __init__(self, input_dim=64*28, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=3, batch_first=True,
                            bidirectional=True, dropout=0.3)

    def forward(self, featmaps):
        B, C, H, W = featmaps.shape
        seq = featmaps.permute(0, 3, 1, 2).reshape(B, W, C*H)
        out, _ = self.lstm(seq)
        return out[:, -1, :]  # (B, 2*hidden_dim)

# ===== MKL BRANCH =====
class MKLBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.fc1, self.fc2 = None, None
        self.out_dim = out_dim

    def forward(self, featmaps):
        B = featmaps.size(0)
        flat = featmaps.view(B, -1)
        if self.fc1 is None:
            in_dim = flat.shape[1]
            self.fc1 = nn.Linear(in_dim, 512).to(flat.device)
            self.fc2 = nn.Linear(512, self.out_dim).to(flat.device)
        x = F.relu(self.fc1(flat))
        x = F.dropout(x, 0.4, training=self.training)
        x = F.relu(self.fc2(x))
        return x

# ===== HYBRID MODEL =====
class HybridCNN_RNN_MKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.rnn_branch = RNNBranch()
        self.mkl_branch = MKLBranch(out_dim=256)
        self.mlp = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        #self.fc_out = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.fc_out = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x):
        feats = self.cnn(x)
        rnn_out = self.rnn_branch(feats)
        mkl_out = self.mkl_branch(feats)
        fused = torch.cat([rnn_out, mkl_out], dim=1)
        refined = self.mlp(fused)
        out = self.fc_out(refined)
        return out
