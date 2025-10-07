import librosa, librosa.display, numpy as np, parselmouth
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import pandas as pd

SAMPLE_RATE = 22050
N_MELS = 128
N_MFCC = 13
FFT = 2048
HOP = 512
SPEC_SIZE = (224, 224)

def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # (Iyer et al. pre-processing already trimmed silence/noise)
    return librosa.util.normalize(y)

def extract_spectrogram(y):
    mel = librosa.feature.melspectrogram(y, sr=SAMPLE_RATE,
                                         n_fft=FFT, hop_length=HOP,
                                         n_mels=N_MELS)
    logmel = librosa.power_to_db(mel)
    # resize to 224×224 for CNN input
    spec = librosa.util.fix_length(logmel, size=SPEC_SIZE[1], axis=1)
    spec = spec[:SPEC_SIZE[0], :]  # crop/pad vertically
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec.astype(np.float32)

def extract_acoustics_parselmouth(path):
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch()
    meanF0 = pitch.get_mean()
    minF0 = pitch.get_minimum()
    maxF0 = pitch.get_maximum()
    jitter_local = snd.to_jitter_local()
    shimmer_local = snd.to_shimmer_local()
    hnr = snd.to_harmonicity_ac().get_mean()
    return np.array([meanF0, minF0, maxF0, jitter_local, shimmer_local, hnr],
                    dtype=np.float32)

class VoicePDDataset(Dataset):
    def __init__(self, df, scaler=None):
        """
        df columns: ['path', 'label']
        scaler: fitted StandardScaler for acoustic features
        """
        self.df = df.reset_index(drop=True)
        self.scaler = scaler or StandardScaler()

        # Pre-fit scaler on all data (as paper standardized acoustic features)
        feats = [extract_acoustics_parselmouth(p) for p in df['path']]
        self.scaler.fit(feats)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path, label = Path(row['path']), int(row['label'])
        y = load_audio(path)
        spec = extract_spectrogram(y)         # (128,224)
        spec_tensor = torch.tensor(spec).unsqueeze(0)  # (1,128,224)
        spec_tensor = torch.nn.functional.interpolate(
            spec_tensor.unsqueeze(0), size=(224,224),
            mode='bilinear', align_corners=False).squeeze(0)

        acoustics = extract_acoustics_parselmouth(path)
        acoustics = self.scaler.transform([acoustics])[0]
        acoustics_tensor = torch.tensor(acoustics, dtype=torch.float32)

        return spec_tensor, acoustics_tensor, torch.tensor(label, dtype=torch.float32)


def make_loaders(df, kfold_idx, batch_size=16):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(df['path'], df['label']))
    train_idx, val_idx = folds[kfold_idx]
    scaler = StandardScaler()

    train_ds = VoicePDDataset(df.iloc[train_idx], scaler)
    val_ds   = VoicePDDataset(df.iloc[val_idx], scaler)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size))

if __name__ == "__main__":
    df = pd.read_csv('dataset_manifest.csv')
    train_loader, val_loader = make_loaders(df, kfold_idx=0, batch_size=8)

    for spec, acoustics, label in train_loader:
        print(spec.shape, acoustics.shape, label.shape)
        break