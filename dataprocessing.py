import librosa, librosa.display, numpy as np
import parselmouth
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import pandas as pd

# SAMPLE_RATE = 22050
# N_MELS = 128
# #N_MFCC = 13
# FFT = 2048
# HOP = 512
# SPEC_SIZE = (224, 224)

SAMPLE_RATE = 22050
FFT = 2048
HOP = 512
SPEC_SIZE = (224, 224)

def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # (Iyer et al. pre-processing already trimmed silence/noise)
    return librosa.util.normalize(y)

def extract_spectrogram(y, sr=SAMPLE_RATE, n_fft=FFT, hop_length=HOP):
    """
    Compute Short-Time Fourier Transform (STFT) spectrogram,
    apply logarithmic scaling (10 * log10(|S| / max|S|)), normalize,
    and resize to 224×224 for CNN input.
    """
    # --- 1️⃣ Short-Time Fourier Transform ---
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)  # |S|

    # --- 2️⃣ Convert to decibel (logarithmic) scale ---
    eps = 1e-10
    log_spectrogram = 10 * np.log10(magnitude / (np.max(magnitude) + eps) + eps)

    # --- 3️⃣ Resize for CNN input ---
    spec = librosa.util.fix_length(log_spectrogram, size=SPEC_SIZE[1], axis=1)
    spec = spec[:SPEC_SIZE[0], :]

    # --- 4️⃣ Normalize to [0, 1] for stability ---
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

    return spec.astype(np.float32)

def extract_acoustics_parselmouth(path):
    snd = parselmouth.Sound(str(path))
    pitch = snd.to_pitch()
    # meanF0 = pitch.get_mean() we need to get the mean aka pitch mean aka pitch frequency mean
    meanF0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    minF0 = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "None")
    maxF0 = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "None")
    
    pointProcess = parselmouth.praat.call(pitch, "To PointProcess")
    jitter_local = parselmouth.praat.call(pointProcess, "Get jitter (local)...",0,0,0.0001,0.02,1.3)
    shimmer_local = parselmouth.praat.call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)...",0.01, 75,0.1,4.5)
    hnr_mean = parselmouth.praat.call(hnr, "Get mean...",0,0)
    return np.array([meanF0, minF0, maxF0, jitter_local, shimmer_local, hnr_mean],
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
        spec = extract_spectrogram(y)         # Frequency domain spectrogram
        spec_tensor = torch.tensor(spec).unsqueeze(0)  # (1, F, T)
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

import matplotlib.pyplot as plt

def show_stft_spectrogram(y, sr=SAMPLE_RATE, n_fft=FFT, hop_length=HOP):
    """
    Visualize STFT spectrogram as described in the paper:
    time–frequency domain with log scaling.
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)
    log_spec = 10 * np.log10(magnitude / (np.max(magnitude) + 1e-10) + 1e-10)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_spec, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='log', cmap='magma'
    )
    plt.colorbar(format='%+2.0f dB', label='Amplitude (dB)')
    plt.title('Short-Time Fourier Transform (STFT) Log-Magnitude Spectrogram')
    plt.tight_layout()
    plt.show()

def show_processed_spectrogram(spec):
    """Visualize normalized spectrogram as CNN input."""
    plt.figure(figsize=(8, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    plt.title("Normalized Spectrogram (CNN Input)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.colorbar(label="Normalized Amplitude")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    df = pd.read_csv('dataset_manifest.csv')
    train_loader, val_loader = make_loaders(df, kfold_idx=0, batch_size=8)

    for spec, acoustics, label in train_loader:
        print("Spec:", spec.shape, "Acoustics:", acoustics.shape, "Labels:", label.shape)
        break

    # Visualize a random sample
    import random
    sample_path = df.iloc[random.randint(0, len(df)-1)]['path']
    y = load_audio(sample_path)
    show_stft_spectrogram(y)                 # Fourier-based STFT visualization
    spec = extract_spectrogram(y)
    show_processed_spectrogram(spec)