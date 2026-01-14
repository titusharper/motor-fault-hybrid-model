import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft
from PIL import Image
from tqdm import tqdm
import shutil
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import random
import pandas as pd

# =============== Reproducibility ===============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

fs = 10240
start_idx = int(1.5 * fs)
end_idx   = int(4.5 * fs)
segment_length = 1024
hop = 512
# 70% train / 10% validation / 20% test
train_ratio = 0.7
val_ratio   = 0.1  
Ts = 1 / fs

# =============== Reassignment method ===============
def reassigned_spectrogram(x, fs, window='hann', nperseg=512, noverlap=384, nfft=1024):
    # Standard STFT
    f, t, Zxx = stft(x, fs=fs, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft, return_onesided=True)
    # Meshgrid
    T, F = np.meshgrid(t, f)
    # Time derivative (finite difference)
    t_derivative = np.zeros_like(Zxx, dtype=np.complex128)
    t_derivative[:, 1:-1] = (Zxx[:, 2:] - Zxx[:, :-2]) / (t[2] - t[0])
    t_derivative[:, 0]    = (Zxx[:, 1] - Zxx[:, 0])   / (t[1] - t[0])
    t_derivative[:, -1]   = (Zxx[:, -1] - Zxx[:, -2]) / (t[-1] - t[-2])
    # Frequency derivative (finite difference)
    f_derivative = np.zeros_like(Zxx, dtype=np.complex128)
    f_derivative[1:-1, :] = (Zxx[2:, :] - Zxx[:-2, :]) / (f[2:, np.newaxis] - f[:-2, np.newaxis])
    f_derivative[0,  :]   = (Zxx[1,  :] - Zxx[0,  :])  / (f[1] - f[0])
    f_derivative[-1, :]   = (Zxx[-1, :] - Zxx[-2, :])  / (f[-1] - f[-2])
    power = np.abs(Zxx)**2
    # Reassignment operators
    mask = (np.abs(Zxx) > 1e-10)
    t_reassigned = np.copy(T)
    f_reassigned = np.copy(F)

    if np.any(mask):
        t_reassigned[mask] = T[mask] - np.real(np.conj(Zxx[mask]) * t_derivative[mask]
                                               / (2j * np.pi * fs * np.abs(Zxx[mask])**2))
        f_reassigned[mask] = F[mask] + np.real(np.conj(Zxx[mask]) * f_derivative[mask]
                                               / (2j * np.pi * Ts * np.abs(Zxx[mask])**2))

    f_reassigned = np.clip(f_reassigned, f[0], f[-1])
        t_bins = np.linspace(np.min(t), np.max(t), len(t) * 2)
        f_bins = np.linspace(np.min(f), np.max(f), len(f) * 2)
        dt = t_bins[1] - t_bins[0]
        df = f_bins[1] - f_bins[0]
        reassigned_spec = np.zeros((len(f_bins)-1, len(t_bins)-1), dtype=np.float32)
    
        for i in range(len(f)):
            for j in range(len(t)):
                if mask[i, j]:
                    t_idx = int((t_reassigned[i, j] - t_bins[0]) / dt)
                    f_idx = int((f_reassigned[i, j] - f_bins[0]) / df)
                    if 0 <= t_idx < len(t_bins)-1 and 0 <= f_idx < len(f_bins)-1:
                        reassigned_spec[f_idx, t_idx] += power[i, j].astype(np.float32)
    
        reassigned_spec = gaussian_filter(reassigned_spec, sigma=1.0)
        return f_bins[:-1], t_bins[:-1], reassigned_spec
