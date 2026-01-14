from pathlib import Path
import shutil, os
from collections import defaultdict
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
root_dir = REPO_ROOT / "data" / "raw"
output_dir = REPO_ROOT / "data" / "processed" / "reassigned_new_1p5_4p5s"
csv_out = REPO_ROOT / "data" / "processed" / "reassigned_index_map.csv"

if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
labels = ['healthy', 'imbalance', 'loose', 'misalignment']
class_segments = defaultdict(list)

for label in labels:
    folder_path = root_dir / f"{label}_start"
    mat_files = [p for p in folder_path.glob("*.mat")]

    for path in tqdm(mat_files, desc=f"Loading: {label}"):
        data = loadmat(path)
        filename_lower = path.name.lower()
        if 'cur1' in filename_lower:
            signal_key = 'cur1'
        elif 'cur2' in filename_lower:
            signal_key = 'cur2'
        else:
            # fallback
            if 'cur1' in data:
                signal_key = 'cur1'
            elif 'cur2' in data:
                signal_key = 'cur2'
            else:
                continue

        if signal_key not in data:
            continue

        signal = data[signal_key].squeeze().astype(np.float64)
        # Z-normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        signal = signal[start_idx:end_idx]

        for j in range(0, len(signal) - segment_length + 1, hop):
            segment = signal[j:j + segment_length]
            class_segments[label].append((path.stem, signal_key, j, segment))

min_count = min(len(v) for v in class_segments.values())
print(f"Equal segment length for each class: {min_count}")

for sp in ['train', 'validation', 'test']:
    for lb in labels:
        (output_dir / sp / lb).mkdir(parents=True, exist_ok=True)

index_rows = []

for label, segments in class_segments.items():
    rng = np.random.RandomState(42)
    rng.shuffle(segments)
    selected = segments[:min_count]

    n_train = int(train_ratio * min_count)
    n_val   = int(val_ratio * min_count)
    n_test  = min_count - n_train - n_val

    split_map = {
        'train'     : selected[:n_train],
        'validation': selected[n_train:n_train+n_val],
        'test'      : selected[n_train+n_val:]
    }

    for split, segment_list in split_map.items():
        save_dir = output_dir / split / label
        save_dir.mkdir(parents=True, exist_ok=True)

        for (basename, key, j, segment) in tqdm(segment_list, desc=f"Processing: {split}/{label}"):
            f_axis, t_axis, Sxx = reassigned_spectrogram(
                segment, fs=fs, window='hann', nperseg=512, noverlap=384, nfft=1024
            )

            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            vmin = np.percentile(Sxx_db, 5)
            vmax = np.percentile(Sxx_db, 95)

            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.axis('off')
            ax.pcolormesh(t_axis, f_axis, Sxx_db, shading='gouraud',
                          cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_ylim(0, 2000)
            plt.tight_layout(pad=0)

            safe_base = basename.replace(" ", "_")
            filename = f"{label}__{safe_base}__{key}__{j}.png"
            save_path = save_dir / filename

            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            index_rows.append({
                'split'     : split,
                'label'     : label,
                'filename'  : filename,
                'basename'  : safe_base,
                'signal_key': key,
                'seg_start' : j
            })

# Index CSV
index_df = pd.DataFrame(index_rows)
index_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
print("\nReassigned spectrogram dataset created!")
print(f"Index map: {csv_out}")
print(f"Output directory: {output_dir}")
