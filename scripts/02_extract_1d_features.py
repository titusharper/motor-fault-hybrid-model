import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motor_fault.features_1d import extract_features

fs = 10240
start_idx = int(1.5 * fs)
end_idx = int(4.5 * fs)
segment_length = 1024

root_dir = REPO_ROOT / "data" / "raw"
index_map_path = REPO_ROOT / "data" / "processed" / "reassigned_index_map.csv"
save_csv_path = REPO_ROOT / "data" / "processed" / "all_segments_1d_features_new.csv"
save_csv_path.parent.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

def resolve_mat_path(label: str, basename_safe: str) -> str | None:
    base_dir = root_dir / f"{label}_start"

    cand = base_dir / (basename_safe + ".mat")
    if cand.exists():
        return str(cand)

    flipped = basename_safe.replace("_", " ")
    cand2 = base_dir / (flipped + ".mat")
    if cand2.exists():
        return str(cand2)

    patt1 = str(base_dir / f"*{basename_safe}*.mat")
    patt2 = str(base_dir / f"*{flipped}*.mat")

    hits = glob.glob(patt1) or glob.glob(patt2)
    if hits:
        hits.sort(key=lambda p: len(os.path.basename(p)))
        return hits[0]

    return None

if not index_map_path.exists():
    raise FileNotFoundError(
        f"Index map not found: {index_map_path}."
    )

idx_df = pd.read_csv(index_map_path)
required_cols = {"split", "label", "filename", "basename", "signal_key", "seg_start"}
missing = required_cols - set(idx_df.columns)
if missing:
    raise ValueError(f"Index map is missing required columns: {missing}")

features_list = []
meta_list = []
mat_cache = {}

for _, row in idx_df.iterrows():
    split = str(row["split"])
    label = str(row["label"])
    filename = str(row["filename"])
    basename = str(row["basename"])
    channel = str(row["signal_key"])
    seg_start = int(row["seg_start"])

    mat_path = resolve_mat_path(label, basename)
    if mat_path is None:
        print(f".mat not found -> label={label}, basename={basename}")
        continue

    cache_key = f"{mat_path}__{channel}"
    if cache_key in mat_cache:
        signal = mat_cache[cache_key]
    else:
        try:
            matdata = scipy.io.loadmat(mat_path)
        except Exception as e:
            print(f"Failed to read .mat: {mat_path} ({e})")
            continue

        if channel not in matdata:
            print(f"Channel '{channel}' not found in {mat_path}")
            continue

        signal_full = np.asarray(matdata[channel]).squeeze().astype(np.float64)
        signal_full = (signal_full - np.mean(signal_full)) / (np.std(signal_full) + EPS)
        signal = signal_full[start_idx:end_idx]
        mat_cache[cache_key] = signal

    if seg_start + segment_length > len(signal):
        print(
            f"Segment exceeds bounds: {os.path.basename(mat_path)}, {channel}, start={seg_start}"
        )
        continue

    segment = signal[seg_start : seg_start + segment_length]
    feat_vec = extract_features(segment)

    features_list.append(feat_vec)

    mat_stem = os.path.splitext(os.path.basename(mat_path))[0]
    meta_list.append([split, label, filename, mat_stem, channel, seg_start])

if not features_list:
    raise RuntimeError("No features were extracted.")

features_arr = np.array(features_list, dtype=np.float64)
features_df = pd.DataFrame(
    features_arr, columns=[f"feat_{i+1}" for i in range(features_arr.shape[1])]
)

meta_df = pd.DataFrame(
    meta_list, columns=["split", "label", "filename", "matfile", "channel", "start_index"]
)

out_df = pd.concat([meta_df, features_df], axis=1)
out_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

print("1D feature extraction completed.")
print(f"Output: {save_csv_path}")
print(f"Total segments: {len(out_df)}")
