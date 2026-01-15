import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from motor_fault.augment import simple_aug, IMAGENET_MEAN, IMAGENET_STD


class FusionDataset(Dataset):
    def __init__(self, df, img_root, split, label_list, feature_cols, augment=False):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_root = str(img_root)
        self.labels = label_list
        self.augment = augment
        self.feature_cols = feature_cols

        if "filename" not in self.df.columns:
            raise ValueError("Missing 'filename' column in the input dataframe.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_dir = os.path.join(self.img_root, row["split"], row["label"])
        img_name = row["filename"]
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0

        if self.augment:
            img = simple_aug(img)
            img = img.copy()

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # CHW

        x1 = row[self.feature_cols].values.astype(np.float32)
        y = self.labels.index(row["label"])

        return (
            torch.from_numpy(img),
            torch.from_numpy(x1),
            torch.tensor(y, dtype=torch.long),
        )

