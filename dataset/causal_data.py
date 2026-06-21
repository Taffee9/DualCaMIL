import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedRCMPathDataset(Dataset):
    def __init__(
        self,
        rcm_csv: Path,
        rcm_root: Path,
        path_root: Path,
        patient_ids: list,
        transform,
        max_rcm: int = 32,
        training: bool = True,
        path_csv: Optional[Path] = None,
    ):
        self.transform = transform
        self.max_rcm = max_rcm
        self.training = training
        self.rcm_root = rcm_root
        self.path_root = path_root

        # 1) Load RCM data
        rcm_df = pd.read_csv(rcm_csv)
        rcm_df["ImagePath"] = rcm_df["ImagePath"].str.replace("\\", "/", regex=False)
        rcm_root_str = str(rcm_root).replace("\\", "/")
        self.rcm_groups = defaultdict(list)
        self.labels = {}

        for _, row in rcm_df.iterrows():
            rel = row["ImagePath"]
            pid = Path(rel).parts[-2]  # assume .../PatientID/Image.png
            rel_norm = rel.replace("\\", "/")
            rcm_path = Path(rel_norm) if rel_norm.startswith(rcm_root_str) else rcm_root / rel_norm
            # handle missing/incorrect suffix, default to .png
            if not rcm_path.suffix:
                cand = rcm_path.with_suffix(".png")
                rcm_path = cand if cand.exists() else rcm_path
            elif (not rcm_path.exists()) and rcm_path.suffix.lower() != ".png":
                cand = rcm_path.with_suffix(".png")
                rcm_path = cand if cand.exists() else rcm_path
            self.rcm_groups[pid].append(rcm_path)
            self.labels[pid] = int(row["Label"])

        # 2) Load pathology data (optional) as a label-wise pool (ignore patient ID)
        #    Build a pool: label -> list[path]
        self.path_label_pool = defaultdict(list)
        if training and path_csv is not None and Path(path_csv).exists():
            path_df = pd.read_csv(path_csv)
            path_df["ImagePath"] = path_df["ImagePath"].str.replace("\\", "/", regex=False)
            root_str = str(path_root).replace("\\", "/")

            for _, row in path_df.iterrows():
                rel = row["ImagePath"]
                rel_norm = rel.replace("\\", "/")
                full_path = Path(rel_norm) if rel_norm.startswith(root_str) else path_root / rel_norm
                lbl = int(row["Label"]) if "Label" in row else None
                if lbl is not None:
                    self.path_label_pool[lbl].append(full_path)
        elif training:
            print(f"[Warning] Path CSV not found or None: {path_csv}")

        # 3) Build final index
        self.data = []
        for pid in patient_ids:
            if pid not in self.rcm_groups:
                continue


            path_imgs_list = self.path_label_pool.get(self.labels[pid], [])

            self.data.append(
                {
                    "pid": pid,
                    "label": self.labels[pid],
                    "rcm_paths": self.rcm_groups[pid],
                    "path_paths": path_imgs_list,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- RCM handling ---
        rcm_paths = item["rcm_paths"]
        if self.max_rcm and len(rcm_paths) > self.max_rcm:
            if self.training:
                rcm_paths = random.sample(rcm_paths, self.max_rcm)
            else:
                indices = np.linspace(0, len(rcm_paths) - 1, self.max_rcm, dtype=int)
                rcm_paths = [rcm_paths[i] for i in indices]

        rcm_stack = []
        for p in rcm_paths:
            try:
                with Image.open(p) as img:
                    rcm_stack.append(self.transform(img.convert("RGB")))
            except Exception:
                pass

        if not rcm_stack:
            rcm_stack = [torch.zeros(3, 224, 224)]
        try:
            rcm_tensor = torch.stack(rcm_stack)
        except RuntimeError as e:
            # Fallback on CPU OOM: downsample the list and retry
            if "not enough memory" in str(e).lower() and len(rcm_stack) > 1:
                keep = max(1, len(rcm_stack) // 2)
                rcm_tensor = torch.stack(rcm_stack[:keep])
            else:
                raise

        # --- Path handling (teacher) ---
        path_tensor = torch.zeros(3, 224, 224)  # placeholder
        has_path = False

        if self.training and item["path_paths"]:
            p_path = random.choice(item["path_paths"])
            try:
                with Image.open(p_path) as img:
                    path_tensor = self.transform(img.convert("RGB"))
                    has_path = True
            except Exception:
                pass

        return rcm_tensor, path_tensor, item["label"], item["pid"], has_path


def causal_collate(batch):
    """
    Return:
      rcm_all: [Sum(N), 3, H, W]
      path_all: [B, 3, H, W]
      labels: [B]
      bag_sizes: List[int]
      path_masks: [B] (bool)
    """
    rcm_list, path_list = [], []
    labels, pids, bag_sizes, path_masks = [], [], [], []

    for rcm, path, label, pid, has_path in batch:
        rcm_list.append(rcm)
        path_list.append(path)
        labels.append(label)
        pids.append(pid)
        bag_sizes.append(rcm.shape[0])
        path_masks.append(has_path)

    return (
        torch.cat(rcm_list, dim=0),
        torch.stack(path_list, dim=0),
        torch.tensor(labels, dtype=torch.long),
        pids,
        bag_sizes,
        torch.tensor(path_masks, dtype=torch.bool),
    )
