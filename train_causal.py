import argparse
import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor
from collections import Counter, defaultdict
from models.causal_net import CausalConvNeXtMIL
from dataset.causal_data import PairedRCMPathDataset, causal_collate
from torchvision import transforms


# ================= Utils =================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_processor(model_name):
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        size = 224
        if hasattr(processor, "size"):
            size = processor.size.get("height", 224)
    except:
        processor = None
        size = 224
    return processor, size


def build_transforms(processor, image_size, train=True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


@dataclass
class Splits:
    train: List[str]
    val: List[str]
    test: List[str]


def stratified_split(label_dict: Dict[str, int], seed: int) -> Splits:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for pid, label in label_dict.items():
        by_label[label].append(pid)

    train, val, test = [], [], []
    for _, pids in by_label.items():
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(n * 0.7)
        n_val = int(n * 0.15)
        train.extend(pids[:n_tr])
        val.extend(pids[n_tr : n_tr + n_val])
        test.extend(pids[n_tr + n_val :])

    return Splits(train, val, test)


def get_patient_labels(csv_path: Path) -> Dict[str, int]:
    df = pd.read_csv(csv_path)
    df["ImagePath"] = df["ImagePath"].str.replace("\\", "/", regex=False)
    patient_labels = {}
    for _, row in df.iterrows():
        rel = row["ImagePath"]
        pid = Path(rel).parts[-2]
        patient_labels[pid] = int(row["Label"])
    return patient_labels


# ================= Evaluation Loop =================
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for rcm, path, labels, pids, bag_sizes, path_masks in tqdm(loader, desc="Eval", leave=False):
            rcm = rcm.to(device)
            labels = labels.to(device)

            out = model(rcm, bag_sizes, path_images=None)
            loss = criterion(out["logits"], labels)

            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    if total > 0:
        print(f"  [Debug] Pred distribution: {Counter(all_preds)}")

    return {
        "loss": total_loss / total if total > 0 else 0,
        "acc": correct / total if total > 0 else 0,
    }


# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    # Data Config
    parser.add_argument("--rcm-csv", type=Path, default=Path("Datasets/RCM/DataFile.csv"))
    parser.add_argument("--rcm-root", type=Path, default=Path("Datasets/RCM"))
    parser.add_argument("--path-csv", type=Path, default=Path("Datasets/Path/DataFile.csv"), help="Pathology CSV")
    parser.add_argument("--path-root", type=Path, default=Path("Datasets/Path/tiles"), help="Pathology Root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/causal_mil"))
    parser.add_argument("--split-file", type=Path, default=Path("outputs/convnextv2_mil_patient/splits.json"))

    # Model Config
    parser.add_argument("--model-name", type=str, default="facebook/convnextv2-tiny-1k-224")
    parser.add_argument("--num-classes", type=int, default=2)

    # Train Config
    parser.add_argument("--batch-size", type=int, default=4, help="Patients per batch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rcm", type=int, default=16)

    # Causal Loss Weights
    parser.add_argument("--w-rec", type=float, default=1.0)
    parser.add_argument("--w-orth", type=float, default=0.2)
    parser.add_argument("--w-align", type=float, default=0.1)
    parser.add_argument("--w-inv", type=float, default=1.0)

    # Gate regularization for PSCG
    parser.add_argument("--gate-mu", type=float, default=0.15,help="target mean of PSCG gate")
    parser.add_argument("--w-gate", type=float, default=0.03,help="weight of gate mean regularizer")


    parser.add_argument("--use-pce", action="store_false", help="Disable Patient Context Encoder (PCE)")
    parser.add_argument("--use-pscg", action="store_false", help="Disable Patient-Aware Soft Causal Gate (PSCG)")
    parser.add_argument("--use-cbn", action="store_false", help="Disable Causal Budget Normalization (CBN)")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Splits
    split_file = args.split_file
    if split_file.exists():
        print(f"[Info] Loading splits from {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            splits = Splits(**data)
    else:
        print(f"[Info] Generating new splits based on RCM data...")
        p_labels = get_patient_labels(args.rcm_csv)
        splits = stratified_split(p_labels, args.seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(splits.__dict__, f, indent=2)

    # 2. Datasets
    proc, size = load_processor(args.model_name)
    train_tf = build_transforms(proc, size, train=True)
    eval_tf = build_transforms(proc, size, train=False)

    train_ds = PairedRCMPathDataset(
        args.rcm_csv, args.rcm_root, args.path_root,
        splits.train, train_tf, max_rcm=args.max_rcm, training=True, path_csv=args.path_csv
    )
    val_ds = PairedRCMPathDataset(
        args.rcm_csv, args.rcm_root, args.path_root,
        splits.val, eval_tf, max_rcm=args.max_rcm, training=False, path_csv=args.path_csv
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=causal_collate,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=causal_collate,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train Patients: {len(train_ds)}, Val Patients: {len(val_ds)}")

    # 3. Model
    model = CausalConvNeXtMIL(
        model_name=args.model_name,
        num_classes=args.num_classes,
        # PIC-MIL++: c_rcm -> [PIC-MIL++] -> Attention-MIL
        use_pce=args.use_pce,
        use_pscg=args.use_pscg,
        use_cbn=args.use_cbn,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Debug info about device/AMP/accum
    print(
        f"[debug] device={device}, cuda_available={torch.cuda.is_available()}, "
        f"amp_enabled={scaler.is_enabled()}, accum_steps={args.accum_steps}"
    )

    best_acc = 0.0

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_meter = 0.0
        train_correct = 0
        train_total = 0

        if epoch <= 5:
            w_rec, w_orth, w_align = 0.0, 0.0, 0.0
        else:
            w_rec, w_orth, w_align = args.w_rec, args.w_orth, args.w_align

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs}")
        for step, (rcm, path, labels, _, bag_sizes, path_masks) in enumerate(pbar):
            rcm, path = rcm.to(device), path.to(device)
            labels, path_masks = labels.to(device), path_masks.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward Pass
                out = model(rcm, bag_sizes, path_images=path)


                if step == 0:
                    if "gate_mean" in out:
                        print(
                            f"[debug][train][epoch {epoch}] "
                            f"gate mean={out['gate_mean'].item():.3f}, "
                            f"min={out['gate_min'].item():.3f}, "
                            f"max={out['gate_max'].item():.3f}"
                        )
                    if "alpha_mean" in out:
                        print(
                            f"[debug][train][epoch {epoch}] "
                            f"alpha mean={out['alpha_mean'].item():.3f}, "
                            f"min={out['alpha_min'].item():.3f}, "
                            f"max={out['alpha_max'].item():.3f}"
                        )
                    if "attn_mean" in out:
                        print(
                            f"[debug][train][epoch {epoch}] "
                            f"attn mean={out['attn_mean'].item():.3f}, "
                            f"min={out['attn_min'].item():.3f}, "
                            f"max={out['attn_max'].item():.3f}"
                        )

                # --- Loss Calculation ---

                loss_cls = criterion(out["logits"], labels)


                loss_rec = F.mse_loss(out["rec_rcm"], out["rcm_raw"])


                c_norm = F.normalize(out["c_rcm"], dim=1)
                s_norm = F.normalize(out["s_rcm"], dim=1)
                dim = min(c_norm.size(1), s_norm.size(1))
                loss_orth = torch.mean((c_norm[:, :dim] * s_norm[:, :dim]).sum(dim=1) ** 2)


                loss_align = torch.tensor(0.0, device=device)
                loss_cf = torch.tensor(0.0, device=device)
                loss_inv = torch.tensor(0.0, device=device)
                if out["c_prime"] is not None:  # Only in training
                    # c_prime (hybrid) should be close to c_rcm (original content)
                    loss_inv = F.mse_loss(out["c_prime"], out["c_rcm"])

                if path_masks.any():
                    valid_c_rcm = out["patient_content"][path_masks]
                    valid_c_path = out["c_path"][path_masks]
                    valid_labels = labels[path_masks]

                    loss_align = F.mse_loss(valid_c_rcm, valid_c_path)
                    loss_cf = criterion(out["cf_logits"][path_masks], valid_labels)


                loss_gate = torch.tensor(0.0, device=device)

                if "gate_mean" in out and args.w_gate > 0:
                     loss_gate = (out["gate_mean"] - args.gate_mu) ** 2

                loss = (
                    loss_cls
                    + w_rec * loss_rec
                    + w_orth * loss_orth
                    + w_align * (loss_align + loss_cf)
                    + args.w_inv * loss_inv
                    + args.w_gate * loss_gate
                )

                loss = loss / args.accum_steps

            if torch.isnan(loss):
                print(f"[warn] NaN loss at epoch {epoch} step {step}")
                break

            # Backprop
            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()

            train_loss_meter += loss.item() * args.accum_steps
            preds = out["logits"].argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(
                {
                    "L_total": f"{loss.item()*args.accum_steps:.3f}",
                    "L_cls": f"{loss_cls.item():.3f}",
                }
            )

        # End of Epoch
        train_loss = train_loss_meter / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Val Acc: {val_metrics['acc']:.4f}"
        )

        # Save Best
        if val_metrics["acc"] >= best_acc:
            best_acc = val_metrics["acc"]
            save_path = args.output_dir / "best"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "model.pt")
            print(f"--> Best model saved with Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
