import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.causal_data import PairedRCMPathDataset, causal_collate
from models.causal_net import CausalConvNeXtMIL
from train_causal import set_seed, load_processor, build_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Test CausalConvNeXtMIL (v5) on patient-level RCM.")


    parser.add_argument("--rcm-csv", type=Path, default=Path("Datasets/RCM/DataFile.csv"))
    parser.add_argument("--rcm-root", type=Path, default=Path("Datasets/RCM"))
    parser.add_argument("--path-csv", type=Path, default=Path("Datasets/Path/DataFile.csv"))
    parser.add_argument("--path-root", type=Path, default=Path("Datasets/Path/tiles"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/causal_mil_test"))
    parser.add_argument("--split-file", type=Path, default=Path("outputs/convnextv2_mil_patient/splits.json"))

    # Model config
    parser.add_argument("--model-name", type=str, default="facebook/convnextv2-tiny-1k-224")
    parser.add_argument("--num-classes", type=int, default=2)

    # Inference config
    parser.add_argument("--batch-size", type=int, default=4, help="Patients per batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rcm", type=int, default=16)


    parser.add_argument("--use-pce", action="store_false", help="Disable Patient Context Encoder (PCE)")
    parser.add_argument("--use-pscg", action="store_false", help="Disable Patient-Aware Soft Causal Gate (PSCG)")
    parser.add_argument("--use-cbn", action="store_false", help="Disable Causal Budget Normalization (CBN)")

    # Checkpoint
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints") / "causalv5" / "model.pt",
        help="Checkpoint path for CausalConvNeXtMIL.",
    )

    return parser.parse_args()


def run_test(model, loader, device, num_classes: int, json_path: Path):
    model.eval()
    all_logits = []
    all_labels = []
    all_pids = []

    with torch.no_grad():
        for rcm, path, labels, pids, bag_sizes, path_masks in tqdm(loader, desc="Test", leave=False):
            rcm = rcm.to(device)
            labels = labels.to(device)


            out = model(rcm, bag_sizes, path_images=None)
            logits = out["logits"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_pids.extend(pids)

    if not all_logits:
        print("[warn] No samples in test loader.")
        return None

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = torch.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_pred = probs.argmax(axis=1)

    acc = float((y_pred == y_true).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    auc = None
    if num_classes == 2:
        try:
            auc = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            auc = None

    print(f"AUC: {auc}")
    print(f"ACC: {acc:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")


    patient_dict = {}
    for pid, prob, label, pred in zip(all_pids, probs, y_true, y_pred):
        patient_dict[pid] = {
            "true_label": int(label),
            "pred_label": int(pred),
            "probs": prob.tolist(),
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(patient_dict, f, ensure_ascii=False, indent=2)

    print(f"[info] Patient probabilities saved -> {json_path}")

    return {"auc": auc, "acc": acc, "f1_macro": f1_macro}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)


    proc, size = load_processor(args.model_name)
    eval_tf = build_transforms(proc, size, train=False)


    split_file = args.split_file
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    split_data = json.loads(split_file.read_text(encoding="utf-8"))

    if "test" not in split_data:
        raise KeyError(f"'test' not found in split file: {split_file}")
    test_ids = split_data["test"]


    test_ds = PairedRCMPathDataset(
        args.rcm_csv,
        args.rcm_root,
        args.path_root,
        test_ids,
        eval_tf,
        max_rcm=args.max_rcm,
        training=False,
        path_csv=args.path_csv,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=causal_collate,
        num_workers=2,
        pin_memory=True,
    )


    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = CausalConvNeXtMIL(
        model_name=args.model_name,
        num_classes=args.num_classes,
        use_pce=args.use_pce,
        use_pscg=args.use_pscg,
        use_cbn=args.use_cbn,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    json_path = args.output_dir / "test_patient_probs.json"
    metrics = run_test(model, test_loader, device, args.num_classes, json_path)
    if metrics is not None:
        print(
            "[summary] AUC={auc}, ACC={acc:.4f}, F1-macro={f1:.4f}".format(
                auc=metrics["auc"], acc=metrics["acc"], f1=metrics["f1_macro"]
            )
        )


if __name__ == "__main__":
    main()

