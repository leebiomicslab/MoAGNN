#!/usr/bin/env python3
"""
data_load.py
------------
Load custom PyG graphs (.pt) from a directory and create 5-fold splits.
- Only supports 5-fold CV (as per paper).
- No TU datasets.
- No hardcoded paths.

Returns:
  splits: List[Tuple[Subset, Subset, List[Data]]]
          Each item is (train_set, val_set, test_set)
  num_features: int
  num_classes:  int

Usage (preview stats):
  python data_load.py --data_dir graphs_pt --seed 777 --val_ratio 0.2
"""

import os
import argparse
import torch
from typing import List, Tuple
from torch.utils.data import random_split, Subset
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data

def _load_pt_dir(data_dir: str) -> List[Data]:
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))
    if not files:
        raise RuntimeError(f"No .pt files found in {data_dir}")
    dataset = [torch.load(os.path.join(data_dir, f)) for f in files]
    return dataset

def _infer_shapes(dataset: List[Data]) -> Tuple[int, int]:
    # num_features
    if hasattr(dataset[0], "num_features") and dataset[0].num_features is not None:
        num_features = int(dataset[0].num_features)
    else:
        num_features = int(dataset[0].x.size(-1))

    # num_classes (from y scalar per-graph)
    labels = [int(d.y.item()) for d in dataset]
    num_classes = len(set(labels))
    return num_features, num_classes

def load_5fold_splits(
    data_dir: str,
    seed: int = 777,
    val_ratio: float = 0.2,
) -> Tuple[List[Tuple[Subset, Subset, List[Data]]], int, int]:
    """
    Create 5-fold splits:
      - For each fold: hold out that fold as TEST
      - Split remaining (train+val) by val_ratio into TRAIN / VAL
    """
    dataset = _load_pt_dir(data_dir)
    num_features, num_classes = _infer_shapes(dataset)

    y = [int(d.y.item()) for d in dataset]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    splits: List[Tuple[Subset, Subset, List[Data]]] = []
    all_indices = list(range(len(dataset)))

    for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(all_indices, y), start=1):
        train_val_idx = list(train_val_idx)
        test_idx = list(test_idx)

        # Build train/val split from train_val_idx by ratio
        gen = torch.Generator().manual_seed(seed + fold_id)
        n_tv = len(train_val_idx)
        n_train = int(n_tv * (1.0 - val_ratio))
        n_val = n_tv - n_train

        # Use Subset with explicit indices to avoid materializing big copies
        tv_subset = Subset(dataset, train_val_idx)
        train_set, val_set = random_split(tv_subset, [n_train, n_val], generator=gen)

        test_set = [dataset[i] for i in test_idx]  # keep as list for per-graph evaluation

        splits.append((train_set, val_set, test_set))

    return splits, num_features, num_classes

# ---------- CLI Preview ----------
def _print_dataset_summary(dataset: List[Data]):
    print(f"Number of graphs: {len(dataset)}")
    g0 = dataset[0]
    print(f"Number of features: {g0.num_features if hasattr(g0,'num_features') else g0.x.size(-1)}")
    labels = [int(d.y.item()) for d in dataset]
    print(f"Number of classes: {len(set(labels))}")
    print("\nFirst graph summary:")
    print(g0)
    print("-" * 46)
    print(f"Nodes: {g0.num_nodes} | Edges: {g0.num_edges} "
          f"| Avg degree: {g0.num_edges / max(g0.num_nodes,1):.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing *.pt graphs")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Within-fold validation ratio")
    args = ap.parse_args()

    dataset = _load_pt_dir(args.data_dir)
    _print_dataset_summary(dataset)

    splits, num_features, num_classes = load_5fold_splits(
        data_dir=args.data_dir, seed=args.seed, val_ratio=args.val_ratio
    )

    print("\n=== 5-Fold summary ===")
    for i, (tr, va, te) in enumerate(splits, start=1):
        n_tr = len(tr)
        n_va = len(va)
        n_te = len(te)
        print(f"Fold {i}: train={n_tr} | val={n_va} | test={n_te}")
    print(f"\nnum_features={num_features} | num_classes={num_classes}")

if __name__ == "__main__":
    main()
