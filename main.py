#!/usr/bin/env python3
"""
main.py
-------
Train MoAGNN on custom graphs (.pt) with 5-fold cross-validation.
- Reads folds from data_load.load_5fold_splits (no TU datasets).
- Early stopping on val loss; saves best model per fold.
- NLLLoss (Net outputs log_softmax).
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from networks import Net
from data_load import load_5fold_splits


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument('--data_dir', type=str, required=True, help='Directory containing *.pt graphs')
    p.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio within each fold')
    p.add_argument('--seed', type=int, default=777)

    # model (P = pooling_ratio)
    p.add_argument('--nhid', type=int, default=128)
    p.add_argument('--pooling_ratio', type=float, default=0.35)
    p.add_argument('--dropout_ratio', type=float, default=0.2)

    # train
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=50)

    # misc
    p.add_argument('--save_dir', type=str, default='./runs')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = F.nll_loss(out, batch.y, reduction='sum')
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_fold(args, fold_id, train_set, val_set, test_set, num_features, num_classes):
    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"fold{fold_id}"

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=1)

    # pack args for Net
    net_args = argparse.Namespace(
        num_features=num_features,
        nhid=args.nhid,
        num_classes=num_classes,
        pooling_ratio=args.pooling_ratio,
        dropout_ratio=args.dropout_ratio
    )

    model = Net(net_args).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    best_path = os.path.join(args.save_dir, f'best_{tag}.pth')
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(args.device)
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        val_loss, val_acc = eval_loop(model, val_loader, args.device)
        print(f'[{tag}] Epoch {epoch} | val_loss={val_loss:.4f} acc={val_acc:.4f}')

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f'[{tag}] Early stopping at epoch {epoch}')
                break

    # test with best
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_loss, test_acc = eval_loop(model, test_loader, args.device)
    print(f'[{tag}] TEST  acc={test_acc:.4f} loss={test_loss:.4f}')
    return {'val_loss': best_val, 'test_acc': test_acc, 'test_loss': test_loss}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 5-fold splits
    splits, num_features, num_classes = load_5fold_splits(
        data_dir=args.data_dir, seed=args.seed, val_ratio=args.val_ratio
    )

    fold_metrics = []
    for i, (train_set, val_set, test_set) in enumerate(splits, start=1):
        m = train_one_fold(args, i, train_set, val_set, test_set, num_features, num_classes)
        fold_metrics.append(m)

    # summary
    mean_acc = sum(m['test_acc'] for m in fold_metrics) / len(fold_metrics)
    mean_loss = sum(m['test_loss'] for m in fold_metrics) / len(fold_metrics)
    print('=== 5-Fold Summary ===')
    for i, m in enumerate(fold_metrics, 1):
        print(f'Fold {i}: test_acc={m["test_acc"]:.4f} test_loss={m["test_loss"]:.4f}')
    print(f'Mean:  test_acc={mean_acc:.4f}  test_loss={mean_loss:.4f}')


if __name__ == '__main__':
    main()
