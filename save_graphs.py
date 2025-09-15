#!/usr/bin/env python3
"""
save_graphs.py
--------------
Convert R-exported CSVs into per-sample PyG graphs (.pt).

Inputs (in --in_dir):
  - gene_order.csv        : column ['gene']
  - edge_list.csv         : columns ['src','dst','weight']
  - sample_labels.csv     : columns ['sample_id','label']
  - expression_matrix.csv : samples x genes, index=sample_id

Outputs (in --out_dir):
  - <sample_id>.pt        : torch_geometric.data.Data(x, edge_index, y)
  - label_vocab.txt       : mapping idx -> class
  - meta_edge_stats.json  : summary info

Usage:
  python save_graphs.py --in_dir graph_export --out_dir graphs_pt
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="2_graph_export")
    ap.add_argument("--out_dir", type=str, default="graphs_pt")
    ap.add_argument("--directed", action="store_true", help="Keep edges directed (default: undirected).")
    ap.add_argument("--add_self_loops", action="store_true")
    return ap.parse_args()

def edge_list_to_index(edges, gene_to_idx, directed=False):
    src = edges['src'].map(gene_to_idx).astype(int).values
    dst = edges['dst'].map(gene_to_idx).astype(int).values
    ei = torch.tensor([src, dst], dtype=torch.long)
    if not directed:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        ei = torch.unique(ei, dim=1)  # remove duplicates
    return ei

def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 讀檔
    genes = pd.read_csv(in_dir / "gene_order.csv")['gene'].astype(str).tolist()
    gene_to_idx = {g:i for i,g in enumerate(genes)}

    edges = pd.read_csv(in_dir / "edge_list.csv")
    edge_index = edge_list_to_index(edges, gene_to_idx, directed=args.directed)

    expr = pd.read_csv(in_dir / "expression_matrix.csv", index_col=0)
    expr = expr.reindex(columns=genes)

    labels = pd.read_csv(in_dir / "sample_labels.csv")
    vocab = {lab:i for i,lab in enumerate(sorted(labels['label'].unique()))}
    label_map = dict(zip(labels['sample_id'], labels['label'].map(vocab)))

    # 逐樣本存檔
    saved = 0
    for sid, row in expr.iterrows():
        if sid not in label_map: continue
        x = torch.tensor(row.values, dtype=torch.float32).view(-1,1)
        y = torch.tensor(label_map[sid], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_features = x.size(1)
        torch.save(data, out_dir / f"{sid}.pt")
        saved += 1

    # 輔助檔
    with open(out_dir / "label_vocab.txt", "w") as f:
        for lab, idx in vocab.items():
            f.write(f"{idx}\t{lab}\n")
    meta = {
        "num_samples": saved,
        "num_genes": len(genes),
        "num_edges": edge_index.size(1),
        "directed": args.directed,
        "classes": {v:k for k,v in vocab.items()}
    }
    with open(out_dir / "meta_edge_stats.json","w") as f:
        json.dump(meta,f,indent=2)

    print(f"✅ Saved {saved} graphs to {out_dir}")

if __name__ == "__main__":
    main()
