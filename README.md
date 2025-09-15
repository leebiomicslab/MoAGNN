# MoAGNN: Hierarchical Multi‑omics Graph Neural Network with SAGPooling
We propose MoAGNN, a gene-centric hierarchical graph neural network that integrates transcriptome, miRNA, methylation, and CNV data within a biologically informed framework. Incorporating regulatory directions and attention-based hierarchical pooling, MoAGNN models gene-level dynamics to provide interpretable multi-omics integration.

## 📦 Project Structure
```
save_graphs.py   # Convert CSVs → PyG graphs (.pt)
data_load.py     # 5-fold stratified split loader
networks.py      # MoAGNN (GCN + SAGPool + MLP)
layers.py        # SAGPool layer
main.py          # Training & evaluation
environment.yml  # Conda environment
```

```
conda env create -f environment.yml
conda activate moagnn
```

