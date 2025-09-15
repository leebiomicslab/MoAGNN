# MoAGNN: Hierarchical Multi‑omics Graph Neural Network with SAGPooling


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
