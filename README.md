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

```
## Citation

If you use MoAGNN in your work, please cite:

Lin, C. P., Ho, Y. J., Chiu, Y. P., Tang, Y., Paik, Y. S., Chen, G. T., ... & Lee, T. Y. (2026). *MoAGNN: a multi-omics hierarchical graph neural network for subtype classification and prognosis prediction in lung adenocarcinoma*. Briefings in Bioinformatics, 27(1), bbaf735.
