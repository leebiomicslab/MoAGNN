#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(WGCNA)
  library(tidyverse)
})

# ---------- CLI 參數 ----------
args <- commandArgs(trailingOnly = TRUE)
# 預設參數（可在 CLI 覆寫）
opts <- list(
  expr_csv = "RNAseq_pre.csv",    # 樣本 x 基因
  pheno_csv = "luad_clinical.csv", # 至少要含 sample rownames + 一個標籤欄位
  label_col = "expression_subtype",          # 要輸出的標籤欄位名稱
  out_dir = "graph_export",                # 輸出資料夾
  thresh_percentile = 0.75,                  # TOM 的門檻分位數 (e.g., 0.75 = P75)
  soft_power = NA,                           # 若 NA 就自動依樣本數給建議值
  network_type = "signed"                    # WGCNA adjacency 類型
)

# 解析 key=value 形式參數
if (length(args) > 0) {
  for (kv in args) {
    if (!grepl("=", kv)) next
    key <- sub("=.*$", "", kv)
    val <- sub("^[^=]*=", "", kv)
    if (key %in% names(opts)) {
      if (key %in% c("thresh_percentile")) val <- as.numeric(val)
      if (key %in% c("soft_power"))        val <- as.numeric(val)
      opts[[key]] <- val
    }
  }
}

# ---------- 輔助：自動 soft power ----------
auto_soft_power <- function(n_samples) {
  if (n_samples < 20) return(18)
  if (n_samples < 30) return(16)
  if (n_samples < 40) return(14)
  return(12)
}

# ---------- 讀資料 ----------
message("Reading expression from: ", opts$expr_csv)
expr <- read.csv(opts$expr_csv, header = TRUE, row.names = 1, check.names = FALSE)

# 轉 log2(x+1)（若已經轉過可以拿掉）
if (max(as.matrix(expr), na.rm = TRUE) > 50) {
  expr <- log2(expr + 1)
}

message("Reading phenotype from: ", opts$pheno_csv)
pheno <- read.csv(opts$pheno_csv, header = TRUE, row.names = 1, check.names = FALSE)

# （可選）你原本的 subtype 正規化邏輯
if (opts$label_col == "expression_subtype" && "expression_subtype" %in% colnames(pheno)) {
  pheno$expression_subtype[pheno$expression_subtype == "prox.-prolif."] <- "PP"
  pheno$expression_subtype[pheno$expression_subtype == "prox.-inflam"]  <- "PI"
}

# 對齊樣本
common <- intersect(rownames(expr), rownames(pheno))
expr  <- expr [common, , drop = FALSE]
pheno <- pheno[common, , drop = FALSE]

# ---------- 品質檢查 + 去除不良基因 ----------
gsg <- goodSamplesGenes(expr)
if (!gsg$allOK) {
  expr <- expr[, gsg$goodGenes, drop = FALSE]
  message("Removed bad genes; kept: ", ncol(expr))
}

# ---------- 取得 soft power ----------
n_samples <- nrow(expr)
sp <- if (is.na(opts$soft_power)) auto_soft_power(n_samples) else opts$soft_power
message("Using soft power: ", sp, " (", opts$network_type, ")")

# ---------- adjacency + TOM → 邊清單 ----------
# 注意：WGCNA 期望行為樣本、列為基因（此處 expr 已是樣本 x 基因）
adj <- adjacency(expr, power = sp, type = opts$network_type)
TOM <- TOMsimilarity(adj)

# 依分位數設定門檻
tau <- as.numeric(quantile(TOM, opts$thresh_percentile))
message("Using TOM threshold (percentile=", opts$thresh_percentile, "): ", signif(tau, 4))

# 只取上三角，去除自環
idx <- which(TOM > tau, arr.ind = TRUE)
idx <- idx[idx[,1] < idx[,2], , drop = FALSE]

genes <- colnames(expr)
edge_list <- data.frame(
  src = genes[idx[,1]],
  dst = genes[idx[,2]],
  weight = TOM[idx],
  stringsAsFactors = FALSE
)

# ---------- 標籤與輸出矩陣 ----------
if (!(opts$label_col %in% colnames(pheno))) {
  stop("label_col '", opts$label_col, "' not found in phenotype columns.")
}
labels <- data.frame(
  sample_id = rownames(pheno),
  label = as.character(pheno[[opts$label_col]]),
  stringsAsFactors = FALSE
)

# ---------- 寫檔 ----------
dir.create(opts$out_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(data.frame(gene = genes),
          file = file.path(opts$out_dir, "gene_order.csv"),
          row.names = FALSE)

write.csv(edge_list,
          file = file.path(opts$out_dir, "edge_list.csv"),
          row.names = FALSE)

write.csv(labels,
          file = file.path(opts$out_dir, "sample_labels.csv"),
          row.names = FALSE)

write.csv(expr,
          file = file.path(opts$out_dir, "expression_matrix.csv"),
          row.names = TRUE)

# ---------- 簡單統計 ----------
edge_density <- nrow(edge_list) / (length(genes) * (length(genes) - 1) / 2)
cat(paste0(
  "\n✅ Export finished @ ", opts$out_dir, "\n",
  "Samples        : ", nrow(expr), "\n",
  "Genes (nodes)  : ", length(genes), "\n",
  "Edges          : ", nrow(edge_list), " (undirected, TOM > ", signif(tau, 4), ")\n",
  "Edge density   : ", signif(edge_density, 3), "\n",
  "Label column   : ", opts$label_col, "\n"
))
