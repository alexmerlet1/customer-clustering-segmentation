# Customer Clustering (RFM) — Online Retail

This repo performs **customer segmentation** using **RFM features**:

- **Recency**: days since last purchase  
- **Frequency**: number of unique invoices  
- **Monetary**: total spend  

It compares multiple clustering approaches and produces a **business-friendly report** plus figures and CSV outputs.

## What you get

- **A full text report**: `results/clustering_report.txt`
- **Reusable outputs**: cluster assignments + cluster summaries per method (CSV)
- **Figures**: saved under `results/figures/`

## Example figures

### Choosing the number of clusters

![Elbow + silhouette plot](results/figures/optimal_clusters.png)

### Comparing clustering methods (PCA projection)

![Clustering comparison PCA](results/figures/clustering_comparison_pca.png)

### Metrics comparison across methods

![Metrics comparison](results/figures/metrics_comparison.png)

## Quickstart

### 1) Run the report script

```bash
python clustering-report.py
```

Optional flags:

```bash
python clustering-report.py --clusters 4 --min-clusters 4
```

### 2) View outputs

- **Report**: `results/clustering_report.txt`
- **Figures**: `results/figures/*.png`
- **Data**:
  - `results/k_means_clustered_data.csv`, `results/k_means_cluster_summary.csv`
  - `results/dbscan_clustered_data.csv`, `results/dbscan_cluster_summary.csv`
  - `results/hierarchical_clustered_data.csv`, `results/hierarchical_cluster_summary.csv`
  - `results/gmm_clustered_data.csv`, `results/gmm_cluster_summary.csv`

## Data source

The script expects the dataset file to be present at:

- `Online Retail.xlsx`

It loads the workbook, filters invalid rows (missing `CustomerID`, non-positive `Quantity` / `UnitPrice`), then computes RFM per `CustomerID`.

## Methods compared

- **K-Means**
- **DBSCAN**
- **Hierarchical** (Agglomerative, Ward linkage)
- **Gaussian Mixture Model (GMM)**

## Repo layout

- `clustering-report.py`: end-to-end analysis pipeline (RFM → scaling → clustering → evaluation → report + figures + CSVs)
- `customer-cluster.ipynb`: exploratory notebook
- `results/`: generated artifacts (report, figures, and CSVs). See `results/README.md` for details.

## Requirements

The script uses common Python data science libraries, including:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`, `scipy`
- `openpyxl` (Excel reader)

