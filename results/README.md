# Advanced Customer Clustering Analysis

This directory contains the results of an advanced customer clustering analysis performed on RFM (Recency, Frequency, Monetary) customer data.

## Files Generated

### Report
- `clustering_report.txt` - Comprehensive text report with analysis results, metrics comparison, and business recommendations

### Visualizations (`figures/` directory)
- `optimal_clusters.png` - Elbow method and Silhouette score analysis to determine optimal number of clusters
- `clustering_comparison_pca.png` - PCA visualization comparing all clustering methods
- `metrics_comparison.png` - Bar charts comparing performance metrics across methods
- `{method}_detailed_analysis.png` - Detailed RFM analysis for the best performing method
- `{method}_3d_visualization.png` - 3D scatter plot of RFM features colored by clusters
- `{method}_cluster_sizes.png` - Bar chart showing distribution of customers across clusters

### Data Files
- `{method}_clustered_data.csv` - Complete dataset with cluster assignments for each method
- `{method}_cluster_summary.csv` - Statistical summary for each cluster

## Clustering Methods Compared

1. **K-Means** - Partition-based clustering algorithm
2. **DBSCAN** - Density-based clustering that can identify noise points
3. **Hierarchical** - Agglomerative clustering with ward linkage
4. **Gaussian Mixture Model (GMM)** - Probabilistic clustering method

## Evaluation Metrics

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (higher is better, range: -1 to 1)
- **Calinski-Harabasz Score**: Ratio of between-cluster dispersion to within-cluster dispersion (higher is better)
- **Davies-Bouldin Score**: Average similarity ratio of each cluster with its most similar cluster (lower is better)

## Usage

Run the analysis script:
```bash
python clustering-report.py
```

The script will:
1. Load and prepare the RFM data from `Online Retail.xlsx`
2. Determine optimal number of clusters
3. Apply all clustering methods
4. Compare methods using multiple metrics
5. Generate comprehensive visualizations
6. Create detailed reports
7. Save all results to this directory

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- openpyxl (for reading Excel files)
