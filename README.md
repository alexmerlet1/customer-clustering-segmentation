# Customer Segmentation Using RFM Analysis and Machine Learning

**Alexandre Merlet** · [LinkedIn](https://www.linkedin.com/in/alexmerlet) · [alexmerlet1@gmail.com](mailto:alexmerlet1@gmail.com)

---

## Overview

This project applies machine learning clustering methods to a real retail transaction dataset in order to segment customers into behaviorally distinct groups. The goal is to move beyond aggregate marketing and produce segments that are statistically valid and directly actionable by a CRM or marketing team.

The dataset contains transactions from 4,338 customers across roughly 500,000 line items. Each customer is described using three behavioral features derived from their purchase history: recency, frequency, and monetary value (RFM). Four clustering algorithms were tested and evaluated against three standard metrics. K-Means clustering produced the strongest result and was selected as the final model.

---

## Background: The RFM Framework

RFM is a customer analysis framework widely used in CRM and direct marketing. It summarises a customer's purchase history into three measurable dimensions:

| Metric | Definition |
|--------|-----------|
| **Recency** | Number of days since the customer's last purchase |
| **Frequency** | Total number of unique orders placed |
| **Monetary** | Total spend across all transactions |

These three features are sufficient to distinguish meaningfully different customer types without requiring more complex behavioral or demographic data. The simplicity of RFM also makes it practical: the outputs can be exported directly into most CRM platforms.

---

## Data

**Source:** UCI Machine Learning Repository, Online Retail Dataset  
**Period:** December 2010 to December 2011  
**Retailer:** UK-based e-commerce business  
**Raw transactions:** approximately 500,000 line items  
**Customers after cleaning:** 4,338

The following cleaning steps were applied before feature engineering:

- Removed all records with a missing CustomerID
- Removed transactions with negative quantities (returns) and zero unit prices
- Computed a TotalPrice field as Quantity multiplied by UnitPrice
- Aggregated all transactions to one row per customer containing their RFM values

---

## Methodology

### Feature scaling

RFM values vary considerably in range. Recency is measured in days, while monetary values can reach into the hundreds of thousands. Without scaling, algorithms that rely on distance calculations would be dominated by the monetary dimension. All three features were standardised using scikit-learn's StandardScaler prior to clustering.

### Selecting the number of clusters

Two methods were used to determine an appropriate number of clusters:

- **Elbow method:** inertia (within-cluster sum of squared distances) is plotted against cluster count. The point where the rate of improvement begins to flatten suggests a reasonable number of clusters.
- **Silhouette analysis:** measures how similar each observation is to its own cluster compared to other clusters. Scores range from -1 to 1, with higher values indicating better-defined separation.

A minimum of four clusters was set as a business constraint, since fewer segments are unlikely to provide enough differentiation for a marketing team to act on meaningfully.

### Algorithms compared

Four clustering methods were evaluated:

| Method | Description |
|--------|-------------|
| **K-Means** | Partitions data by minimising within-cluster variance around centroids |
| **DBSCAN** | Density-based method; identifies clusters of arbitrary shape and flags outliers |
| **Hierarchical (Ward)** | Builds a cluster hierarchy by minimising within-cluster variance at each merge |
| **Gaussian Mixture Model (GMM)** | Probabilistic model that assigns soft cluster membership |

Each method was evaluated on three metrics: Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.

### Results

| Method | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|--------|-----------------|-------------------|----------------|
| **K-Means** | **0.6162** | **3,149.72** | **0.7534** |
| Hierarchical | 0.58 | 2,814.3 | 0.81 |
| GMM | 0.54 | 2,103.6 | 0.94 |
| DBSCAN | 0.31 | 487.2 | 2.14 |

K-Means ranked first on all three metrics and was selected as the final model. A silhouette score of 0.6162 indicates strong cluster separation, meaning the four segments reflect genuinely different behavioral groups rather than arbitrary divisions.

DBSCAN performed poorly on this dataset. RFM data tends to form roughly spherical clusters in scaled feature space, which is well-suited to K-Means. DBSCAN is better suited to datasets with irregular cluster shapes or where anomaly detection is a primary objective.

---

## Visualisations

### 1. Optimal cluster selection
The elbow curve and silhouette scores used to determine the final cluster count of four.

![Optimal cluster selection](results/figures/optimal_clusters.png)

### 2. Algorithm comparison (PCA projection)
All four clustering methods projected into two dimensions using Principal Component Analysis, allowing a visual comparison of cluster boundary quality.

![Clustering comparison PCA](results/figures/clustering_comparison_pca.png)

### 3. Evaluation metrics comparison
Side-by-side comparison of all three evaluation metrics across the four algorithms.

![Metrics comparison](results/figures/metrics_comparison.png)

### 4. K-Means segment analysis
RFM heatmap showing average values per cluster, alongside distribution plots for each feature broken down by segment.

![K-Means detailed analysis](results/figures/k-means_detailed_analysis.png)

### 5. Segment size distribution
The proportion of customers assigned to each of the four segments.

![Cluster sizes](results/figures/k-means_cluster_sizes.png)

### 6. 3D RFM visualisation
All customers plotted in three-dimensional RFM space, coloured by segment. The VIP cluster is visually isolated in the high-frequency, high-monetary, low-recency region.

![3D RFM visualization](results/figures/k-means_3d_visualization.png)

---

## Segment Profiles

### Cluster 2: VIP Customers (13 customers, 0.3% of base)

| Recency | Frequency | Average Spend |
|---------|-----------|---------------|
| 7 days | 83 orders | $127,338 |

This segment contains 13 customers who collectively account for over $1.65 million in revenue. They purchase with very high frequency, have an extremely short recency, and their average spend is an order of magnitude above any other segment. Despite their small size, they represent a disproportionate share of total revenue and carry the highest retention risk.

**Recommended actions:**
- Assign a dedicated account contact or relationship manager
- Provide early access to new products ahead of general release
- Invite to private or exclusive brand events
- Treat any service issue as a priority escalation
- For B2B accounts, conduct quarterly reviews; for B2C, invest in personalised communication and gifting

### Cluster 3: Loyal High-Value Customers (204 customers, 4.7%)

| Recency | Frequency | Average Spend |
|---------|-----------|---------------|
| 16 days | 22 orders | $12,709 |

These customers purchase regularly, spend significantly above the base average, and have bought recently. They represent the most likely candidates for migration into the VIP segment given sufficient engagement and incentive.

**Recommended actions:**
- Enrol in a loyalty programme with transparent progress toward a higher tier
- Develop upsell recommendations based on the gap between their purchase patterns and those of VIP customers
- Use referral incentives given their demonstrated engagement with the brand
- Monitor monthly for customers showing an upward trajectory in all three RFM dimensions

### Cluster 0: Moderate Spenders (3,054 customers, 70.4%)

| Recency | Frequency | Average Spend |
|---------|-----------|---------------|
| 44 days | 4 orders | $1,359 |

This segment comprises the majority of the customer base. Customers here are still active but show limited growth in frequency or spend over time. Because of the segment's volume, even a modest increase in average order value would have a greater absolute revenue impact than improvements in any other segment.

**Recommended actions:**
- Introduce bundle offers and volume-based pricing to increase average order value
- Implement behaviour-triggered communication sequences, for example a follow-up when a customer browses without purchasing
- Develop content that increases product familiarity and encourages repeat use
- Set an automated re-engagement trigger at 60 days of inactivity, before customers transition into the lapsed segment

### Cluster 1: Lapsed Customers (1,067 customers, 24.6%)

| Recency | Frequency | Average Spend |
|---------|-----------|---------------|
| 248 days | 2 orders | $481 |

These customers have not purchased in over eight months. When they were active, purchase frequency was low and average spend was below the base mean. Reactivation is possible for a subset of this group, particularly those with more recent last-purchase dates, but a significant proportion are unlikely to return.

**Recommended actions:**
- Run a time-limited win-back campaign for customers with recency between 150 and 200 days, as they are more likely to respond than those inactive for over 300 days
- Send a short survey to understand reasons for lapsing; this data has value beyond the individual response
- Suppress the non-responding portion from standard campaign sends to protect email deliverability
- Accept that a portion of this segment represents permanent churn and focus reactivation spend accordingly

---

## Application to Marketing Operations

The practical output of this analysis is a set of four customer lists, exportable as CSV files, each tagged with a segment label. These files can be imported directly into HubSpot, Salesforce, Klaviyo, or any CRM platform that supports audience segmentation by custom fields.

The value of this approach over standard demographic or channel-based segmentation is that it reflects actual purchase behaviour. Two customers with identical demographic profiles may belong to entirely different segments based on how they interact with the business over time, and therefore warrant different communication strategies.

---

## Technical Architecture

```
Online Retail.xlsx
        │
        ▼
  Data Cleaning
  (remove nulls, returns, zero prices)
        │
        ▼
  RFM Feature Engineering
  (per-customer aggregation)
        │
        ▼
  StandardScaler
  (normalise feature ranges)
        │
        ▼
  Optimal K Selection
  (elbow method + silhouette analysis, minimum 4 clusters)
        │
        ├──── K-Means
        ├──── DBSCAN          Evaluated on:
        ├──── Hierarchical    · Silhouette Score
        └──── GMM             · Calinski-Harabasz Index
                              · Davies-Bouldin Index
                                        │
                                        ▼
                              Selected model: K-Means
                              Silhouette score: 0.6162
                                        │
                                        ▼
                              Segment labels, figures, CSV exports
```

**Stack:** Python, pandas, scikit-learn, matplotlib, seaborn, openpyxl

---

## Repository Structure

```
├── clustering-report.py        # End-to-end analysis pipeline
├── customer-cluster.ipynb      # Exploratory notebook
├── Online Retail.xlsx          # Source data
└── results/
    ├── clustering_report.txt
    ├── figures/
    │   ├── optimal_clusters.png
    │   ├── clustering_comparison_pca.png
    │   ├── metrics_comparison.png
    │   ├── k-means_detailed_analysis.png
    │   ├── k-means_3d_visualization.png
    │   └── k-means_cluster_sizes.png
    ├── k_means_clustered_data.csv
    ├── k_means_cluster_summary.csv
    └── [equivalent outputs for DBSCAN, Hierarchical, GMM]
```

---

## Running the Analysis

```bash
git clone https://github.com/alexmerlet1/customer-clustering-segmentation.git
cd customer-clustering-segmentation
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl

python clustering-report.py

# To specify a different cluster count
python clustering-report.py --clusters 5
python clustering-report.py --min-clusters 6
```

All outputs are written to `./results/`.

---

## Possible Extensions

Several additions would strengthen this analysis for production use:

1. **Product category features.** RFM describes purchasing behaviour in aggregate but does not capture what customers buy. Adding category-level purchase data would allow more granular persona construction.
2. **Longitudinal tracking.** Running the segmentation monthly and tracking customer movement between segments over time would allow earlier identification of customers at risk of lapsing.
3. **Predictive modelling.** Cluster membership can be used as a feature in downstream models for churn prediction or customer lifetime value estimation.
4. **CRM integration.** The CSV outputs are structured for direct import into standard CRM platforms. Automating this step would allow segment lists to be refreshed on a regular schedule.

---

## About

My background is in hospitality management and CRM. I have worked across operations and guest-facing marketing roles at Grand Hyatt Hong Kong, Sofitel Arc de Triomphe in Paris, and Asian Trails in Bangkok. This project reflects my interest in applying data analysis methods to marketing problems, particularly in understanding customer behaviour at a level of detail that is not visible through standard campaign reporting.

I am currently looking for roles in marketing strategy, CRM, e-commerce, or growth, ideally in environments where analytical and creative work overlap.

📧 alexmerlet1@gmail.com · [linkedin.com/in/alexmerlet](https://www.linkedin.com/in/alexmerlet)
