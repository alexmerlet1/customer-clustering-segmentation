# Advanced Clustering - Usage Guide

## Why More Clusters?

The script now defaults to using **at least 4 clusters** for better business segmentation. This provides:
- More granular customer segments
- Better actionable insights
- More targeted marketing strategies
- Clearer differentiation between customer types

## Usage Options

### 1. Automatic Selection (Recommended)
The script will automatically select the optimal number of clusters (minimum 4) based on statistical metrics while considering business needs:

```bash
python clustering-report.py
```

### 2. Force Specific Number of Clusters
If you want a specific number of clusters (e.g., 5 or 6):

```bash
python clustering-report.py --clusters 5
```

### 3. Change Minimum Cluster Requirement
To change the minimum number of clusters (default is 4):

```bash
python clustering-report.py --min-clusters 6
```

## Cluster Segmentation Types

With more clusters, you'll get better segmentation:

- **🌟 VIP Customers** - Recent, highly frequent, extremely valuable
- **🟢 Champions** - Recent buyers with high engagement
- **🟢 Loyal High-Value** - Active customers with above-average spending
- **🟢 Loyal Mid-Tier** - Regular shoppers with reasonable spending
- **🟡 Potential Loyalists** - Regular buyers, opportunity to upsell
- **🟡 Regular Customers** - Average engagement patterns
- **💎 Big Spenders** - High value regardless of frequency
- **🆕 New Customers** - Recent first-time buyers
- **😴 Hibernating** - Previously valuable, need reactivation
- **🔴 At-Risk** - Low engagement, need win-back campaigns
- **🔴 Lost Customers** - Haven't purchased in a long time

## Example Output

With 4-6 clusters, you'll see more detailed segmentation like:

```
🔹 Cluster 0 (VIP Customers)
   📍 Size: 26 customers (0.6% of total)
   📅 Recency: ~6 days (very recent buyers, highly active customers)
   🔄 Frequency: ~66 purchases (extremely frequent buyers)
   💰 Monetary: $85,904 total spend (highest)
   🌟 Interpretation: VIP Customers – recent, highly frequent, and extremely valuable.

🔹 Cluster 1 (Loyal Mid-Tier Customers)
   📍 Size: 1,234 customers (28.4% of total)
   📅 Recency: ~44 days (recent buyers, active customers)
   🔄 Frequency: ~5 purchases (moderate repeaters)
   💰 Monetary: $1,535 total spend (medium spenders)
   🟢 Interpretation: Loyal Mid-Tier Customers – shop somewhat often and spend a reasonable amount.
```

## Tips

1. **Start with automatic selection** - The script balances statistical optimality with business needs
2. **Try different cluster counts** - Run with 4, 5, and 6 clusters to see which provides the best insights
3. **Review cluster sizes** - Ensure no cluster is too small (<1% of customers) or too large (>80%)
4. **Check business interpretation** - The report automatically classifies each cluster with actionable insights

## Technical Details

- **Silhouette Score**: Measures cluster quality (higher is better, range: -1 to 1)
- **Minimum Clusters**: Default 4 for meaningful business segmentation
- **Selection Logic**: Chooses clusters that meet minimum requirement while maintaining good silhouette score (within 0.1 of best)
