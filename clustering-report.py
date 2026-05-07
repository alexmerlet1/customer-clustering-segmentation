"""
Advanced Customer Clustering Analysis Report
============================================
This script performs advanced clustering analysis on RFM customer data,
comparing multiple clustering algorithms and generating comprehensive reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Workaround for a Windows + BLAS edge case where `threadpoolctl.get_config()`
# can return None, causing scikit-learn to crash when limiting BLAS threads.
try:
    import threadpoolctl  # type: ignore

    _orig_get_config = getattr(threadpoolctl, "get_config", None)
    if callable(_orig_get_config):
        def _safe_get_config():  # type: ignore
            cfg = _orig_get_config()
            return cfg or ""

        threadpoolctl.get_config = _safe_get_config  # type: ignore
except Exception:
    # If threadpoolctl isn't present or changes, continue without patching.
    pass

# Plot styling for consistent report figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Output folders for reports and figures
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/figures', exist_ok=True)

def load_and_prepare_data():
    """Load the retail dataset and build an RFM table per customer."""
    print("Loading and preparing data...")
    
    DATASET_PATH = "./Online Retail.xlsx"
    
    # Read source workbook
    df = pd.read_excel(DATASET_PATH)
    
    # Basic cleanup: keep valid customers and positive transactions
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # Build RFM: Recency (days), Frequency (unique invoices), Monetary (total spend)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })
    
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    print(f"Data loaded: {len(rfm)} customers")
    print(f"RFM Statistics:\n{rfm.describe()}\n")
    
    return rfm

def determine_optimal_clusters(rfm_scaled, max_clusters=10, min_clusters=4, force_clusters=None):
    """
    Pick a reasonable number of clusters using elbow + silhouette, with a small
    business-friendly bias toward having enough segments to act on.
    
    Parameters:
    -----------
    rfm_scaled : array-like
        Scaled RFM data
    max_clusters : int
        Maximum number of clusters to test
    min_clusters : int
        Minimum number of clusters for business segmentation (default: 4)
    force_clusters : int, optional
        Force a specific number of clusters (overrides automatic selection)
    """
    print("Determining optimal number of clusters...")
    
    if force_clusters is not None:
        print(f"Using forced number of clusters: {force_clusters}")
        return force_clusters, None, None
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, labels))
    
    # Elbow heuristic: look for the largest change in inertia improvement
    rate_of_change = np.diff(inertias) / np.diff(list(k_range))
    optimal_k_elbow = k_range[np.argmax(rate_of_change) + 1]
    
    # Find optimal k (silhouette)
    optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
    
    # Business-oriented selection: favor a few more clusters if quality stays similar
    valid_k_scores = [(k, score) for k, score in zip(k_range, silhouette_scores) 
                      if k >= min_clusters]
    
    if valid_k_scores:
        # Consider the best few candidates by silhouette score
        valid_k_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = valid_k_scores[:3]
        
        # Prefer more clusters if the silhouette score is still close to the best
        best_score = top_candidates[0][1]
        good_candidates = [k for k, score in top_candidates if score >= best_score - 0.1]
        
        if good_candidates:
            # Choose the one with most clusters among good candidates
            optimal_k_business = max(good_candidates)
        else:
            optimal_k_business = top_candidates[0][0]
    else:
        # If no valid k meets minimum, use minimum anyway
        optimal_k_business = min_clusters
    
    # Plot elbow and silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(list(k_range), inertias, 'bo-')
    ax1.axvline(optimal_k_elbow, color='r', linestyle='--', label=f'Optimal (Elbow): {optimal_k_elbow}')
    if optimal_k_business != optimal_k_elbow:
        ax1.axvline(optimal_k_business, color='g', linestyle='--', 
                   label=f'Selected (Business): {optimal_k_business}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(list(k_range), silhouette_scores, 'go-')
    ax2.axvline(optimal_k_silhouette, color='r', linestyle='--', 
               label=f'Optimal (Silhouette): {optimal_k_silhouette}')
    if optimal_k_business != optimal_k_silhouette:
        ax2.axvline(optimal_k_business, color='g', linestyle='--', 
                   label=f'Selected (Business): {optimal_k_business}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Method')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/figures/optimal_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final choice: business-oriented k (keeps segmentation actionable)
    optimal_k = optimal_k_business
    print(f"Statistical optimal (silhouette): {optimal_k_silhouette}")
    print(f"Business-oriented selection: {optimal_k} clusters")
    print(f"  (Minimum required: {min_clusters}, Silhouette: {silhouette_scores[optimal_k-2]:.4f})\n")
    
    return optimal_k, inertias, silhouette_scores

def perform_kmeans_clustering(rfm_scaled, n_clusters=4):
    """Perform K-Means clustering."""
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    
    silhouette = silhouette_score(rfm_scaled, labels)
    calinski = calinski_harabasz_score(rfm_scaled, labels)
    davies = davies_bouldin_score(rfm_scaled, labels)
    
    print(f"K-Means - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies:.3f}")
    
    return labels, {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'n_clusters': len(np.unique(labels))
    }

def perform_dbscan_clustering(rfm_scaled, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering."""
    print(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(rfm_scaled)
    
    # Remove noise points for metrics calculation
    mask = labels != -1
    if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
        silhouette = silhouette_score(rfm_scaled[mask], labels[mask])
        calinski = calinski_harabasz_score(rfm_scaled[mask], labels[mask])
        davies = davies_bouldin_score(rfm_scaled[mask], labels[mask])
    else:
        silhouette = -1
        calinski = 0
        davies = float('inf')
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN - Clusters: {n_clusters}, Noise points: {n_noise}")
    print(f"DBSCAN - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies:.3f}")
    
    return labels, {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }

def perform_hierarchical_clustering(rfm_scaled, n_clusters=4):
    """Perform Hierarchical clustering."""
    print(f"Performing Hierarchical clustering with {n_clusters} clusters...")
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(rfm_scaled)
    
    silhouette = silhouette_score(rfm_scaled, labels)
    calinski = calinski_harabasz_score(rfm_scaled, labels)
    davies = davies_bouldin_score(rfm_scaled, labels)
    
    print(f"Hierarchical - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies:.3f}")
    
    return labels, {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'n_clusters': len(np.unique(labels))
    }

def perform_gmm_clustering(rfm_scaled, n_components=4):
    """Perform Gaussian Mixture Model clustering."""
    print(f"Performing GMM clustering with {n_components} components...")
    
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    labels = gmm.fit_predict(rfm_scaled)
    
    silhouette = silhouette_score(rfm_scaled, labels)
    calinski = calinski_harabasz_score(rfm_scaled, labels)
    davies = davies_bouldin_score(rfm_scaled, labels)
    
    print(f"GMM - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies:.3f}")
    
    return labels, {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'n_clusters': len(np.unique(labels))
    }

def create_comparison_visualizations(rfm, rfm_scaled, results):
    """Create comprehensive comparison visualizations."""
    print("\nCreating visualizations...")
    
    # 2D projection for quick visual comparison across methods
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    
    # Side-by-side clustering view (PCA space)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    methods = ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM']
    colors = ['tab10', 'tab20', 'tab10', 'tab10']
    
    for idx, method in enumerate(methods):
        labels = results[method]['labels']
        ax = axes[idx]
        
        # DBSCAN can produce noise points (-1). Plot them distinctly.
        if method == 'DBSCAN':
            scatter = ax.scatter(rfm_pca[:, 0], rfm_pca[:, 1], 
                               c=labels, cmap=colors[idx], alpha=0.6, s=50)
            # Noise points
            noise_mask = labels == -1
            if noise_mask.any():
                ax.scatter(rfm_pca[noise_mask, 0], rfm_pca[noise_mask, 1],
                          c='black', marker='x', s=100, label='Noise', alpha=0.5)
        else:
            scatter = ax.scatter(rfm_pca[:, 0], rfm_pca[:, 1], 
                               c=labels, cmap=colors[idx], alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(f'{method} Clustering\n'
                    f'Silhouette: {results[method]["metrics"]["silhouette"]:.3f}')
        ax.legend() if method == 'DBSCAN' else None
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig('./results/figures/clustering_comparison_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics comparison across methods
    metrics_df = pd.DataFrame({
        method: results[method]['metrics'] 
        for method in methods
    }).T
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    metric_names = ['Silhouette Score\n(Higher is Better)', 
                   'Calinski-Harabasz Score\n(Higher is Better)',
                   'Davies-Bouldin Score\n(Lower is Better)']
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        bars = ax.bar(metrics_df.index, metrics_df[metric], 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylabel('Score')
        ax.set_title(name)
        ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
        
        # Label bars with values for readability
        for bar in bars:
            height = bar.get_height()
            if not (np.isinf(height) or np.isnan(height)):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('./results/figures/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed view for the best method (by silhouette)
    best_method = max(methods, key=lambda m: results[m]['metrics']['silhouette'])
    best_labels = results[best_method]['labels']
    
    # DBSCAN noise is excluded from the per-cluster plots
    if best_method == 'DBSCAN':
        mask = best_labels != -1
        rfm_vis = rfm[mask].copy()
        labels_vis = best_labels[mask]
    else:
        rfm_vis = rfm.copy()
        labels_vis = best_labels
    
    rfm_vis['Cluster'] = labels_vis
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RFM heatmap
    cluster_stats = rfm_vis.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    sns.heatmap(cluster_stats, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0, 0])
    axes[0, 0].set_title(f'{best_method} - Cluster Profiles (Average RFM Values)')
    
    # Recency distribution
    for cluster in sorted(rfm_vis['Cluster'].unique()):
        axes[0, 1].hist(rfm_vis[rfm_vis['Cluster'] == cluster]['Recency'], 
                        alpha=0.6, label=f'Cluster {cluster}', bins=30)
    axes[0, 1].set_xlabel('Recency')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Recency Distribution by Cluster')
    axes[0, 1].legend()
    
    # Frequency distribution
    for cluster in sorted(rfm_vis['Cluster'].unique()):
        axes[1, 0].hist(rfm_vis[rfm_vis['Cluster'] == cluster]['Frequency'], 
                        alpha=0.6, label=f'Cluster {cluster}', bins=30)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Frequency Distribution by Cluster')
    axes[1, 0].legend()
    
    # Monetary distribution (log scale for better visualization)
    for cluster in sorted(rfm_vis['Cluster'].unique()):
        monetary = rfm_vis[rfm_vis['Cluster'] == cluster]['Monetary']
        monetary = monetary[monetary > 0]  # Remove zeros for log
        axes[1, 1].hist(np.log10(monetary + 1), alpha=0.6, 
                       label=f'Cluster {cluster}', bins=30)
    axes[1, 1].set_xlabel('Log10(Monetary)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Monetary Distribution by Cluster (Log Scale)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'./results/figures/{best_method.lower()}_detailed_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3D scatter in original RFM space
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(rfm_vis['Recency'], rfm_vis['Frequency'], rfm_vis['Monetary'],
                        c=labels_vis, cmap='tab10', alpha=0.6, s=50)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title(f'{best_method} - 3D RFM Visualization')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.savefig(f'./results/figures/{best_method.lower()}_3d_visualization.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cluster size breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_sizes = pd.Series(labels_vis).value_counts().sort_index()
    bars = ax.bar(cluster_sizes.index.astype(str), cluster_sizes.values, 
                 color=plt.cm.tab10(np.linspace(0, 1, len(cluster_sizes))))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Customers')
    ax.set_title(f'{best_method} - Cluster Sizes')
    
    # Add counts and percentages
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({height/len(labels_vis)*100:.1f}%)',
               ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'./results/figures/{best_method.lower()}_cluster_sizes.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to ./results/figures/")

def get_segment_emoji_and_description(recency, frequency, monetary, cluster_profiles, cluster_id):
    """
    Assign a segment label and description based on relative ranking.
    The goal is to give each cluster a distinct, business-usable interpretation.
    """
    # Get all cluster values for relative comparison
    all_recencies = cluster_profiles['Recency'].values
    all_frequencies = cluster_profiles['Frequency'].values
    all_monetaries = cluster_profiles['Monetary'].values
    
    # Get current cluster's position in sorted arrays
    sorted_clusters = sorted(cluster_profiles.index)
    cluster_pos = sorted_clusters.index(cluster_id)
    
    # Calculate relative rankings (0 = best, 1 = worst)
    # For recency: lower is better (0 = most recent)
    recency_sorted = np.argsort(all_recencies)
    recency_rank = np.where(recency_sorted == cluster_pos)[0][0] / max(len(all_recencies) - 1, 1)
    
    # For frequency: higher is better (0 = highest frequency)
    frequency_sorted = np.argsort(all_frequencies)[::-1]
    frequency_rank = np.where(frequency_sorted == cluster_pos)[0][0] / max(len(all_frequencies) - 1, 1)
    
    # For monetary: higher is better (0 = highest monetary)
    monetary_sorted = np.argsort(all_monetaries)[::-1]
    monetary_rank = np.where(monetary_sorted == cluster_pos)[0][0] / max(len(all_monetaries) - 1, 1)
    
    # Calculate medians for thresholds
    recency_median = cluster_profiles['Recency'].median()
    frequency_median = cluster_profiles['Frequency'].median()
    monetary_median = cluster_profiles['Monetary'].median()
    
    # Priority: Monetary > Frequency > Recency for VIP classification
    
    # VIP/Champions: Top tier in all metrics
    if monetary_rank < 0.25 and frequency_rank < 0.25 and recency_rank < 0.25:
        if monetary > monetary_median * 2 and frequency > frequency_median * 2:
            return "VIP", "VIP Customers", "top-tier customers - recent, highly frequent, and extremely valuable"
        else:
            return "Champion", "Champions", "recent buyers with high engagement and spending"
    
    # High-Value segments (monetary is key differentiator)
    elif monetary_rank < 0.5:
        if frequency_rank < 0.5 and recency_rank < 0.5:
            return "Loyal", "Loyal High-Value Customers", "active customers with above-average spending and frequency"
        elif recency_rank < 0.5:
            return "High-Value", "High-Value Recent Buyers", "recent customers with high spending but moderate frequency"
        else:
            return "High-Value", "High-Value Less Frequent", "valuable customers who don't shop as often"
    
    # Mid-tier segments (moderate monetary)
    elif monetary_rank < 0.75:
        if frequency_rank < 0.5 and recency_rank < 0.5:
            return "Loyal", "Loyal Mid-Tier Customers", "shop somewhat often and spend a reasonable amount"
        elif recency_rank < 0.5:
            return "Active", "Regular Active Customers", "recent buyers with moderate spending - opportunity to increase value"
        else:
            return "Mid", "Moderate Spenders", "average spending customers who need re-engagement"
    
    # Low-value segments
    else:
        if recency_rank > 0.75:
            if frequency_rank > 0.75:
                return "Lost", "Lost Customers", "haven't purchased in a very long time with low engagement"
            else:
                return "At-Risk", "At-Risk Customers", "old customers with low engagement and low spend"
        elif frequency_rank > 0.75:
            return "One-Time", "One-Time Buyers", "customers who made few purchases and haven't returned"
        else:
            return "Low", "Low-Value Customers", "customers with below-average engagement and spending"

def get_business_recommendations(cluster_profiles, cluster_sizes):
    """Generate business recommendations for each cluster."""
    recommendations = []
    
    # Sort clusters by monetary value (descending) for priority
    sorted_clusters = sorted(cluster_profiles.index, 
                            key=lambda x: cluster_profiles.loc[x, 'Monetary'], 
                            reverse=True)
    
    for cluster in sorted_clusters:
        recency = cluster_profiles.loc[cluster, 'Recency']
        frequency = cluster_profiles.loc[cluster, 'Frequency']
        monetary = cluster_profiles.loc[cluster, 'Monetary']
        size = cluster_sizes[cluster]
        percentage = (size / cluster_sizes.sum()) * 100
        
        recency_median = cluster_profiles['Recency'].median()
        frequency_median = cluster_profiles['Frequency'].median()
        monetary_median = cluster_profiles['Monetary'].median()
        
        emoji, segment, description = get_segment_emoji_and_description(
            recency, frequency, monetary, cluster_profiles, cluster
        )
        
        # Generate recommendations based on segment
        if "VIP" in segment:
            rec = f"Keep them happy → loyalty rewards, exclusive offers, early access to new products, VIP treatment"
        elif "Champions" in segment:
            rec = f"Maintain engagement → loyalty programs, exclusive offers, early access to new products"
        elif "Loyal High-Value" in segment:
            rec = f"Encourage upsell/cross-sell to move them toward VIP status, reward loyalty"
        elif "Loyal Mid-Tier" in segment:
            rec = f"Encourage upsell/cross-sell to move them toward VIP status, increase purchase frequency"
        elif "High-Value Recent" in segment:
            rec = f"Maintain relationship, increase purchase frequency through targeted campaigns"
        elif "High-Value Less Frequent" in segment:
            rec = f"Re-engagement campaigns to increase frequency, maintain high-value relationship"
        elif "Regular Active" in segment:
            rec = f"Upsell and cross-sell opportunities, bundle offers to increase average order value"
        elif "Moderate Spenders" in segment:
            rec = f"Re-engagement campaigns, offer bundles/discounts to increase spending"
        elif "One-Time Buyers" in segment:
            rec = f"Welcome back campaigns, special offers to encourage repeat purchases"
        elif "Low-Value" in segment:
            rec = f"Offer bundles/discounts to increase spending, cross-sell complementary products"
        elif "At-Risk" in segment:
            rec = f"Send win-back campaigns (emails, reactivation offers, special discounts), survey for feedback"
        elif "Lost" in segment:
            rec = f"Aggressive win-back campaigns, special reactivation offers, understand why they left"
        else:
            rec = f"Develop targeted campaigns based on their specific RFM profile"
        
        recommendations.append((cluster, emoji, segment, rec, size, percentage))
    
    return recommendations

def generate_report(rfm, results, optimal_k):
    """Generate a comprehensive text report with business-friendly formatting."""
    print("\nGenerating report...")
    
    report = []
    report.append("=" * 80)
    report.append("ADVANCED CUSTOMER CLUSTERING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("Dataset: Online Retail")
    report.append(f"Total Customers: {len(rfm):,}")
    
    # Best method
    methods = ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM']
    best_method = max(methods, key=lambda m: results[m]['metrics']['silhouette'])
    n_clusters = results[best_method]['metrics']['n_clusters']
    report.append(f"Best Method: {best_method} (Silhouette Score: {results[best_method]['metrics']['silhouette']:.4f})")
    report.append(f"Number of Clusters: {n_clusters}")
    if n_clusters >= 4:
        report.append(f"Note: Using {n_clusters} clusters for more actionable segmentation")
    
    # Get clustered data
    best_labels = results[best_method]['labels']
    
    if best_method == 'DBSCAN':
        mask = best_labels != -1
        rfm_clustered = rfm[mask].copy()
        labels_clustered = best_labels[mask]
    else:
        rfm_clustered = rfm.copy()
        labels_clustered = best_labels
    
    rfm_clustered['Cluster'] = labels_clustered
    
    cluster_profiles = rfm_clustered.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    cluster_sizes = pd.Series(labels_clustered).value_counts().sort_index()
    
    # Cluster Profiles Section
    report.append("\n" + "=" * 80)
    report.append("CLUSTER PROFILES")
    report.append("=" * 80)
    
    # Sort clusters by recency (most recent first) for better presentation
    sorted_clusters = sorted(cluster_profiles.index, 
                            key=lambda x: cluster_profiles.loc[x, 'Recency'])
    
    for cluster in sorted_clusters:
        recency = cluster_profiles.loc[cluster, 'Recency']
        frequency = cluster_profiles.loc[cluster, 'Frequency']
        monetary = cluster_profiles.loc[cluster, 'Monetary']
        size = cluster_sizes[cluster]
        percentage = (size / cluster_sizes.sum()) * 100
        
        recency_median = cluster_profiles['Recency'].median()
        frequency_median = cluster_profiles['Frequency'].median()
        monetary_median = cluster_profiles['Monetary'].median()
        
        emoji, segment, description = get_segment_emoji_and_description(
            recency, frequency, monetary, cluster_profiles, cluster
        )
        
        # Recency interpretation
        if recency < 30:
            recency_desc = "very recent buyers, highly active customers"
        elif recency < 60:
            recency_desc = "recent buyers, active customers"
        elif recency < 90:
            recency_desc = "moderately recent buyers"
        else:
            recency_desc = "haven't purchased in a long time"
        
        # Frequency interpretation
        if frequency > frequency_median * 2:
            freq_desc = "extremely frequent buyers"
        elif frequency > frequency_median:
            freq_desc = "frequent repeaters"
        elif frequency > frequency_median * 0.5:
            freq_desc = "moderate repeaters"
        else:
            freq_desc = "infrequent or one-time buyers"
        
        # Monetary interpretation
        if monetary > monetary_median * 3:
            mon_desc = f"${monetary:,.0f} total spend (highest)"
        elif monetary > monetary_median * 2:
            mon_desc = f"${monetary:,.0f} total spend (very high)"
        elif monetary > monetary_median:
            mon_desc = f"${monetary:,.0f} total spend (high)"
        elif monetary > monetary_median * 0.5:
            mon_desc = f"${monetary:,.0f} total spend (medium spenders)"
        else:
            mon_desc = f"${monetary:,.0f} total spend (low)"
        
        report.append(f"\nCluster {cluster}")
        report.append(f"   Size: {size:,} customers ({percentage:.1f}% of total)")
        report.append(f"   Recency: ~{recency:.0f} days ({recency_desc})")
        report.append(f"   Frequency: ~{frequency:.1f} purchases ({freq_desc})")
        report.append(f"   Monetary: {mon_desc}")
        report.append(f"   Segment: {segment} - {description}.")
    
    # Business Insights Section
    report.append("\n" + "=" * 80)
    report.append("FINAL BUSINESS INSIGHTS & RECOMMENDATIONS")
    report.append("=" * 80)
    
    recommendations = get_business_recommendations(cluster_profiles, cluster_sizes)
    
    for cluster, emoji, segment, rec, size, percentage in recommendations:
        report.append(f"\nCluster {cluster} ({segment}): {rec}")
    
    # Additional strategic recommendations
    report.append("\n" + "-" * 80)
    report.append("STRATEGIC RECOMMENDATIONS")
    report.append("-" * 80)
    
    # Find VIP/Champions clusters (top 25% in all metrics)
    vip_clusters = []
    for cluster in cluster_profiles.index:
        recency = cluster_profiles.loc[cluster, 'Recency']
        frequency = cluster_profiles.loc[cluster, 'Frequency']
        monetary = cluster_profiles.loc[cluster, 'Monetary']
        
        all_recencies = cluster_profiles['Recency'].values
        all_frequencies = cluster_profiles['Frequency'].values
        all_monetaries = cluster_profiles['Monetary'].values
        
        recency_rank = np.argsort(all_recencies).argsort()[cluster] / (len(all_recencies) - 1)
        frequency_rank = np.argsort(all_frequencies)[::-1].argsort()[cluster] / (len(all_frequencies) - 1)
        monetary_rank = np.argsort(all_monetaries)[::-1].argsort()[cluster] / (len(all_monetaries) - 1)
        
        if monetary_rank < 0.25 and frequency_rank < 0.25 and recency_rank < 0.25:
            vip_clusters.append(cluster)
    
    if vip_clusters:
        report.append(f"\nVIP Focus: Protect and grow your VIP customers (Clusters {', '.join(map(str, vip_clusters))})")
        report.append("   → Implement a VIP program with exclusive benefits")
        report.append("   → Assign dedicated account managers for top customers")
        report.append("   → Provide early access to new products and special events")
    
    # Find at-risk clusters (bottom 50% in recency and frequency)
    at_risk_clusters = []
    for cluster in cluster_profiles.index:
        all_recencies = cluster_profiles['Recency'].values
        all_frequencies = cluster_profiles['Frequency'].values
        
        recency_rank = np.argsort(all_recencies).argsort()[cluster] / (len(all_recencies) - 1)
        frequency_rank = np.argsort(all_frequencies)[::-1].argsort()[cluster] / (len(all_frequencies) - 1)
        
        if recency_rank > 0.5 and frequency_rank > 0.5:
            at_risk_clusters.append(cluster)
    
    if at_risk_clusters:
        report.append(f"\n  Win-Back Campaign: Target at-risk customers (Clusters {', '.join(map(str, at_risk_clusters))})")
        report.append("   → Send personalized reactivation emails with special offers")
        report.append("   → Create 'We miss you' campaigns with discounts")
        report.append("   → Survey to understand why they stopped purchasing")
    
    report.append(f"\n Monitoring: Track cluster evolution monthly to identify trends")
    report.append("Next Steps: Consider adding product categories and seasonality for deeper insights")
    
    # Technical details (optional, at the end)
    report.append("\n" + "=" * 80)
    report.append(" TECHNICAL DETAILS")
    report.append("=" * 80)
    report.append(f"\nClustering Method: {best_method}")
    report.append(f"Silhouette Score: {results[best_method]['metrics']['silhouette']:.4f}")
    report.append(f"Calinski-Harabasz Score: {results[best_method]['metrics']['calinski_harabasz']:.2f}")
    report.append(f"Davies-Bouldin Score: {results[best_method]['metrics']['davies_bouldin']:.4f}")
    
    if best_method == 'DBSCAN':
        report.append(f"Noise Points: {results[best_method]['metrics']['n_noise']}")
    
    # Save report
    report_text = "\n".join(report)
    with open('./results/clustering_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Report saved to ./results/clustering_report.txt")
    
    return report_text

def save_clustered_data(rfm, results):
    """Save clustered data to CSV files."""
    print("\nSaving clustered data...")
    
    methods = ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM']
    
    for method in methods:
        labels = results[method]['labels']
        rfm_copy = rfm.copy()
        rfm_copy['Cluster'] = labels
        
        # Save all data
        filename = f'./results/{method.lower().replace("-", "_")}_clustered_data.csv'
        rfm_copy.to_csv(filename)
        
        # Save cluster summaries
        if method == 'DBSCAN':
            mask = labels != -1
            rfm_clustered = rfm_copy[mask]
        else:
            rfm_clustered = rfm_copy
        
        cluster_summary = rfm_clustered.groupby('Cluster').agg({
            'Recency': ['mean', 'std', 'count'],
            'Frequency': ['mean', 'std'],
            'Monetary': ['mean', 'std', 'sum']
        }).round(2)
        
        summary_filename = f'./results/{method.lower().replace("-", "_")}_cluster_summary.csv'
        cluster_summary.to_csv(summary_filename)
    
    print("Clustered data saved to ./results/")

def main(n_clusters=None, min_clusters=4):
    """
    Main execution function.
    
    Parameters:
    -----------
    n_clusters : int, optional
        Force a specific number of clusters (overrides automatic selection)
    min_clusters : int
        Minimum number of clusters for business segmentation (default: 4)
    """
    print("=" * 80)
    print("ADVANCED CUSTOMER CLUSTERING ANALYSIS")
    print("=" * 80)
    print()
    
    # Load and prepare data
    rfm = load_and_prepare_data()
    
    # Scale data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Determine optimal clusters
    optimal_k, inertias, silhouette_scores = determine_optimal_clusters(
        rfm_scaled, 
        max_clusters=10, 
        min_clusters=min_clusters,
        force_clusters=n_clusters
    )
    
    # Perform different clustering methods
    results = {}
    
    # K-Means
    labels_kmeans, metrics_kmeans = perform_kmeans_clustering(rfm_scaled, n_clusters=optimal_k)
    results['K-Means'] = {'labels': labels_kmeans, 'metrics': metrics_kmeans}
    
    # DBSCAN (try different parameters)
    # Estimate eps using k-distance graph
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(rfm_scaled)
    distances, indices = neighbors_fit.kneighbors(rfm_scaled)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 4]
    eps_estimate = np.percentile(distances, 50)  # Use median
    
    # Try DBSCAN with estimated eps, if it fails, try a smaller value
    labels_dbscan, metrics_dbscan = perform_dbscan_clustering(
        rfm_scaled, eps=eps_estimate, min_samples=5
    )
    
    # If too many noise points, try with smaller eps
    if metrics_dbscan.get('n_noise', 0) > len(rfm_scaled) * 0.5:
        eps_estimate = np.percentile(distances, 25)  # Try 25th percentile
        labels_dbscan, metrics_dbscan = perform_dbscan_clustering(
            rfm_scaled, eps=eps_estimate, min_samples=5
        )
    results['DBSCAN'] = {'labels': labels_dbscan, 'metrics': metrics_dbscan}
    
    # Hierarchical
    labels_hierarchical, metrics_hierarchical = perform_hierarchical_clustering(
        rfm_scaled, n_clusters=optimal_k
    )
    results['Hierarchical'] = {'labels': labels_hierarchical, 'metrics': metrics_hierarchical}
    
    # GMM
    labels_gmm, metrics_gmm = perform_gmm_clustering(rfm_scaled, n_components=optimal_k)
    results['GMM'] = {'labels': labels_gmm, 'metrics': metrics_gmm}
    
    # Create visualizations
    create_comparison_visualizations(rfm, rfm_scaled, results)
    
    # Generate report
    report = generate_report(rfm, results, optimal_k)
    
    # Save clustered data
    save_clustered_data(rfm, results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll results saved to ./results/")
    print("  - Report: ./results/clustering_report.txt")
    print("  - Figures: ./results/figures/")
    print("  - Clustered data: ./results/*_clustered_data.csv")
    print("  - Cluster summaries: ./results/*_cluster_summary.csv")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Customer Clustering Analysis')
    parser.add_argument('--clusters', type=int, default=4,
                       help='Number of clusters (default: 4)')
    parser.add_argument('--min-clusters', type=int, default=4,
                       help='Minimum number of clusters for business segmentation (default: 4)')
    
    args = parser.parse_args()
    
    main(n_clusters=args.clusters, min_clusters=args.min_clusters)
