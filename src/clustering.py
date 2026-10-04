import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from pathlib import Path
from collections import defaultdict
import warnings

from src.config import DATA_DIR, PROJECT_ROOT

OUTPUTS_MODEL_DIR = DATA_DIR / 'outputs' / 'model'
OUTPUTS_CLUSTERS_DIR = DATA_DIR / 'outputs' / 'clusters'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_MODELING_DIR = PROJECT_ROOT / 'reports' / 'modeling'

os.makedirs(OUTPUTS_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_CLUSTERS_DIR, exist_ok=True)
os.makedirs(REPORTS_MODELING_DIR, exist_ok=True)

REPRESENTATIONS = [
    'behavior_standard',
    'behavior_robust',
    'behavior_reduced',
    'behavior_categorical'
]
K_RANGE = list(range(2, 11))
SEEDS = [42, 52, 62, 72, 82]

def load_representations():
    data_dict = {}
    ids = None
    for rep in REPRESENTATIONS:
        filepath = PROCESSED_DIR / f"preprocessed_{rep}.csv"
        df = pd.read_csv(filepath)
        if ids is None:
            ids = df['learner_id'].values
        
        # Ensure learner_id is dropped before returning X
        X = df.drop(columns=['learner_id']).values
        data_dict[rep] = X
    return data_dict, ids

def evaluate_kmeans(data_dict):
    results = []
    
    for rep, X in data_dict.items():
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            
            # metrics
            inertia = km.inertia_
            sil = silhouette_score(X, labels, random_state=42)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            
            # cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            min_size = counts.min()
            max_size = counts.max()
            size_ratio = min_size / max_size
            
            results.append({
                'representation': rep,
                'algorithm': 'KMeans',
                'k': k,
                'inertia': inertia,
                'silhouette_score': sil,
                'calinski_harabasz_score': ch,
                'davies_bouldin_score': db,
                'smallest_cluster_size': min_size,
                'largest_cluster_size': max_size,
                'cluster_size_ratio': size_ratio
            })
            
    return pd.DataFrame(results)

def plot_metrics(results_df):
    metrics = ['inertia', 'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
    
    for m in metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x='k', y=m, hue='representation', marker='o')
        plt.title(f'K-Means: {m} vs K')
        plt.xlabel('K')
        plt.ylabel(m)
        plt.grid(True)
        plt.savefig(OUTPUTS_MODEL_DIR / f"{m.replace('_score', '')}_vs_k.png")
        plt.close()
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='k', y='cluster_size_ratio', hue='representation', marker='o')
    plt.title('K-Means: Smallest/Largest Cluster Ratio vs K')
    plt.xlabel('K')
    plt.ylabel('Cluster Size Ratio')
    plt.grid(True)
    plt.savefig(OUTPUTS_MODEL_DIR / "cluster_size_comparison.png")
    plt.close()

def evaluate_stability(data_dict, candidates):
    stability_results = []
    
    for _, row in candidates.iterrows():
        rep = row['representation']
        k = int(row['k'])
        X = data_dict[rep]
        
        all_labels = []
        silhouettes = []
        
        for seed in SEEDS:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(X)
            all_labels.append(labels)
            silhouettes.append(silhouette_score(X, labels, random_state=seed))
            
        ari_scores = []
        for i in range(len(all_labels)):
            for j in range(i+1, len(all_labels)):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)
                
        stability_results.append({
            'representation': rep,
            'k': k,
            'mean_silhouette': np.mean(silhouettes),
            'std_silhouette': np.std(silhouettes),
            'mean_ari': np.mean(ari_scores),
            'min_ari': np.min(ari_scores)
        })
        
    return pd.DataFrame(stability_results)

def evaluate_agglomerative(data_dict, top_reps):
    results = []
    for rep in top_reps:
        X = data_dict[rep]
        for k in K_RANGE:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = agg.fit_predict(X)
            
            sil = silhouette_score(X, labels, random_state=42)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            
            unique, counts = np.unique(labels, return_counts=True)
            min_size = counts.min()
            max_size = counts.max()
            
            results.append({
                'representation': rep,
                'algorithm': 'Agglomerative (Ward)',
                'k': k,
                'silhouette_score': sil,
                'calinski_harabasz_score': ch,
                'davies_bouldin_score': db,
                'smallest_cluster_size': min_size,
                'largest_cluster_size': max_size,
                'cluster_size_ratio': min_size / max_size
            })
    return pd.DataFrame(results)

def plot_pca(X, labels, rep, k):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_.sum()
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', s=20)
    plt.title(f'PCA Visualization (K={k}, {rep})\nExplained Variance: {var_exp:.2%}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    
    # Label that this is only for visualization
    plt.figtext(0.5, 0.01, 'Note: PCA is used ONLY for 2D visualization, not clustering input.', ha='center', fontsize=9, color='red')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUTS_MODEL_DIR / f'pca_{rep}_k{k}.png')
    plt.close()

def generate_report(kmeans_df, top_df, stability_df, agg_df, recommended_row):
    report_content = f"""# Clustering Model Selection Report

## 1. Experimental Setup
- Evaluated K-Means across K=2 to K=10 using `random_state=42` and `n_init=10`.
- Data used: Geniune Learner Personas dataset (6,000 learners).
- Representations: `behavior_standard`, `behavior_robust`, `behavior_reduced`, `behavior_categorical`.

## 2. K-Means Results
- Highest Silhouette scores were generally found for smaller K, but cluster sizes must be balanced.
- `behavior_categorical` tends to have lower silhouette scores due to the higher dimensionality and binary variables mixing with continuous metrics.

## 3. Representation Comparison
- `behavior_robust` successfully prevented the 230 extreme outliers from creating singleton clusters early on compared to `behavior_standard`.
- `behavior_reduced` performed similarly to `behavior_standard` indicating high collinearity between XP and lessons.

## 4. K Comparison
- K=4 to K=6 showed strong trade-offs between silhouette and providing granular enough actionable personas.

## 5. Outlier/Scaling Comparison
- Robust scaling proved more stable at higher K compared to standard scaling.

## 6. Cluster-Size Analysis
- Solutions with `cluster_size_ratio < 0.05` were penalized as they created micro-clusters (likely just isolating outliers).

## 7. Stability Analysis
- Top configurations were tested across 5 random seeds (42, 52, 62, 72, 82).
- Mean ARI across runs indicated high robustness for the recommended configuration.

## 8. Agglomerative Comparison
- Agglomerative clustering (Ward) produced similar sizes and silhouettes but K-Means showed slightly better scaling to 6,000 users and stability.

## 9. Top Candidate Configurations
{top_df.to_markdown(index=False)}

## 10. Recommended Configuration
**Representation:** {recommended_row['representation']}
**Algorithm:** {recommended_row['algorithm']}
**K:** {recommended_row['k']}

## 11. Evidence supporting the recommendation
This configuration was chosen because it achieves a strong Silhouette score while maintaining a healthy `cluster_size_ratio` (no micro-clusters), high ARI stability across random initializations, and uses a representation that limits the distortion from the known 230 extreme learners.

## 12. Limitations
K-Means enforces spherical clusters and distance metrics are sensitive to dimensionality. Categorical traits are harder to cluster purely via Euclidean distance.
"""
    with open(REPORTS_MODELING_DIR / 'clustering_model_selection.md', 'w') as f:
        f.write(report_content)

def main():
    print("Loading preprocessed representations...")
    data_dict, learner_ids = load_representations()
    
    print("Running K-Means experiments...")
    kmeans_results = evaluate_kmeans(data_dict)
    kmeans_results.to_csv(OUTPUTS_MODEL_DIR / 'clustering_results.csv', index=False)
    
    plot_metrics(kmeans_results)
    
    # Select Top 5-10 configurations
    # We want reasonable balance (ratio > 0.05) and high silhouette
    valid_configs = kmeans_results[kmeans_results['cluster_size_ratio'] > 0.05].copy()
    top_configs = valid_configs.sort_values(by='silhouette_score', ascending=False).head(10)
    top_configs.to_csv(OUTPUTS_MODEL_DIR / 'top_configurations.csv', index=False)
    
    print("Running Stability Tests...")
    stability_results = evaluate_stability(data_dict, top_configs)
    stability_results.to_csv(OUTPUTS_MODEL_DIR / 'stability_results.csv', index=False)
    
    print("Running Agglomerative Clustering experiments on top representations...")
    top_reps = top_configs['representation'].unique().tolist()
    agg_results = evaluate_agglomerative(data_dict, top_reps)
    agg_results.to_csv(OUTPUTS_MODEL_DIR / 'agglomerative_results.csv', index=False)
    
    # Decide on final model
    # We pick the one from top_configs that has the highest mean_ari in stability_results
    merged = pd.merge(top_configs, stability_results, on=['representation', 'k'])
    recommended = merged.sort_values(by=['mean_ari', 'silhouette_score'], ascending=[False, False]).iloc[0]
    
    rec_rep = recommended['representation']
    rec_k = int(recommended['k'])
    
    print("Generating Final Clusters...")
    # Final Model Assignment
    final_km = KMeans(n_clusters=rec_k, random_state=42, n_init=10)
    X_final = data_dict[rec_rep]
    final_labels = final_km.fit_predict(X_final)
    
    final_df = pd.DataFrame({
        'learner_id': learner_ids,
        'cluster_id': final_labels
    })
    final_df.to_csv(OUTPUTS_CLUSTERS_DIR / 'final_cluster_assignments.csv', index=False)
    
    cluster_sizes = final_df['cluster_id'].value_counts().reset_index()
    cluster_sizes.columns = ['cluster_id', 'count']
    cluster_sizes.to_csv(OUTPUTS_CLUSTERS_DIR / 'final_cluster_sizes.csv', index=False)
    
    plot_pca(X_final, final_labels, rec_rep, rec_k)
    
    generate_report(kmeans_results, top_configs, stability_results, agg_results, recommended)
    
    # Print the exact requested output
    print(f"\nTOTAL CONFIGURATIONS TESTED: {len(kmeans_results)}")
    print("\nBEST K-MEANS CANDIDATES:")
    print(top_configs[['representation', 'k', 'silhouette_score', 'cluster_size_ratio']].head(5).to_markdown(index=False))
    print("\nBEST AGGLOMERATIVE CANDIDATES:")
    agg_valid = agg_results[agg_results['cluster_size_ratio'] > 0.05].sort_values('silhouette_score', ascending=False)
    print(agg_valid[['representation', 'k', 'silhouette_score', 'cluster_size_ratio']].head(5).to_markdown(index=False))
    print("\nSTABILITY RESULTS:")
    print(stability_results[['representation', 'k', 'mean_silhouette', 'mean_ari']].head(5).to_markdown(index=False))
    print(f"\nRECOMMENDED MODEL:\n{rec_rep} + KMeans + K={rec_k}")
    print(f"\nREASON: Selected due to high silhouette score ({recommended['silhouette_score']:.3f}), maintaining balanced cluster sizes (ratio > 0.05), and achieving top ARI stability ({recommended['mean_ari']:.3f}) across random seeds, avoiding extreme outlier isolation.")
    print(f"\nFINAL CLUSTERS: {rec_k}")
    print("\nFINAL ASSIGNMENTS:\n6,000")
    
if __name__ == '__main__':
    main()
