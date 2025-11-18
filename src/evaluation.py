import numpy as np
import pandas as pd
from scipy.signal import periodogram
from pymultifracs.simul import mrw_cumul, fbm
from scipy.signal import periodogram
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import networkx as nx


def spectral_overlap_index(mode1, mode2, fs):
    """Compute spectral overlap index between two modes."""
    f1, Pxx1 = periodogram(mode1, fs=fs)
    f2, Pxx2 = periodogram(mode2, fs=fs)
    if not np.allclose(f1, f2):
        raise ValueError("Frequency bins do not match.")
    num = np.sum(np.minimum(Pxx1, Pxx2))
    den = min(np.sum(Pxx1), np.sum(Pxx2))
    return 0.0 if den == 0 else num / den

def compute_soi_matrix(modes, fs):
    """Compute full SOI matrix and return mean off-diagonal value."""
    K = len(modes)
    soi = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            soi[i, j] = spectral_overlap_index(modes[i], modes[j], fs)
    mean_soi = np.mean(soi[np.triu_indices(K, k=1)])
    return soi, mean_soi

def get_band(freq):
    if 1 <= freq < 4:
        return "Delta"
    elif 4 <= freq < 8:
        return "Theta"
    elif 8 <= freq < 13:
        return "Alpha"
    elif 13 <= freq < 30:
        return "Beta"
    elif 30 <= freq < 80:
        return "Low Gamma"
    elif freq >= 80:
        return "High Gamma"
    return "NotClassified"

def label_band(cluster_id,cluster_peak_freq):
    f = cluster_peak_freq[cluster_id]
    if f < 1:
        return "Low-Freq Noise"
    elif 1 <= f < 13:
        return "Alpha"
    elif 13 <= f < 30:
        return "Beta"
    elif 30 <= f < 60:
        return "Low Gamma"
    elif 60 <= f <= 100:
        return "High Gamma"
    else:
        return "High-Freq Noise"
    
def reorder_corr_with_cah(corr, method='average'):
    """Return reordered correlation matrix and order from CAH."""
    dist = 1 - corr
    Z = linkage(squareform(dist, checks=False), method=method)
    order = dendrogram(Z, no_plot=True)["leaves"]
    corr_reordered = corr[np.ix_(order, order)]
    return corr_reordered, order

def assign_subgroups_from_corr(corr, threshold=0.7):
    """
    Use graph clustering to assign subgroups based on correlation threshold.
    Returns a list of subgroup labels (same order as input corr matrix).
    """
    adj = (corr >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    G = nx.from_numpy_array(adj)
    components = list(nx.connected_components(G))

    labels = np.full(len(corr), -1)
    for group_id, comp in enumerate(components):
        for i in comp:
            labels[i] = group_id
    return labels

def save_cluster_corr_bundle(cluster_id, corr, corr_reordered, indices, reordered_indices, output_dir="corr_matrices"):
    """Save everything in a single .npz file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"cluster_{cluster_id}_corr_bundle.npz")
    np.savez(out_path,
             corr=corr,
             corr_reordered=corr_reordered,
             indices=indices,
             indices_reordered=reordered_indices)
    print(f"💾 Saved: {out_path}")

def extract_subgroup_corrs_from_dendrogram(cluster_corrs, n_subgroups=2, method='average'):
    """
    For each cluster correlation matrix, cut the dendrogram into subgroups using hierarchical clustering.
    Return a dict of subgroup correlation matrices per cluster.
    """
    subgroup_corrs = {}

    for cluster_id, corr_mat in cluster_corrs.items():
        print(f"\n📊 Cluster {cluster_id}")

        if corr_mat.shape[0] < 2:
            print("⚠️ Skipping (less than 2 elements)")
            continue

        # Step 1: Distance matrix
        dist_mat = 1 - corr_mat
        dist_condensed = squareform(dist_mat, checks=False)

        # Step 2: Hierarchical clustering
        Z = linkage(dist_condensed, method=method)

        # Step 3: Cut into subgroups
        labels = fcluster(Z, t=n_subgroups, criterion='maxclust')

        # Step 4: Extract sub-correlation matrices
        subgroup_matrices = {}
        for g in range(1, n_subgroups + 1):
            subgroup_indices = np.where(labels == g)[0]
            if len(subgroup_indices) < 2:
                print(f"⚠️ Subgroup {g} has < 2 elements, skipping")
                continue
            sub_corr = corr_mat[np.ix_(subgroup_indices, subgroup_indices)]
            subgroup_matrices[g] = {
                "corr": sub_corr,
                "indices": subgroup_indices
            }

        subgroup_corrs[cluster_id] = subgroup_matrices

    return subgroup_corrs
