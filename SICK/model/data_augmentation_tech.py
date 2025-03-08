import numpy as np
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm  # Import tqdm for progress bars
import os
os.environ["OMP_NUM_THREADS"] = "1"

def conditional_bootstrap_sampling(x, y, n_samples, condition=None):
    """
    Generate new samples by bootstrap sampling (with replacement), optionally conditioned on y.
    If condition is an array-like, generate n_samples for each condition and concatenate.
    
    Parameters:
        x (np.ndarray): Array of shape (N, d) representing features.
        y (np.ndarray): Array of shape (N,) representing labels.
        n_samples (int): Number of samples to generate PER condition.
        condition (optional): Scalar or array-like specifying condition(s) on y.
        
    Returns:
        tuple: (new_x, new_y) arrays.
    """
    if condition is not None and not np.isscalar(condition):
        samples_list = []
        labels_list = []
        # Loop over each condition with a progress bar
        for cond in tqdm(condition, desc="Bootstrap conditions"):
            new_x_cond, new_y_cond = conditional_bootstrap_sampling(x, y, n_samples, condition=cond)
            samples_list.append(new_x_cond)
            labels_list.append(new_y_cond)
        return np.concatenate(samples_list, axis=0), np.concatenate(labels_list, axis=0)
    
    if condition is not None:
        mask = (y == condition)
        if np.sum(mask) == 0:
            raise ValueError("No samples with the specified condition.")
        x_filtered = x[mask]
        new_indices = np.random.choice(x_filtered.shape[0], size=n_samples, replace=True)
        new_x = x_filtered[new_indices]
        new_y = np.full(n_samples, condition)
    else:
        new_indices = np.random.choice(x.shape[0], size=n_samples, replace=True)
        new_x = x[new_indices]
        new_y = y[new_indices]
    return new_x, new_y

def conditional_linear_interpolation_sampling(x, y, n_samples, condition=None):
    """
    Generate new samples by linearly interpolating between two random data points,
    optionally conditioned on y. If condition is array-like, process each condition separately.
    
    Parameters:
        x (np.ndarray): Array of shape (N, d) representing features.
        y (np.ndarray): Array of shape (N,) representing labels.
        n_samples (int): Number of samples to generate PER condition.
        condition (optional): Scalar or array-like specifying condition(s) on y.
        
    Returns:
        tuple: (new_x, new_y) arrays.
    """
    if condition is not None and not np.isscalar(condition):
        samples_list = []
        labels_list = []
        for cond in tqdm(condition, desc="Interpolation conditions"):
            new_x_cond, new_y_cond = conditional_linear_interpolation_sampling(x, y, n_samples, condition=cond)
            samples_list.append(new_x_cond)
            labels_list.append(new_y_cond)
        return np.concatenate(samples_list, axis=0), np.concatenate(labels_list, axis=0)
    
    if condition is not None:
        mask = (y == condition)
        if np.sum(mask) < 2:
            raise ValueError("Not enough samples with the specified condition to perform interpolation.")
        x_filtered = x[mask]
        y_filtered = y[mask]
    else:
        x_filtered = x
        y_filtered = y
        
    new_x_list = []
    new_y_list = []
    n_filtered = x_filtered.shape[0]
    # Progress bar for generating samples
    for _ in tqdm(range(n_samples), desc="Interpolating samples", leave=False):
        idx1, idx2 = np.random.choice(n_filtered, size=2, replace=False)
        alpha = np.random.rand()
        sample = (1 - alpha) * x_filtered[idx1] + alpha * x_filtered[idx2]
        new_x_list.append(sample)
        new_y_list.append(y_filtered[idx1])
    return np.array(new_x_list), np.array(new_y_list)

def conditional_smote_sampling(x, y, n_samples, k=5, condition=None):
    """
    Generate new samples using a SMOTE-like approach, optionally conditioned on y.
    If condition is array-like, process each condition separately and concatenate.
    
    Parameters:
        x (np.ndarray): Array of shape (N, d) representing features.
        y (np.ndarray): Array of shape (N,) representing labels.
        n_samples (int): Number of synthetic samples to generate PER condition.
        k (int): Number of nearest neighbors to consider.
        condition (optional): Scalar or array-like specifying condition(s) on y.
        
    Returns:
        tuple: (new_x, new_y) arrays.
    """
    if condition is not None and not np.isscalar(condition):
        samples_list = []
        labels_list = []
        for cond in tqdm(condition, desc="SMOTE conditions"):
            new_x_cond, new_y_cond = conditional_smote_sampling(x, y, n_samples, k=k, condition=cond)
            samples_list.append(new_x_cond)
            labels_list.append(new_y_cond)
        return np.concatenate(samples_list, axis=0), np.concatenate(labels_list, axis=0)
    
    if condition is not None:
        mask = (y == condition)
        if np.sum(mask) < 2:
            raise ValueError("Not enough samples with the specified condition for SMOTE.")
        x_filtered = x[mask]
        y_filtered = y[mask]
    else:
        x_filtered = x
        y_filtered = y
        
    new_x_list = []
    new_y_list = []
    nbrs = NearestNeighbors(n_neighbors=min(k+1, x_filtered.shape[0]), n_jobs=-1).fit(x_filtered)
    n_filtered = x_filtered.shape[0]
    for _ in tqdm(range(n_samples), desc="Generating SMOTE samples", leave=False):
        idx = np.random.choice(n_filtered)
        sample = x_filtered[idx]
        distances, indices = nbrs.kneighbors(sample.reshape(1, -1))
        if indices.shape[1] < 2:
            neighbor = sample
        else:
            neighbor_idx = np.random.choice(indices[0][1:])
            neighbor = x_filtered[neighbor_idx]
        alpha = np.random.rand()
        synthetic_sample = sample + alpha * (neighbor - sample)
        new_x_list.append(synthetic_sample)
        new_y_list.append(y_filtered[idx])
    return np.array(new_x_list), np.array(new_y_list)

def conditional_kde_sampling(x, y, n_samples, bandwidth=None, condition=None, n_components=None):
    """
    Generate new samples using Gaussian KDE with an optional PCA step to reduce dimensions,
    optionally conditioned on y. If condition is array-like, process each condition separately.
    
    Parameters:
        x (np.ndarray): Array of shape (N, d) representing features.
        y (np.ndarray): Array of shape (N,) representing labels.
        n_samples (int): Number of samples to generate PER condition.
        bandwidth (float or str, optional): Bandwidth for KDE.
        condition (optional): Scalar or array-like specifying condition(s) on y.
        n_components (int, optional): Number of PCA components to reduce to.
    
    Returns:
        tuple: (samples, new_y) arrays.
    """
    if condition is not None and not np.isscalar(condition):
        samples_list = []
        labels_list = []
        for cond in tqdm(condition, desc="KDE conditions"):
            samples_cond, labels_cond = conditional_kde_sampling(x, y, n_samples, bandwidth, condition=cond, n_components=n_components)
            samples_list.append(samples_cond)
            labels_list.append(labels_cond)
        return np.concatenate(samples_list, axis=0), np.concatenate(labels_list, axis=0)
    
    if condition is not None:
        mask = (y == condition)
        if np.sum(mask) == 0:
            raise ValueError("No samples with the specified condition for KDE.")
        x_filtered = x[mask]
    else:
        x_filtered = x

    if n_components is not None:
        # Compute the effective rank of the filtered data
        effective_rank = np.linalg.matrix_rank(x_filtered)
        # Automatically reset n_components to the minimum of the provided n_components, 
        # the number of available samples, and the effective rank.
        n_components = min(n_components, x_filtered.shape[0], effective_rank) - 1
        pca = PCA(n_components=n_components)
        x_reduced = pca.fit_transform(x_filtered)
    else:
        x_reduced = x_filtered



    kde = gaussian_kde(x_reduced.T, bw_method=bandwidth)
    samples_reduced = kde.resample(n_samples).T

    if n_components is not None:
        samples = pca.inverse_transform(samples_reduced)
    else:
        samples = samples_reduced

    new_y = np.full(n_samples, condition) if np.isscalar(condition) else np.full(n_samples, condition[0])
    return samples, new_y

def conditional_gmm_sampling(x, y, n_samples, n_components=4, random_state=None, condition=None):
    """
    Generate new samples by fitting a GMM to the data, optionally conditioned on y.
    If condition is array-like, process each condition separately.
    
    Parameters:
        x (np.ndarray): Array of shape (N, d) representing features.
        y (np.ndarray): Array of shape (N,) representing labels.
        n_samples (int): Number of samples to generate PER condition.
        n_components (int): Number of Gaussian components.
        random_state (int, optional): Random state for reproducibility.
        condition (optional): Scalar or array-like specifying condition(s) on y.
    
    Returns:
        tuple: (new_x, new_y) arrays.
    """
    if condition is not None and not np.isscalar(condition):
        samples_list = []
        labels_list = []
        for cond in tqdm(condition, desc="GMM conditions"):
            new_x_cond, new_y_cond = conditional_gmm_sampling(x, y, n_samples, n_components, random_state, condition=cond)
            samples_list.append(new_x_cond)
            labels_list.append(new_y_cond)
        return np.concatenate(samples_list, axis=0), np.concatenate(labels_list, axis=0)
    
    if condition is not None:
        mask = (y == condition)
        if np.sum(mask) == 0:
            raise ValueError("No samples with the specified condition for GMM.")
        x_filtered = x[mask]
    else:
        x_filtered = x
        
    gmm = GaussianMixture(n_components=n_components, init_params='k-means++', warm_start=True, random_state=random_state, n_init=5)
    gmm.fit(x_filtered)
    samples, _ = gmm.sample(n_samples)
    new_y = np.full(n_samples, condition) if np.isscalar(condition) else np.full(n_samples, condition[0])
    return samples, new_y

# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.randn(500, 3)
    y = np.random.choice([0, 1], size=500)
    n_new_samples = 10
    condition = [0, 1]  # Process both conditions

    bs_x, bs_y = conditional_bootstrap_sampling(x, y, n_new_samples, condition=condition)
    li_x, li_y = conditional_linear_interpolation_sampling(x, y, n_new_samples, condition=condition)
    smote_x, smote_y = conditional_smote_sampling(x, y, n_new_samples, k=5, condition=condition)
    kde_x, kde_y = conditional_kde_sampling(x, y, n_new_samples, condition=condition, n_components=2)
    gmm_x, gmm_y = conditional_gmm_sampling(x, y, n_new_samples, n_components=3, random_state=42, condition=condition)

    print("Bootstrap Samples:\n", bs_x, "\nLabels:", bs_y)
    print("\nLinear Interpolation Samples:\n", li_x, "\nLabels:", li_y)
    print("\nSMOTE-like Samples:\n", smote_x, "\nLabels:", smote_y)
    print("\nKDE Samples:\n", kde_x, "\nLabels:", kde_y)
    print("\nGMM Samples:\n", gmm_x, "\nLabels:", gmm_y)
