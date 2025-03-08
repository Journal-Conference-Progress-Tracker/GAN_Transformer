import numpy as np
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

def bootstrap_sampling(data, n_samples):
    """
    Generate new samples by bootstrap sampling (sampling with replacement).

    Parameters:
        data (np.ndarray): Array of shape (N, d) where N is number of samples and d is dimensionality.
        n_samples (int): Number of new samples to generate.
    
    Returns:
        np.ndarray: New samples sampled with replacement from the data.
    """
    indices = np.random.choice(data.shape[0], size=n_samples, replace=True)
    return data[indices]

def linear_interpolation_sampling(data, n_samples):
    """
    Generate new samples by linearly interpolating between two random data points.

    Parameters:
        data (np.ndarray): Array of shape (N, d).
        n_samples (int): Number of new samples to generate.
    
    Returns:
        np.ndarray: New synthetic samples generated via linear interpolation.
    """
    new_samples = []
    for _ in range(n_samples):
        # Pick two distinct random indices
        idx1, idx2 = np.random.choice(data.shape[0], size=2, replace=False)
        alpha = np.random.rand()  # random weight between 0 and 1
        sample = (1 - alpha) * data[idx1] + alpha * data[idx2]
        new_samples.append(sample)
    return np.array(new_samples)

def smote_sampling(data, n_samples, k=5):
    """
    Generate new samples using a SMOTE-like approach.
    For each synthetic sample, a point is chosen and one of its k-nearest neighbors is used to interpolate.

    Parameters:
        data (np.ndarray): Array of shape (N, d).
        n_samples (int): Number of synthetic samples to generate.
        k (int): Number of nearest neighbors to consider (default is 5).
    
    Returns:
        np.ndarray: Synthetic samples generated via SMOTE-like interpolation.
    """
    new_samples = []
    # Fit k-nearest neighbors model. k+1 because the point itself is included.
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    for _ in range(n_samples):
        idx = np.random.choice(data.shape[0])
        sample = data[idx]
        distances, indices = nbrs.kneighbors(sample.reshape(1, -1))
        # Exclude the point itself (first neighbor) and choose a random neighbor from the rest
        neighbor_idx = np.random.choice(indices[0][1:])
        neighbor = data[neighbor_idx]
        alpha = np.random.rand()
        synthetic_sample = sample + alpha * (neighbor - sample)
        new_samples.append(synthetic_sample)
    return np.array(new_samples)

def kde_sampling(data, n_samples, bandwidth=None):
    """
    Generate new samples from an estimated density using Gaussian Kernel Density Estimation (KDE).

    Parameters:
        data (np.ndarray): Array of shape (N, d). Note: gaussian_kde expects data of shape (d, N).
        n_samples (int): Number of new samples to generate.
        bandwidth (float or str, optional): The bandwidth to use. If None, the default method is used.
    
    Returns:
        np.ndarray: New samples generated from the KDE model.
    """
    # Transpose data to shape (d, N) as expected by gaussian_kde
    kde = gaussian_kde(data.T, bw_method=bandwidth)
    # kde.resample returns an array of shape (d, n_samples), so transpose it back.
    samples = kde.resample(n_samples).T
    return samples

def gmm_sampling(data, n_samples, n_components=2, random_state=None):
    """
    Generate new samples by fitting a Gaussian Mixture Model (GMM) to the data.

    Parameters:
        data (np.ndarray): Array of shape (N, d).
        n_samples (int): Number of new samples to generate.
        n_components (int): Number of Gaussian components for the mixture model.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        np.ndarray: New samples generated from the fitted GMM.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(data)
    samples, _ = gmm.sample(n_samples)
    return samples

