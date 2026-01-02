import numpy as np
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification, make_gaussian_quantiles
# ==========================================
# PART 1: DATASET GENERATION
# ==========================================
def create_dataset(dataset_type, n_samples=1000, imbalance_ratio=0.1, random_state=42):
    # 1. Generate Raw Data
    if dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state)
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    elif dataset_type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.5, random_state=random_state)
    elif dataset_type == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, flip_y=0.05, random_state=random_state)
    elif dataset_type == 'gaussian':
        X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=2, random_state=random_state)
    else:
        print("Invalid dataset type")

    # 2. Create Imbalance
    # We want minority class to be approximately n_samples * imbalance_ratio
    n_minority_target = int(n_samples * imbalance_ratio)
    minority_indices = np.where(y == 1)[0]
    # If we have too many minority samples, delete some
    if len(minority_indices) > n_minority_target:
        remove_n = len(minority_indices) - n_minority_target
        remove_indices = np.random.choice(minority_indices, remove_n, replace=False)
        X = np.delete(X, remove_indices, axis=0)
        y = np.delete(y, remove_indices)
    return X, y