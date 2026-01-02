import numpy as np
from sklearn.neighbors import NearestNeighbors
# ==========================================
# PART 2: CUSTOM OVERSAMPLING ALGORITHM
# ==========================================

# Implementation of Oversampling using KNN and Linear Interpolation.
def custom_oversample(X, y, k=5):
    X_minority = X[y == 1]
    X_majority = X[y == 0]
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    n_to_generate = n_majority - n_minority
    if n_to_generate <= 0:
        return X, y
    # Find k nearest neighbours of p in the minority class
    k_adjusted = min(len(X_minority), k+1)
    nbrs = NearestNeighbors(n_neighbors=k_adjusted).fit(X_minority)

    synthetic_samples = []

    # Repeat step 1-4 given above until the number of instances in minority and majority classes become equal.
    for _ in range(n_to_generate):
        """ 1)	Randomly select a point p in the minority class."""
        random_idx = np.random.randint(0, n_minority)
        p = X_minority[random_idx]

        """ 2)	Find k nearest neighbours of p in that minority class."""
        _, indices = nbrs.kneighbors([p])
        neighbor_indices = indices[0][1:] # Skip self

        if len(neighbor_indices) == 0:
             synthetic_samples.append(p)
             continue

        """ 3)	Randomly pick one of those k neighbours. """
        random_neighbor_idx = np.random.choice(neighbor_indices)
        neighbor = X_minority[random_neighbor_idx]  

        """ 4)	Create a linear equation with the randomly selected point p and its selected neighbour.
        Then, generate a synthetic sample s along that linear equation."""
        gap = np.random.random()
        s = p + (neighbor - p) * gap
        synthetic_samples.append(s)

    if len(synthetic_samples) > 0:
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.ones(len(X_synthetic))
        X_final = np.vstack((X, X_synthetic))
        y_final = np.hstack((y, y_synthetic))
        return X_final, y_final
    else:
        return X, y