"""Partitioning Around Medoids (PAM) — a minimal k-medoids implementation.

scikit-learn does not provide k-medoids, so we implement the classic algorithm here.
It is intentionally short and readable rather than fast.
"""

import numpy as np
from sklearn.metrics import pairwise_distances


def pam(X, k, max_iter=100, random_state=0):
    """Cluster X into k clusters around medoids.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    k : int
        Number of clusters (must be chosen in advance).
    max_iter : int
        Maximum number of swap iterations.
    random_state : int
        Seed for the initial medoid selection.

    Returns
    -------
    medoids : ndarray of shape (k,)
        Indices into X of the medoids.
    cost : float
        Total cost: the sum over all points of the distance to the nearest medoid.
    """
    D = pairwise_distances(X)          # full n x n distance matrix
    n = len(X)
    rng = np.random.default_rng(random_state)

    medoids = rng.choice(n, size=k, replace=False)

    def cost(idx):
        return D[:, idx].min(axis=1).sum()

    current = cost(medoids)

    for _ in range(max_iter):
        best_swap, best_cost = None, current

        for m in range(k):                      # for each medoid ...
            for cand in range(n):               # ... try every other point
                if cand in medoids:
                    continue
                trial = medoids.copy()
                trial[m] = cand
                c = cost(trial)
                if c < best_cost:
                    best_swap, best_cost = trial, c

        if best_swap is None:                   # no improving swap -> stop
            break

        medoids, current = best_swap, best_cost

    return medoids, current


def labels(X, medoids):
    """Assign every point in X to its nearest medoid."""
    return pairwise_distances(X)[:, medoids].argmin(axis=1)
