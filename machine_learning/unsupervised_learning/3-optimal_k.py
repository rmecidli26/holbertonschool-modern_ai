#!/usr/bin/env python3
"""Module to find the optimal number of clusters for K-Means"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluates K-Means clustering quality using inertia and silhouette score"""
    ks = list(range(1, max_clusters + 1))
    inertia_values = []
    silhouette_values = []

    for k in ks:
        model = K_Means(X, n_clusters=k, random_state=random_state)
        inertia_values.append(model.inertia_)

        if k == 1:
            silhouette_values.append(0.0)
        else:
            score = metrics.silhouette_score(X, model.labels_)
            silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
