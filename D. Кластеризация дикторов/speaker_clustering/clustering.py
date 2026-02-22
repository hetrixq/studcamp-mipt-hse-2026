from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_cosine_average(x: np.ndarray, *, n_clusters: int) -> np.ndarray:
    # scikit-learn >= 1.2 uses metric= instead of affinity=
    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="cosine",
            linkage="average",
        )

    return clustering.fit_predict(x)
