"""Natural semantic clustering helpers."""

from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from intent_utils import pick_cluster_center


def build_natural_clusters(
    embeddings: np.ndarray,
    distance_threshold: float = 0.8,
) -> np.ndarray:
    # Group embeddings by cosine distance without fixing the cluster count.
    raw_labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(embeddings)

    # Remap generated labels to compact deterministic ids.
    label_map = {label: index for index, label in enumerate(sorted(set(raw_labels)))}

    # Return one integer cluster id per intent.
    return np.asarray([label_map[label] for label in raw_labels], dtype=np.int32)


def build_cluster_rows(
    intents: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    representatives: int = 6,
) -> list[dict[str, Any]]:
    # Build the viewer record for each natural cluster.
    clusters: list[dict[str, Any]] = []

    # Process clusters in stable label order.
    for cluster_id in sorted(set(labels.tolist())):
        # Select all embedding rows assigned to this cluster.
        member_indices = np.where(labels == cluster_id)[0]

        # Use the real intent nearest the centroid as the cluster center.
        center, ranked_members = pick_cluster_center(intents, embeddings, member_indices)

        # Store the center, representative intents, and complete membership.
        clusters.append(
            {
                "id": int(cluster_id),
                "center": center,
                "representatives": ranked_members[:representatives],
                "intents": sorted(ranked_members),
                "size": len(ranked_members),
            }
        )

    # Put larger clusters first for easier browsing.
    clusters.sort(key=lambda item: (-item["size"], item["center"]))

    # Keep ids aligned with the sorted viewer list.
    for cluster_id, cluster in enumerate(clusters):
        cluster["id"] = cluster_id

    # Return browser-ready cluster records.
    return clusters
