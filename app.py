#!/usr/bin/env python3
"""No-code web interface for intent clustering and retrieval."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.manifold import TSNE

from clustering import build_cluster_rows, build_natural_clusters
from intent_utils import DEFAULT_EMBEDDING_MODEL, MiniLMEmbedder, load_intents

# Create the Flask application.
app = Flask(__name__)

# Keep generated workspace state in memory for fast UI interaction.
workspace: dict[str, Any] = {}

# Protect model loading and workspace rebuilds from overlapping requests.
workspace_lock = threading.Lock()

# Load MiniLM lazily because startup should stay quick.
embedder: MiniLMEmbedder | None = None


def get_embedder() -> MiniLMEmbedder:
    # Reuse one model instance across all requests.
    global embedder
    # Load MiniLM only on the first operation that needs it.
    if embedder is None:
        embedder = MiniLMEmbedder(DEFAULT_EMBEDDING_MODEL)
    # Return the shared embedding model.
    return embedder


def parse_intents(value: Any) -> list[str]:
    # Accept either a JSON array or newline-separated text.
    raw_items = value if isinstance(value, list) else str(value or "").splitlines()
    # Normalize whitespace and discard blank rows.
    intents = [str(item).strip() for item in raw_items if str(item).strip()]
    # Reject duplicate names because ranking labels must stay unambiguous.
    if len(intents) != len(set(intents)):
        raise ValueError("Intent names must be unique.")
    # Require the same kebab-case contract used by the original catalog.
    invalid = [item for item in intents if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item)]
    # Return a useful validation message.
    if invalid:
        raise ValueError(f"Use kebab-case intent names. Invalid: {', '.join(invalid[:5])}")
    # Require enough examples for meaningful clustering.
    if len(intents) < 2:
        raise ValueError("Add at least two intents.")
    # Return the validated catalog.
    return intents


def project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    # Choose a valid perplexity for the current catalog size.
    perplexity = min(20, max(1, len(embeddings) - 1))
    # Project MiniLM vectors into two dimensions for the UI map.
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)


def rebuild_workspace(intents: list[str], threshold: float) -> dict[str, Any]:
    # Start timing the full build operation.
    started = time.perf_counter()
    # Embed every intent once for both methods.
    embeddings = get_embedder().encode(intents)
    # Create natural clusters without size constraints.
    labels = build_natural_clusters(embeddings, distance_threshold=threshold)
    # Build cluster centers and readable member lists.
    clusters = build_cluster_rows(intents, embeddings, labels)
    # Map intent names to the newly sorted cluster ids.
    intent_to_cluster = {
        intent: cluster["id"]
        for cluster in clusters
        for intent in cluster["intents"]
    }
    # Compute the visual projection from the original embeddings.
    projection = project_embeddings(embeddings)
    # Build the points consumed by the browser chart.
    points = [
        {
            "intent": intent,
            "cluster_id": intent_to_cluster[intent],
            "x": float(projection[index][0]),
            "y": float(projection[index][1]),
            "is_center": clusters[intent_to_cluster[intent]]["center"] == intent,
        }
        for index, intent in enumerate(intents)
    ]
    # Replace the shared workspace atomically.
    workspace.clear()
    workspace.update(
        {
            "intents": intents,
            "embeddings": embeddings,
            "clusters": clusters,
            "points": points,
            "threshold": threshold,
            "build_ms": (time.perf_counter() - started) * 1000,
        }
    )
    # Return only browser-safe state.
    return public_workspace()


def public_workspace() -> dict[str, Any]:
    # Return an empty state before the first build.
    if not workspace:
        return {"ready": False}
    # Return all UI-facing workspace fields.
    return {
        "ready": True,
        "intents": workspace["intents"],
        "clusters": workspace["clusters"],
        "points": workspace["points"],
        "threshold": workspace["threshold"],
        "build_ms": workspace["build_ms"],
        "model": DEFAULT_EMBEDDING_MODEL,
    }


def error_response(error: Exception, status: int = 400):
    # Return consistent JSON errors to the frontend.
    return jsonify({"error": str(error)}), status


@app.get("/")
def index():
    # Render the single-page no-code workspace.
    return render_template("index.html")


@app.get("/api/state")
def state():
    # Return the current app state.
    return jsonify(public_workspace())


@app.post("/api/build")
def build():
    try:
        # Read the browser payload.
        payload = request.get_json(force=True)
        # Validate the user-provided intent list.
        intents = parse_intents(payload.get("intents"))
        # Read the natural clustering sensitivity.
        threshold = float(payload.get("threshold", 0.8))
        # Keep the cosine distance threshold inside a meaningful range.
        if not 0.05 <= threshold <= 1.5:
            raise ValueError("Distance threshold must be between 0.05 and 1.5.")
        # Serialize model work so shared state stays consistent.
        with workspace_lock:
            result = rebuild_workspace(intents, threshold)
        # Return the completed workspace.
        return jsonify(result)
    except Exception as exc:
        # Surface validation and model errors to the UI.
        return error_response(exc)


@app.post("/api/query/vector")
def query_vector():
    try:
        # Require a built workspace.
        if not workspace:
            raise ValueError("Build clusters before running a query.")
        # Read the query controls.
        payload = request.get_json(force=True)
        query = str(payload.get("query", "")).strip()
        limit = int(payload.get("limit", 20))
        # Require a non-empty query.
        if not query:
            raise ValueError("Enter a query.")
        # Enforce the requested maximum of 50 results.
        limit = max(1, min(50, limit, len(workspace["intents"])))
        # Start retrieval timing.
        started = time.perf_counter()
        # Embed the user query with the same MiniLM model.
        query_embedding = get_embedder().encode([query])[0]
        # Compute exact cosine similarity against all normalized intent vectors.
        scores = workspace["embeddings"] @ query_embedding
        # Sort all intent positions by descending similarity.
        ranked_indices = np.argsort(-scores)[:limit]
        # Convert ranked positions into UI results.
        results = [
            {
                "rank": rank,
                "intent": workspace["intents"][int(index)],
                "score": float(scores[int(index)]),
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        # Return the exact vector ranking.
        return jsonify(
            {
                "method": "exact-cosine",
                "results": results,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    except Exception as exc:
        # Surface retrieval errors to the UI.
        return error_response(exc)


def initialize_default_workspace() -> None:
    # Load the repository's starter catalog.
    intents = load_intents(Path(__file__).with_name("intents.json"))
    # Build the initial UI state once at startup.
    rebuild_workspace(intents, 0.8)


if __name__ == "__main__":
    # Prepare the default catalog before accepting requests.
    initialize_default_workspace()
    # Start the local no-code application.
    app.run(host="127.0.0.1", port=5173, debug=False)
