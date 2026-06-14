#!/usr/bin/env python3
"""Shared helpers for loading intents and computing embeddings."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Keep the default embedding model in one place.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(path: str | Path) -> object:
    # Read the whole JSON file once.
    return json.loads(Path(path).read_text())


def load_intents(path: str | Path) -> list[str]:
    # Load the raw JSON payload.
    data = load_json(path)
    # Support either {"intents": [...]} or a plain array.
    intents = data["intents"] if isinstance(data, dict) else data
    # Reject empty or malformed payloads early.
    if not isinstance(intents, list) or not intents:
        raise ValueError("Intent file must contain a non-empty 'intents' array.")
    # Make sure each intent slug is unique.
    if len(intents) != len(set(intents)):
        raise ValueError("Intent names must be unique.")
    # Keep the intent format predictable for both models and tools.
    invalid = [intent for intent in intents if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", intent)]
    # Fail with a small sample if any slug breaks kebab-case.
    if invalid:
        raise ValueError(f"Intents must be kebab-case: {invalid[:3]}")
    # Return validated intent slugs.
    return intents


def normalize_intent_text(intent: str) -> str:
    # Replace dashes so the embedding model sees readable phrases.
    return intent.replace("-", " ")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    # Measure the vector length once.
    norm = np.linalg.norm(vector)
    # Return a unit vector when possible.
    return vector / norm if norm else vector


def centroid(vectors: np.ndarray) -> np.ndarray:
    # Average all member embeddings.
    center = vectors.mean(axis=0)
    # Keep the centroid normalized for cosine similarity.
    return normalize_vector(center)


def pick_cluster_center(intents: list[str], embeddings: np.ndarray, member_indices: np.ndarray) -> tuple[str, list[str]]:
    # Compute the semantic centroid for this cluster.
    center = centroid(embeddings[member_indices])
    # Score each member against the centroid.
    similarities = embeddings[member_indices] @ center
    # Rank members by closeness to the centroid.
    ranked_indices = member_indices[np.argsort(-similarities)]
    # Convert ranked indices back into intent names.
    ranked_intents = [intents[index] for index in ranked_indices]
    # Return the nearest real intent plus the full ranked list.
    return ranked_intents[0], ranked_intents


class MiniLMEmbedder:
    """Thin wrapper around the MiniLM sentence-transformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        # Store the chosen model name for artifacts and debugging.
        self.model_name = model_name
        # Load the transformer model once per process.
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Convert slugs into more natural text before embedding.
        readable = [normalize_intent_text(text) for text in texts]
        # Ask the model for normalized embeddings.
        vectors = self.model.encode(
            readable,
            normalize_embeddings=True,
            show_progress_bar=len(readable) > 32,
        )
        # Return a float32 matrix to keep memory usage modest.
        return np.asarray(vectors, dtype=np.float32)
