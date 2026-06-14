#!/usr/bin/env python3
"""Find the closest intents to a natural-language query."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import hnswlib
import numpy as np

from intent_utils import MiniLMEmbedder, load_json, normalize_intent_text


@dataclass
class IntentRecord:
    original_intents: list[str]
    intent: str
    questions: list[str]


def load_pairs(path: str) -> list[IntentRecord]:
    data = load_json(path)
    rows = data.get("pairs") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON must be a non-empty array or contain a 'pairs' array.")

    pairs = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Item {index + 1} must be an object.")
        intent = str(row.get("intent", "")).strip()
        raw_originals = row.get("original_intents", [])
        if not isinstance(raw_originals, list):
            raise ValueError(f"Item {index + 1} 'original_intents' must be an array.")
        if "original_intent" in row:
            raw_originals = [row["original_intent"], *raw_originals]
        original_intents = list(
            dict.fromkeys(str(value).strip() for value in raw_originals if str(value).strip())
        )
        raw_questions = row.get("questions", [])
        if not isinstance(raw_questions, list):
            raise ValueError(f"Item {index + 1} 'questions' must be an array.")
        questions = list(
            dict.fromkeys(str(question).strip() for question in raw_questions if str(question).strip())
        )
        if not intent or not questions:
            raise ValueError(f"Item {index + 1} requires 'intent' and non-empty 'questions'.")
        pairs.append(
            IntentRecord(
                original_intents=original_intents,
                intent=intent,
                questions=questions,
            )
        )
    return pairs


def build_pair_embeddings(
    embedder: MiniLMEmbedder, pairs: list[IntentRecord]
) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    texts = []
    owners = []
    sources = []
    for pair_index, pair in enumerate(pairs):
        entries = [
            ("intent", normalize_intent_text(pair.intent)),
            *[("original_intent", normalize_intent_text(value)) for value in pair.original_intents],
            *[("question", value) for value in pair.questions],
        ]
        for source, text in entries:
            texts.append(text)
            owners.append(pair_index)
            sources.append(source)
    return embedder.encode(texts), owners, sources, texts


def build_hnsw_index(embeddings: np.ndarray) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
    index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
    index.add_items(embeddings, np.arange(len(embeddings)))
    index.set_ef(min(100, len(embeddings)))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Search intents using HNSW cosine similarity.")
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--file", default="intents.json", help="Intent pairs JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum cosine score")
    args = parser.parse_args()

    pairs = load_pairs(args.file)
    embedder = MiniLMEmbedder()
    pair_embeddings, owners, sources, indexed_texts = build_pair_embeddings(embedder, pairs)
    index = build_hnsw_index(pair_embeddings)
    query_embedding = embedder.encode([args.query.strip()])[0]
    limit = max(1, min(args.top_k, len(pairs)))

    candidate_count = min(len(pair_embeddings), max(limit * 10, limit))
    best_by_pair: dict[int, tuple[float, int]] = {}
    while True:
        index.set_ef(max(100, candidate_count))
        labels, distances = index.knn_query(query_embedding, k=candidate_count)
        for vector_index, distance in zip(labels[0], distances[0]):
            vector_index = int(vector_index)
            pair_index = owners[vector_index]
            score = 1.0 - float(distance)
            if pair_index not in best_by_pair or score > best_by_pair[pair_index][0]:
                best_by_pair[pair_index] = (score, vector_index)
        if len(best_by_pair) >= limit or candidate_count == len(pair_embeddings):
            break
        candidate_count = min(len(pair_embeddings), candidate_count * 2)

    ranked = sorted(best_by_pair.items(), key=lambda item: item[1][0], reverse=True)[:limit]

    found = False
    for rank, (item_index, (score, vector_index)) in enumerate(ranked, start=1):
        if args.min_score is not None and score < args.min_score:
            continue
        found = True
        pair = pairs[int(item_index)]
        print(f"{rank}. {pair.intent} ({score:.4f})")
        print(f"   matched {sources[vector_index]}: {indexed_texts[vector_index]}")

    if not found:
        print("No intents matched the minimum score.")


if __name__ == "__main__":
    main()
