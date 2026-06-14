# Intent Retrieval

Semantic intent retrieval using MiniLM embeddings and an HNSW cosine index.

## Data Format

`intents.json` contains one record per canonical intent:

```json
{
  "pairs": [
    {
      "original_intents": [
        "download-document",
        "get-policy-pdf"
      ],
      "intent": "download-policy-document",
      "questions": [
        "Where can I download my policy certificate?",
        "How can I download my policy brochure or PDF?"
      ]
    }
  ]
}
```

- `intent` is the canonical intent returned by search.
- `original_intents` contains zero or more legacy names, aliases, or source-system
  intent labels.
- `questions` contains one or more natural-language examples for the intent.

The root may be either `{"pairs": [...]}` or a plain array. The old singular
`original_intent` field is accepted during ingestion for backward compatibility.

## Ingestion

`load_pairs()` performs the following steps:

1. Load and parse the JSON file.
2. Select the root `pairs` array, or use the root itself when it is an array.
3. Require a non-empty canonical `intent`.
4. Require `original_intents` and `questions` to be arrays.
5. Remove empty values and duplicate aliases or questions while preserving order.
6. Require at least one question for every canonical intent.
7. Convert each validated object into an `IntentRecord`.

No aliases or questions are concatenated into a combined document.

## Embedding

The default model is `sentence-transformers/all-MiniLM-L6-v2`. It creates
384-dimensional normalized vectors.

Every text is embedded independently:

```text
canonical intent -> one vector
original intent  -> one vector per alias
question         -> one vector per question
```

Kebab-case intent names are converted to readable text before embedding:

```text
download-policy-document -> download policy document
```

Questions are embedded unchanged. Labels such as `Intent:` or
`Example question:` are never added.

### Why Vectors Are Not Smoothed

The system does not average, sum, or create a centroid from an intent's vectors.
Each vector retains the exact semantic representation of its own text.

For example, an intent with two aliases and three questions produces six
independent vectors:

```text
1 canonical intent + 2 original intents + 3 questions = 6 vectors
```

During retrieval, the highest-scoring vector becomes the score for its canonical
intent:

```text
intent_score = max(cosine_similarity(query, each intent-owned vector))
```

This max-score strategy means a strong question or alias match is not weakened
by unrelated or less precise examples.

## HNSW Index

The vectors are inserted into an in-memory `hnswlib` index configured with:

- Space: cosine
- `M`: 16
- `ef_construction`: 200
- Initial `ef` search value: up to 100

Each HNSW item ID maps to:

- Its canonical intent record
- Its source type: `intent`, `original_intent`, or `question`
- The exact indexed text

HNSW provides approximate nearest-neighbor search without scanning every vector.
This is unnecessary for 50 intents but scales much better as aliases and
questions grow.

## Retrieval Algorithm

For a query:

1. Embed the query with the same MiniLM model.
2. Ask HNSW for the nearest individual vectors.
3. Convert cosine distance to similarity:

   ```text
   similarity = 1 - cosine_distance
   ```

4. Group vector hits by canonical intent.
5. Keep only the highest-scoring vector for each intent.
6. Sort unique intents by that maximum score.
7. Return up to `--top-k` intents passing `--min-score`.

An intent with many aliases could otherwise fill the candidate list with its own
vectors. To prevent this, retrieval increases the HNSW candidate count until it
finds enough unique intents or exhausts the index.

The CLI prints the source and exact text that produced each intent's score:

```text
1. download-policy-document (0.6857)
   matched question: How can I download my policy brochure or PDF?
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The model is downloaded on first use and then loaded from the local cache.

## Usage

```bash
python get_intents.py --query "how to download brochure"
```

Options:

```bash
python get_intents.py \
  --file intents.json \
  --query "how to download brochure" \
  --top-k 5 \
  --min-score 0.30
```

## Improving Results

- Add real user phrasings to `questions`.
- Add legacy routing labels to `original_intents`.
- Keep examples specific to one canonical intent.
- Avoid duplicate or overly broad questions.
- Use `--min-score` to reject weak matches.
- Consider both the top score and its margin over the second result before
  automatically routing high-risk requests.
