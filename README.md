# Intent Clustering

Natural MiniLM clustering, an interactive viewer, and exact cosine search.

## Approach

1. Convert kebab-case intents into readable text and embed them with
   `all-MiniLM-L6-v2`.
2. Create natural clusters using agglomerative average-linkage clustering with
   cosine distance. The UI threshold controls cluster sensitivity; cluster
   count and size are not fixed.
3. Represent each cluster using the real intent closest to its embedding
   centroid.
4. Project embeddings to two dimensions with t-SNE for the interactive viewer.
5. Embed each user query and rank all original intents by exact cosine
   similarity, returning up to 50 results.

## Requirements

- Python 3.10 or newer
- Internet access on first run to download `all-MiniLM-L6-v2`

Python dependencies are listed in `requirements.txt`:

- Flask
- NumPy
- scikit-learn
- sentence-transformers

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5173`.

The first startup downloads and caches the MiniLM model. Later startups reuse
the cached model.

## One-command setup

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python app.py
```
