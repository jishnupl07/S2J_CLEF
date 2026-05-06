# Pictogram Sequence Prediction

## Overview

Next-token prediction over a constrained pictogram vocabulary. Given a partial sequence of pictogram tokens, the model predicts the most likely next token using a statistical n-gram ensemble (trigram → bigram → unigram fallback with Laplace smoothing and source-word boosting).

## Project Structure

```
IMGCLEF/
├── dataset/
│   ├── train_next_picto.json   # 63,675 training samples
│   └── test_next_picto.json    # 1,986 test samples
├── main.py                     # Prediction pipeline (train + predict + save)
├── submission.json             # Final output — 1,986 entries, 10 candidates each
├── workflow.txt                # Task specification & design notes
├── requirements.txt            # Dependencies (stdlib only)
└── README.md                   # This file
```

## How It Works

### Data Format

Each sample contains:

| Field   | Description                              |
|---------|------------------------------------------|
| `id`    | Unique sample identifier                 |
| `src`   | Natural-language sentence                |
| `tgt`   | Corresponding pictogram token sequence   |
| `pictos`| Pictogram IDs (aligned with `tgt`)       |

### Model

A backoff n-gram model with Laplace smoothing:

1. **Trigram** — Uses the last 2 tokens as context. Primary predictor.
2. **Bigram** — Uses the last token. Fallback when no trigram context exists.
3. **Unigram** — Global token frequencies. Final fallback.

Additionally, tokens appearing in the `src` field receive a 1.5× score boost to leverage the natural-language signal.

### Pipeline

```
Train JSON ──► Build n-gram counts ──► For each test sample:
                                         context tokens → score vocab
                                         → rank top 10 → emit candidates
                                       ──► submission.json
```

## Quick Start

```bash
# No external packages needed — runs on Python 3 stdlib
python main.py
```

**Output:** `submission.json` with 1,986 entries, each containing an `id` and a ranked list of 10 candidate tokens.

## Submission Format

```json
{
  "id": "common_voice_fr_h3zlwha1.mp3",
  "candidates": ["family", "of", "by", "woman", "little_girl",
                  "house", "new", "and", "chapel", "schoolhouse"]
}
```

## Key Statistics

| Metric              | Value   |
|---------------------|---------|
| Training samples    | 63,675  |
| Test samples        | 1,986   |
| Vocabulary size     | ~6,841  |
| Candidates per entry| 10      |
| External deps       | None    |

## Evaluation Metrics

The competition evaluates on:

- **Top-N Accuracy** (N = 1, 3, 5, 10) — is the correct token in the top N?
- **Semantic Similarity** — cosine similarity via `all-MiniLM-L6-v2` embeddings, rewarding near-synonyms.
