import json
from collections import defaultdict, Counter
import os

# -------- FILE PATHS --------
TRAIN_FILE = "dataset/train_next_picto.json"
TEST_FILE = "dataset/test_next_picto.json"
OUTPUT_FILE = "submission.json"

print("Working Dir:", os.getcwd())

# -------- LOAD TRAIN DATA --------
train_sentences = []

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

    for obj in data:
        tokens = obj["tgt"].split()
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]
        train_sentences.append(tokens)

print("Loaded training sentences:", len(train_sentences))

# -------- BUILD COUNTS --------
unigram_counts = Counter()
bigram_counts = defaultdict(Counter)
trigram_counts = defaultdict(Counter)

vocab = set()

for sentence in train_sentences:
    for i in range(len(sentence)):
        token = sentence[i]
        unigram_counts[token] += 1
        vocab.add(token)

        if i >= 1:
            bigram_counts[sentence[i - 1]][token] += 1

        if i >= 2:
            trigram_counts[(sentence[i - 2], sentence[i - 1])][token] += 1

VOCAB_SIZE = len(vocab)
print("Vocab size:", VOCAB_SIZE)

# -------- SCORING FUNCTION --------
def get_scores(context_tokens, src_tokens):
    tokens = ["<s>", "<s>"] + context_tokens
    prev2, prev1 = tokens[-2], tokens[-1]

    scores = {}

    # ---- TRIGRAM ----
    if (prev2, prev1) in trigram_counts:
        candidates = trigram_counts[(prev2, prev1)]
        total = sum(candidates.values())

        for word in vocab:
            if word in ["<s>", "</s>"]:
                continue

            prob = (candidates[word] + 1) / (total + VOCAB_SIZE)
            scores[word] = prob

    # ---- BIGRAM ----
    elif prev1 in bigram_counts:
        candidates = bigram_counts[prev1]
        total = sum(candidates.values())

        for word in vocab:
            if word in ["<s>", "</s>"]:
                continue

            prob = (candidates[word] + 1) / (total + VOCAB_SIZE)
            scores[word] = prob

    # ---- UNIGRAM ----
    else:
        total = sum(unigram_counts.values())
        for word in vocab:
            if word in ["<s>", "</s>"]:
                continue

            scores[word] = unigram_counts[word] / total

    # 🔥 BONUS: boost words appearing in src
    for word in scores:
        if word in src_tokens:
            scores[word] *= 1.5   # tweakable weight

    return scores


# -------- GET TOP-K --------
def predict_top_k(context_tokens, src_tokens, k=10):
    scores = get_scores(context_tokens, src_tokens)

    # Sort by probability
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in ranked[:k]]


# -------- RUN ON TEST --------
results = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

    for obj in data:
        tgt_tokens = obj["tgt"].split()
        src_tokens = obj.get("src", "").split()

        candidates = predict_top_k(tgt_tokens, src_tokens, k=10)

        results.append({
            "id": obj["id"],
            "candidates": candidates
        })

# -------- SAVE SUBMISSION --------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Saved submission to run.json")

# -------- SANITY CHECK --------
print("Total predictions:", len(results))
print("Sample:", results[0])