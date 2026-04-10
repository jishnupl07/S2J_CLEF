import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import json

# Load training data from JSON file
print("Loading training data...")
with open("./dataset/train_next_picto.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract sentences from 'tgt' field
sentences = [item["tgt"] for item in data]
tokenized_data = [s.split() for s in sentences]

# -------------------------
# TRAINING MULTIPLE MODELS (BACKOFF)
# -------------------------
print("Training N-gram models (1-gram, 2-gram, 3-gram)...")
unigram_counts = Counter()
bigram_model = defaultdict(Counter)
trigram_model = defaultdict(Counter)

for tokens in tokenized_data:
    # Update unigram for global fallback
    unigram_counts.update(tokens)
    
    # Bigrams
    padded_2 = ["<s>"] + tokens + ["</s>"]
    for gram in ngrams(padded_2, 2):
        prefix = (gram[0],)
        next_word = gram[1]
        bigram_model[prefix][next_word] += 1
        
    # Trigrams
    padded_3 = ["<s>", "<s>"] + tokens + ["</s>"]
    for gram in ngrams(padded_3, 3):
        prefix = (gram[0], gram[1])
        next_word = gram[2]
        trigram_model[prefix][next_word] += 1

# Get the most common word for absolute fallback
most_common_word = unigram_counts.most_common(1)[0][0] if unigram_counts else "the"

# -------------------------
# PREDICTION FUNCTION (STUPID BACKOFF)
# -------------------------
def predict_next(prefix_tokens):
    # Trigram approach
    if len(prefix_tokens) >= 2:
        prefix = tuple(prefix_tokens[-2:])
        if prefix in trigram_model:
            for pred, _ in trigram_model[prefix].most_common():
                if pred != "</s>": 
                    return pred
    
    # Bigram approach
    if len(prefix_tokens) >= 1:
        prefix = (prefix_tokens[-1],)
        if prefix in bigram_model:
            for pred, _ in bigram_model[prefix].most_common():
                if pred != "</s>": 
                    return pred

    # Absolute fallback
    return most_common_word

# -------------------------
# INFERENCE ON TEST SET
# -------------------------
print("Loading test data...")
with open("./dataset/test_next_picto.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Predicting for {len(test_data)} test samples...")
results = []
for item in test_data:
    prefix_text = item.get("tgt", "")
    prefix_tokens = prefix_text.split()
    prediction = predict_next(prefix_tokens)
    
    # Ensure we don't return </s> as the final prediction for the submission
    if prediction == "</s>":
        prediction = most_common_word

    results.append({
        "id": item["id"],
        "next_picto": prediction
    })

# -------------------------
# SAVING RESULTS
# -------------------------
output_file = "submission.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Successfully saved predictions to {output_file}")

# -------------------------
# QUICK TESTS (DEBUG)
# -------------------------
test_cases = [
    "remain to the house you suffer too",
    "the seat of be",
    "the seat of the county is"
]

print("\nDebug Tests:")
for tc in test_cases:
    tokens = tc.split()
    prediction = predict_next(tokens)
    print(f"Input: {tokens}")
    print(f"Predicted next token: {prediction}")