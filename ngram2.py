import json
from collections import defaultdict, Counter
import math


TRAIN_FILE = "dataset/train_next_picto.json"
TEST_FILE = "dataset/train_next_picto.json"
OUTPUT_FILE = "ngram_predictions.jsonl"


train_sentences = []

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        # tgt contains token sequence
        # Example: "there_is now an bus_station"
        tokens = obj["tgt"].split()

        # Add sentence boundary markers
        tokens = ["<s>", "<s>"] + tokens + ["</s>"]

        train_sentences.append(tokens)

print("Loaded training sentences:", len(train_sentences))



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
            prev1 = sentence[i - 1]
            bigram_counts[prev1][token] += 1

        if i >= 2:
            prev2, prev1 = sentence[i - 2], sentence[i - 1]
            trigram_counts[(prev2, prev1)][token] += 1

VOCAB_SIZE = len(vocab)

print("Vocabulary size:", VOCAB_SIZE)



def predict_next_token(context_tokens):
    """
    Predict next pictogram token using:
    trigram -> bigram -> unigram
    """

    # Add sentence markers
    tokens = ["<s>", "<s>"] + context_tokens

    prev2, prev1 = tokens[-2], tokens[-1]

    
    if (prev2, prev1) in trigram_counts:
        candidates = trigram_counts[(prev2, prev1)]

        best_token = None
        best_score = -1

        total = sum(candidates.values())

        for word in vocab:
            score = (candidates[word] + 1) / (total + VOCAB_SIZE)

            if score > best_score:
                best_score = score
                best_token = word

        return best_token


    elif prev1 in bigram_counts:
        candidates = bigram_counts[prev1]

        best_token = None
        best_score = -1

        total = sum(candidates.values())

        for word in vocab:
            score = (candidates[word] + 1) / (total + VOCAB_SIZE)

            if score > best_score:
                best_score = score
                best_token = word

        return best_token

   
    else:
        return unigram_counts.most_common(1)[0][0]



predictions = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        partial_tokens = obj["tgt"].split()

        predicted_token = predict_next_token(partial_tokens)

        output = {
            "id": obj["id"],
            "input_sequence": obj["tgt"],
            "predicted_next_token": predicted_token
        }

        predictions.append(output)



with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(json.dumps(pred, ensure_ascii=False) + "\n")

print("Predictions saved to:", OUTPUT_FILE)



sample = ["there_is", "now", "an"]
print("Sample Input:", sample)
print("Predicted Next Token:", predict_next_token(sample))