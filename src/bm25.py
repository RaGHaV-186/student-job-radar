import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from preprocess import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "postings.json"

K1 = 1.5
B = 0.75


def load_postings():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def tokenize_postings(postings):
    tokenized = []
    for posting in postings:
        text = f"{posting['title']} {posting['description']} {' '.join(posting['tags'])}"
        tokenized.append(preprocess(text))
    return tokenized


def compute_doc_freq(tokenized):
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    return doc_freq


def compute_idf(doc_freq, n_docs):
    """BM25 IDF. Drops off harder than TF-IDF's for common words."""
    return {
        word: math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        for word, df in doc_freq.items()
    }

def build_index(tokenized, idf, k1=K1, b=B):
    """word -> {doc_id: bm25 score}."""
    doc_lengths = [len(tokens) for tokens in tokenized]
    avg_length = sum(doc_lengths) / len(doc_lengths)

    index = defaultdict(dict)

    for doc_id, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens)
        length_ratio = doc_lengths[doc_id] / avg_length
        denominator_length_part = k1 * (1 - b + b * length_ratio)

        for word, count in counts.items():
            numerator = count * (k1 + 1)
            denominator = count + denominator_length_part
            index[word][doc_id] = idf[word] * (numerator / denominator)

    return index

def search(query, index, top_k=5):
    query_tokens = preprocess(query)

    scores = defaultdict(float)
    for word in query_tokens:
        if word not in index:
            continue
        for doc_id, score in index[word].items():
            scores[doc_id] += score

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    postings = load_postings()
    tokenized = tokenize_postings(postings)
    doc_freq = compute_doc_freq(tokenized)
    idf = compute_idf(doc_freq, len(postings))
    index = build_index(tokenized, idf)

    print(f"Indexed {len(postings)} postings, {len(index)} unique terms\n")

    for query in ["python entwickler", "werkstudent machine learning", "berlin remote"]:
        print(f"QUERY: {query}")
        for doc_id, score in search(query, index):
            print(f"  {score:.4f}  [{doc_id}]  {postings[doc_id]['title'][:60]}")
        print()