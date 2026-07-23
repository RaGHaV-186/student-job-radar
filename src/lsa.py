import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from preprocess import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "postings.json"

K_TOPICS = 100


def load_postings():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def tokenize_postings(postings):
    tokenized = []
    for posting in postings:
        text = f"{posting['title']} {posting['description']} {' '.join(posting['tags'])}"
        tokenized.append(preprocess(text))
    return tokenized


def build_vocabulary(tokenized):
    """word -> column number. Sorted so the mapping is reproducible."""
    vocab = sorted({token for tokens in tokenized for token in tokens})
    return {word: i for i, word in enumerate(vocab)}


def compute_idf(tokenized, n_docs):
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    return {word: math.log(n_docs / (1 + df)) for word, df in doc_freq.items()}

def build_matrix(tokenized, vocab_index, idf):
    """The 300 x 13708 grid, filled with TF-IDF weights."""
    matrix = np.zeros((len(tokenized), len(vocab_index)))

    for doc_id, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens)
        doc_length = len(tokens)
        for word, count in counts.items():
            matrix[doc_id, vocab_index[word]] = (count / doc_length) * idf[word]

    return matrix

def fit_lsa(matrix, k=K_TOPICS):
    """Returns (doc_vectors, topic_terms).

    doc_vectors: 300 x k  -- each posting in topic space
    topic_terms: k x 13708 -- how words map onto topics
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] * S[:k], Vt[:k, :]

def project_query(query, vocab_index, idf, topic_terms):
    """Turn the query into a k-number vector, same space as documents."""
    tokens = preprocess(query)
    q = np.zeros(len(vocab_index))

    if tokens:
        counts = Counter(tokens)
        length = len(tokens)
        for word, count in counts.items():
            if word in vocab_index:
                q[vocab_index[word]] = (count / length) * idf[word]

    return q @ topic_terms.T

def cosine_scores(query_vector, doc_vectors):
    """How closely does the query point in the same direction as each doc?"""
    q_norm = np.linalg.norm(query_vector)
    d_norms = np.linalg.norm(doc_vectors, axis=1)
    denom = q_norm * d_norms

    scores = np.zeros(len(doc_vectors))
    valid = denom > 1e-12
    scores[valid] = (doc_vectors[valid] @ query_vector) / denom[valid]
    return scores


def search(query, vocab_index, idf, topic_terms, doc_vectors, top_k=5):
    q = project_query(query, vocab_index, idf, topic_terms)
    scores = cosine_scores(q, doc_vectors)
    order = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in order]

if __name__ == "__main__":
    postings = load_postings()
    tokenized = tokenize_postings(postings)
    vocab_index = build_vocabulary(tokenized)
    idf = compute_idf(tokenized, len(postings))
    matrix = build_matrix(tokenized, vocab_index, idf)
    doc_vectors, topic_terms = fit_lsa(matrix)

    print(f"Matrix: {matrix.shape}  ->  topic space: {doc_vectors.shape}\n")

    for query in ["python entwickler", "werkstudent machine learning", "berlin remote"]:
        print(f"QUERY: {query}")
        for doc_id, score in search(query, vocab_index, idf, topic_terms, doc_vectors):
            print(f"  {score:.4f}  [{doc_id}]  {postings[doc_id]['title'][:60]}")
        print()