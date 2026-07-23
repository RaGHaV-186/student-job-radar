import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tfidf
import bm25
import lsa
from evaluate import evaluate

LABELS_PATH = PROJECT_ROOT / "eval" / "labels.json"

K_VALUES = [25, 50, 100, 150, 200, 250, 275, 290, 299]
K1_VALUES = [0.5, 1.2, 1.5, 2.0]
B_VALUES = [0.0, 0.5, 0.75, 1.0]


def load_labels():
    with open(LABELS_PATH, encoding="utf-8") as f:
        labelled = json.load(f)
    return [item for item in labelled if item["relevant"]]


def tune_lsa(tokenized, postings, labelled):
    """SVD runs once; each k is just a slice of the result."""
    vocab = lsa.build_vocabulary(tokenized)
    idf = lsa.compute_idf(tokenized, len(postings))
    matrix = lsa.build_matrix(tokenized, vocab, idf)
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    print("\nLSA -- sweeping k")
    print(f"{'k':>6}{'P@5':>9}{'R@5':>9}{'MRR':>9}")
    print("-" * 33)
    for k in K_VALUES:
        kk = min(k, len(S))
        doc_vecs = U[:, :kk] * S[:kk]
        topic_terms = Vt[:kk, :]
        search_fn = lambda q, n: lsa.search(
            q, vocab, idf, topic_terms, doc_vecs, top_k=n
        )
        res = evaluate(search_fn, labelled)
        print(f"{k:>6}{res['P@5']:>9.3f}{res['R@5']:>9.3f}{res['MRR']:>9.3f}")


def tune_bm25(tokenized, postings, labelled):
    doc_freq = bm25.compute_doc_freq(tokenized)
    idf = bm25.compute_idf(doc_freq, len(postings))

    print("\nBM25 -- sweeping k1 and b (P@5)")
    header = "".join(f"{b:>9}" for b in B_VALUES)
    print(f"{'k1\\b':>7}{header}")
    print("-" * (7 + 9 * len(B_VALUES)))

    for k1 in K1_VALUES:
        row = ""
        for b in B_VALUES:
            index = bm25.build_index(tokenized, idf, k1=k1, b=b)
            search_fn = lambda q, n, ix=index: bm25.search(q, ix, top_k=n)
            row += f"{evaluate(search_fn, labelled)['P@5']:>9.3f}"
        print(f"{k1:>7}{row}")


def main():
    labelled = load_labels()
    postings = tfidf.load_postings()
    tokenized = tfidf.tokenize_postings(postings)

    print(f"Tuning on {len(labelled)} queries")
    print("NOTE: no held-out set -- these results are overfitted to this key.")

    tune_lsa(tokenized, postings, labelled)
    tune_bm25(tokenized, postings, labelled)


if __name__ == "__main__":
    main()