import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_pool import build_all

LABELS_PATH = PROJECT_ROOT / "eval" / "labels.json"
TOP_K = 5


def precision_at_k(retrieved, relevant, k=TOP_K):
    """Of the k shown, what fraction were relevant?"""
    if not retrieved:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k=TOP_K):
    """Of all relevant postings, what fraction did we show?"""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant, k=TOP_K):
    """1 / position of the first relevant result. 0 if none in top-k."""
    for position, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def evaluate(search_fn, labelled):
    precisions, recalls, rrs, per_query = [], [], [], []

    for item in labelled:
        relevant = set(item["relevant"])
        retrieved = [doc_id for doc_id, _ in search_fn(item["query"], TOP_K)]

        p = precision_at_k(retrieved, relevant)
        r = recall_at_k(retrieved, relevant)
        rr = reciprocal_rank(retrieved, relevant)

        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        per_query.append((item["query"], p, r, rr))

    n = len(precisions)
    return {
        "P@5": sum(precisions) / n,
        "R@5": sum(recalls) / n,
        "MRR": sum(rrs) / n,
        "per_query": per_query,
    }


def main():
    with open(LABELS_PATH, encoding="utf-8") as f:
        labelled = json.load(f)

    labelled = [item for item in labelled if item["relevant"]]
    print(f"Evaluating on {len(labelled)} queries\n")

    _, methods = build_all()
    results = {name: evaluate(fn, labelled) for name, fn in methods.items()}

    print(f"{'method':<10}{'P@5':>8}{'R@5':>8}{'MRR':>8}")
    print("-" * 34)
    for name, res in results.items():
        print(f"{name:<10}{res['P@5']:>8.3f}{res['R@5']:>8.3f}{res['MRR']:>8.3f}")

    names = list(results)
    print("\nPER-QUERY P@5")
    print(f"{'query':<32}" + "".join(f"{n:>9}" for n in names))
    print("-" * (32 + 9 * len(names)))
    for i, item in enumerate(labelled):
        row = "".join(f"{results[n]['per_query'][i][1]:>9.2f}" for n in names)
        print(f"{item['query'][:31]:<32}{row}")


if __name__ == "__main__":
    main()