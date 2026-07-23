import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tfidf
import bm25
import lsa

EVAL_DIR = PROJECT_ROOT / "eval"
POOL_DEPTH = 10

QUERIES = [
    "machine learning intern",
    "ml intern",
    "werkstudent python",
    "python developer intern",
    "working student ai engineer",
    "java developer intern",
    "softwareentwickler",
    "entwickler berlin",
    "praktikum marketing",
    "koch gastronomie",
    "buchhaltung finanzen",
    "logistik lagermitarbeiter",
]


def build_all():
    """Build all three indexes once, return search functions for each."""
    postings = tfidf.load_postings()
    tokenized = tfidf.tokenize_postings(postings)

    doc_freq = tfidf.compute_doc_freq(tokenized)
    t_idf = tfidf.compute_idf(doc_freq, len(postings))
    t_index = tfidf.build_index(tokenized, t_idf)

    b_idf = bm25.compute_idf(doc_freq, len(postings))
    b_index = bm25.build_index(tokenized, b_idf)

    vocab = lsa.build_vocabulary(tokenized)
    l_idf = lsa.compute_idf(tokenized, len(postings))
    matrix = lsa.build_matrix(tokenized, vocab, l_idf)
    doc_vecs, topic_terms = lsa.fit_lsa(matrix)

    return postings, {
        "tfidf": lambda q, k: tfidf.search(q, t_index, top_k=k),
        "bm25": lambda q, k: bm25.search(q, b_index, top_k=k),
        "lsa": lambda q, k: lsa.search(q, vocab, l_idf, topic_terms, doc_vecs, top_k=k),
    }


def main():
    EVAL_DIR.mkdir(exist_ok=True)
    postings, methods = build_all()

    lines, skeleton = [], []
    for query in QUERIES:
        pool = {}
        for name, search_fn in methods.items():
            for doc_id, _ in search_fn(query, POOL_DEPTH):
                pool.setdefault(doc_id, set()).add(name)

        lines.append(f"\n{'=' * 70}\nQUERY: {query}\n{'=' * 70}")
        for doc_id in sorted(pool):
            found_by = ",".join(sorted(pool[doc_id]))
            title = postings[doc_id]["title"][:60]
            lines.append(f"  [{doc_id:3d}] ({found_by:16s}) {title}")

        skeleton.append({"query": query, "relevant": []})

    (EVAL_DIR / "pool.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(EVAL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(skeleton, f, ensure_ascii=False, indent=2)

    print(f"Wrote {EVAL_DIR / 'pool.txt'} and {EVAL_DIR / 'labels.json'}")


if __name__ == "__main__":
    main()