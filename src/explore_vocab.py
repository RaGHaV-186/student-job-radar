import json
from collections import Counter
from pathlib import Path

from preprocess import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "postings.json"

with open(DATA_PATH, encoding="utf-8") as f:
    postings = json.load(f)

doc_freq = Counter()

for posting in postings:
    text = f"{posting['title']} {posting['description']} {' '.join(posting['tags'])}"
    unique_tokens = set(preprocess(text))
    doc_freq.update(unique_tokens)

n_docs = len(postings)

print(f"Postings: {n_docs}")
print(f"Vocabulary size: {len(doc_freq)}")
print()

print("TOP 40 MOST COMMON")
for word, count in doc_freq.most_common(40):
    print(f"  {count:4d}  ({count / n_docs:5.1%})  {word}")

singletons = [w for w, c in doc_freq.items() if c == 1]
print()
print(f"Words in exactly 1 posting: {len(singletons)} "
      f"({len(singletons) / len(doc_freq):.1%} of vocabulary)")
print("Sample:", singletons[:25])