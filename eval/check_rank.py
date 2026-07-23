import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import tfidf, lsa

postings = tfidf.load_postings()
tokenized = tfidf.tokenize_postings(postings)
vocab = lsa.build_vocabulary(tokenized)
idf = lsa.compute_idf(tokenized, len(postings))
S = np.linalg.svd(lsa.build_matrix(tokenized, vocab, idf), compute_uv=False)

print(f"singular values: {len(S)}")
print(f"above 1e-6: {(S > 1e-6).sum()}")
for i in [0, 50, 100, 200, 240, 250, 260, 280, 298]:
    print(f"  S[{i:3d}] = {S[i]:.6f}")