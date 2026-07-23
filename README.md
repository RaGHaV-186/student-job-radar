# Job Radar

A retrieval system over German/European job postings, with three ranking methods
implemented from scratch and evaluated against hand-labelled queries.

No `scikit-learn`, no `rank_bm25`, no vector database. TF-IDF, BM25 and LSA are
written directly from their formulas; `numpy` is used only for the SVD.

**Live app:** https://raghav-student-job-radar.streamlit.app/

---

## What this is

Most retrieval projects stop at "it returns plausible results." This one measures
whether it actually works, which turned out to be the interesting part — the
evaluation contradicted two of my starting assumptions.

| Method | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| TF-IDF | 0.491 | 0.685 | 0.818 |
| BM25 | 0.545 | 0.716 | 0.803 |
| **LSA (k=250)** | **0.564** | **0.746** | **0.841** |

Measured over 11 hand-labelled queries. Read the [Limitations](#limitations)
before drawing conclusions from these numbers.

---

## Corpus

300 job postings collected from the [Arbeitnow public API](https://arbeitnow.com/api/job-board-api)
(3 pages × 100), cleaned with BeautifulSoup, stored as a **frozen snapshot** in
`data/postings.json`.

Frozen deliberately: evaluation labels are tied to specific document IDs, so a
live corpus would invalidate them on every refresh.

**Schema:** `id`, `slug`, `title`, `company`, `description`, `tags`, `location`,
`remote`, `url`, `posted_at`, `date_collected`

**Searchable fields:** `title`, `description`, `tags`
**Metadata (displayed, not matched):** everything else

**Design decisions**

- *All job types kept* rather than filtering to tech — a diverse corpus gives
  retrieval something to discriminate between. In practice the snapshot skewed
  heavily to tech, marketing, consulting and accounting anyway.
- *Salary dropped* — rarely present in German postings.
- *Student filtering happens at display time, not index time*, so document IDs
  stay stable and the evaluation remains valid.

---

## Preprocessing

`src/preprocess.py` — one `preprocess(text) -> list[str]` function, called on
both documents at index time and queries at search time. Term identity is
defined in exactly one place.

1. **Lowercase**
2. **Tokenize** — regex `[a-zäöüß0-9]+`. The explicit `äöüß` matters: a naive
   `[a-z]+` pattern silently shreds `münchen` into `m` + `nchen` and
   `geschäftsführer` into three fragments.
3. **Stopwords** — German + English lists, plus corpus-specific terms
   (`m`, `w`, `d` from `(m/w/d)`, `gmbh`, `deine`, `unseren`, …) found by
   counting document frequency rather than guessing.

**Final vocabulary: 13,692 terms across 300 documents** — a matrix that is
roughly 99% zeros.

### Data contamination found during preprocessing

Counting document frequency surfaced four words sitting at ~50%: `find`, `jobs`,
`germany`, `arbeitnow`. They form the sentence *"Find Jobs in Germany on
Arbeitnow"* — a footer the API appends to roughly half its records.

Removed at the data layer (`src/clean_footer.py`), not stopworded, because it's a
collection bug rather than a vocabulary decision. The original snapshot is
preserved as `data/postings_raw.json`.

**48.1% of the vocabulary appears in exactly one document** — mostly German
compounds like `geschäftsführungsrolle` and `investitionsprojekten`. Dropping
singletons would have deleted half the vocabulary, including much of what users
actually type.

---

## Methods

### TF-IDF (`src/tfidf.py`)

Inverted index mapping `word -> {doc_id: score}`.

```
TF  = count / document_length
IDF = log(N / (1 + df))
```

Search sums TF·IDF across query terms and sorts. Documents that match no query
term are never touched.

### BM25 (`src/bm25.py`)

Same index structure, different scoring:

```
IDF × [count × (k₁+1)] / [count + k₁ × (1 − b + b × len/avg_len)]
```

- `k₁ = 1.5` — saturates term frequency, so the tenth mention of a word adds
  almost nothing over the second
- `b = 0.75` — tunable length normalization rather than TF-IDF's full division

### LSA (`src/lsa.py`)

Dense 300 × 13,692 TF-IDF matrix, factorised with `np.linalg.svd`, truncated to
`k` topics. Queries are projected into the same topic space and ranked by cosine
similarity.

Matches on latent topics rather than literal words, so it can connect
`werkstudent` to `working student` — in principle. See below for what actually
happened.

---

## Evaluation (`eval/`)

12 queries covering skills, roles, German compounds, bilingual pairs and
non-tech domains. **11 usable** — `koch gastronomie` was dropped because the
corpus contains no gastronomy postings.

**Method:** candidates pooled from the top-10 of all three methods
(`eval/build_pool.py`), then labelled by hand into `eval/labels.json`. Metrics in
`eval/evaluate.py`, parameter sweeps in `eval/tune.py`.

### Finding 1 — LSA's semantic matching did not materialise

The two synonym queries were the direct test of LSA's premise. They came out as
exact three-way ties:

| Query | TF-IDF | BM25 | LSA |
|---|---|---|---|
| `machine learning intern` | 0.60 | 0.60 | 0.60 |
| `ml intern` | 0.40 | 0.40 | 0.40 |

LSA's real wins were elsewhere — `werkstudent python` (1.00 vs BM25's 0.80) and
`entwickler berlin` (0.80 vs TF-IDF's 0.00). It scores best overall, but as a
well-normalized vector space model, not as a semantic one.

### Finding 2 — dimensionality reduction actively hurts on this corpus

Sweeping `k` produced a monotonic climb with no interior optimum:

| k | 25 | 50 | 100 | 150 | 200 | 250 | 275 | 299 |
|---|---|---|---|---|---|---|---|---|
| P@5 | 0.164 | 0.236 | 0.327 | 0.364 | 0.436 | 0.564 | 0.564 | 0.564 |

Performance improves the *less* compression is applied, then plateaus. At its
best setting LSA is effectively cosine similarity over the TF-IDF matrix — the
classic Vector Space Model — with the SVD contributing nothing.

The singular value spectrum explains why: it decays gradually across all 300
dimensions with **no elbow**. There is no compact set of dominant topics, so
truncation removes signal rather than noise. 300 documents is simply too little
text for SVD to learn stable semantic structure.

`eval/check_rank.py` confirms 277 of 300 singular values exceed 1e-6; the ~23
zeros trace to near-duplicate postings in the snapshot (nine near-identical
Steuerberater ads, several repeated titles).

### Finding 3 — BM25's saturation is measurable

`entwickler berlin`: TF-IDF **0.00**, BM25 **0.80** — the largest gap in the
evaluation. TF-IDF's unbounded term frequency let short postings repeating
"Berlin" outrank actual developer roles. BM25's saturation caps that, so matching
a second query term starts to matter more than repeating the first.

Supported independently by the parameter grid: `b = 0.0` (length normalization
off) is the worst column at every value of `k₁`.

### Finding 4 — no out-of-vocabulary handling

For `koch gastronomie`, where no query term exists in the vocabulary, all three
methods returned confident nonsense — nine tax advisors and two pest controllers
— rather than an empty result set. A production system should detect that no
query term is in the index and say so.

---

## Limitations

Stated plainly because they bound what the numbers mean:

- **Labels drawn from titles only**, drafted with LLM assistance rather than by
  reading full descriptions. This systematically under-credits methods that match
  on description text — which is precisely LSA's distinctive behaviour.
- **No held-out set.** `k` was tuned on the same 11 queries used to report
  results. The tuned numbers are overfitted.
- **11 queries is small.** One result moving one rank shifts P@5 by ~0.018, so
  differences under ~0.05 are noise. The BM25 parameter grid is entirely within
  that band.
- **Pooling bias.** Relevant postings that no method retrieved were never
  labelled and count as irrelevant.
- **`location` and `remote` are not searchable**, so `entwickler berlin` tests
  little more than `entwickler`.
- **No stemming** — `aufgabe` and `aufgaben` remain separate terms.
- **No compound splitting** — a query for `entwickler` does not match
  `softwareentwickler`.
- **No deduplication** in collection; near-duplicate postings cost ~23 matrix
  dimensions.

---

## Project structure

```
data/
  postings.json          frozen 300-posting snapshot
  postings_raw.json      pre-cleaning backup
src/
  collect.py             Arbeitnow API collection
  clean_footer.py        one-off footer removal
  preprocess.py          tokenizer + stopwords
  explore_vocab.py       document-frequency analysis
  tfidf.py               TF-IDF from scratch
  bm25.py                BM25 from scratch
  lsa.py                 LSA via truncated SVD
eval/
  build_pool.py          pooled candidate generation
  labels.json            hand-labelled ground truth
  evaluate.py            P@5, R@5, MRR
  tune.py                parameter sweeps
  check_rank.py          singular value diagnostics
app/
  app.py                 Streamlit interface
```

## Running it

```bash
git clone https://github.com/RaGHaV-186/student-job-radar.git
cd student-job-radar
uv venv && uv pip install -r requirements.txt

uv run python src/tfidf.py        # or bm25.py / lsa.py
uv run python eval/evaluate.py    # reproduce the results table
uv run streamlit run app/app.py   # the search interface
```

## What I would do next

1. **Re-label from full descriptions** to remove the bias against LSA, and add a
   held-out query set so tuning isn't self-confirming.
2. **Collect 5,000+ postings.** Every LSA finding here is a small-corpus artifact;
   the method deserves a fair test at a scale where topics can actually form.
3. **Deduplicate at collection time.**
4. **Make location a filter, not a search term** — it's structured metadata being
   handled as free text.
5. **Detect out-of-vocabulary queries** and return nothing rather than noise.