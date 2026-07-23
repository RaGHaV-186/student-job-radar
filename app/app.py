import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from build_pool import build_all

st.set_page_config(page_title="Job Radar", layout="wide")

STUDENT_TERMS = (
    "werkstudent", "praktikum", "praktikant", "internship",
    "working student", "studentische", "abschlussarbeit", "trainee",
)


def is_student_role(posting):
    title = posting["title"].lower()
    return any(term in title for term in STUDENT_TERMS)


@st.cache_resource
def load_everything():
    return build_all()


postings, methods = load_everything()

st.title("Job Radar")
st.caption(
    f"{len(postings)} job postings — TF-IDF, BM25 and LSA implemented from scratch"
)

col_query, col_method = st.columns([3, 1])
with col_query:
    query = st.text_input("Search", placeholder="werkstudent python")
with col_method:
    mode = st.selectbox("Method", ["bm25", "tfidf", "lsa", "compare all"])

col_slider, col_filter = st.columns([3, 1])
with col_slider:
    top_k = st.slider("Results", 3, 20, 5)
with col_filter:
    student_only = st.checkbox("Student roles only")


def get_results(method_name):
    depth = top_k * 4 if student_only else top_k
    results = methods[method_name](query, depth)
    if student_only:
        results = [(d, s) for d, s in results if is_student_role(postings[d])]
    return results[:top_k]


def render(results):
    if not results:
        st.info("No matching postings.")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        posting = postings[doc_id]
        with st.container(border=True):
            st.markdown(f"**{rank}. {posting['title']}**")

            bits = [posting.get("company", ""), posting.get("location", "")]
            if posting.get("remote"):
                bits.append("Remote")
            st.caption(" · ".join(b for b in bits if b))
            st.caption(f"score {score:.4f}")

            description = posting.get("description", "")
            st.write(description[:300] + ("..." if len(description) > 300 else ""))

            if posting.get("url"):
                st.markdown(f"[Open posting]({posting['url']})")


if query:
    if mode == "compare all":
        cols = st.columns(3)
        for col, name in zip(cols, ["tfidf", "bm25", "lsa"]):
            with col:
                st.subheader(name.upper())
                render(get_results(name))
    else:
        render(get_results(mode))
else:
    st.info("Type a query above to search.")