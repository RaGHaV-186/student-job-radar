import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "postings.json"

GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "eines", "einer", "und", "oder", "aber", "auch", "als", "am", "an", "auf",
    "aus", "bei", "bis", "durch", "für", "gegen", "im", "in", "ist", "sind",
    "mit", "nach", "nicht", "von", "vor", "zu", "zum", "zur", "über", "unter",
    "wir", "sie", "sich", "ihre", "ihren", "unser", "unsere", "uns", "du",
    "dich", "dir", "sein", "seine", "haben", "hat", "hast", "werden", "wird",
    "wie", "was", "wenn", "dass", "so", "es", "man", "kann", "können", "soll",
    "sowie", "dabei", "dazu", "damit", "beim", "vom", "ihr", "ihm", "diese",
    "dieser", "dieses", "alle", "allen", "mehr", "noch", "nur", "schon",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "into", "through", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "can", "could", "may", "might", "must", "you", "your",
    "we", "our", "us", "they", "their", "it", "its", "this", "that", "these",
    "those", "as", "than", "then", "there", "here", "all", "also", "more",
    "most", "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "too", "very", "just", "who", "what", "when", "where", "which",
}

CORPUS_STOPWORDS = {
    "m", "w", "d", "gmbh", "co", "kg", "ag",
    "deine", "deinen", "deinem", "deiner", "dein",
    "unseren", "unseres", "unserem",
    "dann", "bist", "um", "sehr", "gute", "guter", "gutes", "gut",
}

STOPWORDS = GERMAN_STOPWORDS | ENGLISH_STOPWORDS | CORPUS_STOPWORDS

TOKEN_PATTERN = re.compile(r"[a-zäöüß0-9]+")

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

MIN_TOKEN_LENGTH = 2

def preprocess(text):
    tokens = tokenize(text)
    return [
        token for token in tokens
        if token not in STOPWORDS and len(token) >= MIN_TOKEN_LENGTH
    ]

if __name__ == "__main__":
    import json

    with open(DATA_PATH, encoding="utf-8") as f:
        postings = json.load(f)

    posting = postings[0]
    text = f"{posting['title']} {posting['description']}"
    tokens = preprocess(text)

    print("TITLE:", posting["title"])
    print("TOKEN COUNT:", len(tokens))
    print("FIRST 40:", tokens[:40])