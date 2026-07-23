"""
AI-Powered Resume Tracker & ATS Optimizer — Core Utilities
==========================================================
Handles resume parsing (PDF / DOCX), job-description extraction,
TF-IDF + Cosine-Similarity scoring, keyword analysis, and
actionable recommendations.
"""

from __future__ import annotations

import io
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# --- PDF parsing ---
import pdfplumber

# --- DOCX parsing ---
from docx import Document

# --- NLP / ML ---
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy English model (small is fast; use "en_core_web_md" for better accuracy)
def _load_spacy_model() -> spacy.Language:
    """Load the spaCy model, downloading it first if necessary."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        )
        return spacy.load("en_core_web_sm")

_NLP = _load_spacy_model()


# ──────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Structured output returned to the front-end."""
    match_score: float                       # 0-100
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    resume_text_preview: str = ""
    job_text_preview: str = ""


# ──────────────────────────────────────────────
#  Stop-word list (lightweight, no NLTK needed)
# ──────────────────────────────────────────────
_CUSTOM_STOPS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "been", "be", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "it", "its", "they", "them", "their", "what", "which", "who", "whom",
    "where", "when", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "about",
    "above", "after", "again", "also", "am", "any", "because", "before",
    "being", "between", "into", "through", "during", "out", "off", "over",
    "under", "further", "then", "once", "here", "there", "up", "down",
    "now", "while", "get", "got", "getting", "make", "made", "making",
    "like", "well", "back", "even", "still", "new", "want", "way", "use",
    "work", "working", "one", "two", "first", "last", "long", "great",
    "little", "right", "big", "high", "old", "different", "next", "small",
    "large", "part", "good", "great", "able", "etc", "eg", "ie",
}


# ──────────────────────────────────────────────
#  Parsing helpers
# ──────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return concatenated text from every page of a PDF."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Return concatenated paragraph text from a DOCX file."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the correct parser based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Upload PDF or DOCX.")


# ──────────────────────────────────────────────
#  Text pre-processing
# ──────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s+#]", " ", text)       # keep # for C#, C++
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Word-level tokenisation with stop-word removal."""
    tokens = _clean(text).split()
    return [t for t in tokens if t not in _CUSTOM_STOPS and len(t) > 1]


# ──────────────────────────────────────────────
#  Keyword extraction
# ──────────────────────────────────────────────

# Common tech / business keywords to look for (expandable)
_TECH_KEYWORDS: list[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    "html", "css", "react", "angular", "vue", "vuejs", "nodejs", "node",
    "express", "django", "flask", "fastapi", "spring", "springboot",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "jenkins", "ci/cd", "terraform", "ansible", "linux", "bash",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "dynamodb",
    "graphql", "rest", "restapi", "grpc", "microservices",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "scipy", "matplotlib", "opencv", "huggingface",
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "agile", "scrum", "kanban", "sprint", "product management",
    "data analysis", "data engineering", "etl", "airflow", "spark", "hadoop",
    "tableau", "power bi", "looker",
    "communication", "leadership", "team management", "problem solving",
    "critical thinking", "project management", "time management",
    "strategic planning", "stakeholder management", "mentoring",
    "cloud computing", "serverless", "lambda", "s3", "ec2",
    "blockchain", "web3", "solidity",
    "figma", "sketch", "adobe xd", "ui/ux", "design thinking",
    "seo", "sem", "digital marketing", "google analytics",
    "unit testing", "integration testing", "tdd", "pytest", "jest",
    "cypress", "selenium", "automation",
]


def extract_keywords(text: str) -> list[str]:
    """Return de-duplicated keywords found in *text*."""
    clean = _clean(text)
    found: list[str] = []
    for kw in _TECH_KEYWORDS:
        if kw in clean:
            found.append(kw)
    # Also grab noun-chunks from spaCy for domain-specific terms
    doc = _NLP(clean)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if (
            2 <= len(phrase) <= 40
            and not all(t in _CUSTOM_STOPS for t in phrase.split())
            and phrase not in found
        ):
            found.append(phrase)
    return found


# ──────────────────────────────────────────────
#  TF-IDF + Cosine Similarity scoring
# ──────────────────────────────────────────────

def compute_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity → 0-100 score."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),     # unigrams + bigrams
        max_features=10_000,
    )
    tfidf = vectorizer.fit_transform([text_a, text_b])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(float(sim) * 100, 2)


# ──────────────────────────────────────────────
#  Recommendation engine
# ──────────────────────────────────────────────

def _generate_recommendations(
    match_score: float,
    matched: list[str],
    missing: list[str],
    resume_text: str,
) -> list[str]:
    """Produce actionable tips based on the analysis."""
    tips: list[str] = []

    if match_score < 40:
        tips.append(
            "⚠️  Your resume has a low match score. Consider re-writing your "
            "summary and experience sections to incorporate more keywords from "
            "the job description."
        )
    elif match_score < 70:
        tips.append(
            "📊  Moderate match. Tailor your resume further by weaving in "
            "the missing keywords naturally within your bullet points."
        )
    else:
        tips.append(
            "✅  Strong match! Fine-tune your resume to pass ATS formatting "
            "checks and ensure consistent tense usage."
        )

    if missing:
        top_missing = missing[:10]
        tips.append(
            f"🔑  Consider adding these missing keywords: {', '.join(top_missing)}."
        )

    if "experience" not in resume_text.lower():
        tips.append(
            "📋  Add an 'Experience' or 'Work History' section — most ATS "
            "scanners prioritise it."
        )

    if "education" not in resume_text.lower():
        tips.append(
            "🎓  Include an 'Education' section. Many recruiters and ATS "
            "filters look for it."
        )

    if len(matched) < 5:
        tips.append(
            "💡  You matched very few keywords. Use the job description's "
            "exact phrasing (where honest) to improve ATS ranking."
        )

    if "achievements" not in resume_text.lower() and "quantif" not in resume_text.lower():
        tips.append(
            "📈  Quantify your achievements (e.g., 'Increased revenue by 30%'). "
            "Numbers stand out to both ATS and human reviewers."
        )

    tips.append(
        "📝  Use a clean, single-column format. Avoid tables, images, and "
        "fancy fonts that ATS parsers can't read."
    )

    return tips


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def analyze_resume(
    resume_bytes: bytes,
    resume_filename: str,
    job_description: str,
) -> AnalysisResult:
    """
    End-to-end pipeline:
    1. Parse resume
    2. Compute TF-IDF similarity
    3. Extract & compare keywords
    4. Generate recommendations
    """
    # 1. Parse
    resume_text = extract_text(resume_bytes, resume_filename)
    if not resume_text.strip():
        raise ValueError("Could not extract text from the uploaded file.")

    # 2. Similarity
    score = compute_similarity(resume_text, job_description)

    # 3. Keywords
    resume_kw = extract_keywords(resume_text)
    job_kw = extract_keywords(job_description)
    matched = sorted(set(resume_kw) & set(job_kw), key=str.lower)
    missing = sorted(set(job_kw) - set(resume_kw), key=str.lower)

    # 4. Recommendations
    recommendations = _generate_recommendations(score, matched, missing, resume_text)

    return AnalysisResult(
        match_score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        recommendations=recommendations,
        resume_text_preview=resume_text[:500] + ("…" if len(resume_text) > 500 else ""),
        job_text_preview=job_description[:500] + ("…" if len(job_description) > 500 else ""),
    )
