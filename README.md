# 📄 AI-Powered Resume Tracker & ATS Optimizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.4+-orange?style=for-the-badge&logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/spaCy-3.7+-green?style=for-the-badge&logo=spacy" alt="spaCy">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

An AI-powered web application that analyzes resumes against job descriptions, calculates ATS (Applicant Tracking System) match scores using **TF-IDF + Cosine Similarity**, extracts matched and missing keywords, and provides actionable recommendations to improve resume ranking.

> Built by **Reina Chaturvedi** — reinachaturvedi09@gmail.com

---

## 🎯 Project Overview

Most job seekers never know why their resume gets filtered out by ATS software. This tool bridges that gap by:

- Parsing PDF and DOCX resumes programmatically
- Comparing resume content against any job description using NLP
- Computing a percentage-based match score
- Identifying keywords the resume has — and the ones it's missing
- Providing clear, actionable tips to improve the resume's ATS compatibility

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **PDF / DOCX Parsing** | Extracts text from both PDF and Word documents using `pdfplumber` and `python-docx` |
| **TF-IDF Similarity Score** | Uses `scikit-learn`'s TF-IDF Vectorizer and Cosine Similarity for semantic matching |
| **Keyword Matching** | Identifies 100+ technical and soft-skill keywords from the job description |
| **Missing Keyword Detection** | Highlights skills the job requires but your resume lacks |
| **Actionable Recommendations** | Generates context-aware tips to improve your resume's ATS score |
| **Interactive Dashboard** | Clean, responsive Streamlit UI with metrics, keyword tags, and progress bars |
| **spaCy NLP Integration** | Leverages named-entity recognition for deeper text analysis |
| **Zero API Keys** | Fully local — no external APIs, no data leaves your machine |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend / Logic** | Python 3.9+ |
| **PDF Parsing** | pdfplumber |
| **DOCX Parsing** | python-docx |
| **Machine Learning** | scikit-learn (TF-IDF, Cosine Similarity) |
| **NLP** | spaCy (en_core_web_sm) |
| **Deployment** | Local / Streamlit Cloud |

---

## 📁 Project Structure

```
ai-resume-tracker/
├── app.py              # Streamlit web application (UI + entry point)
├── utils.py            # Core logic: parsing, scoring, keyword extraction
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step-by-Step

1. **Clone the repository**
   ```bash
   git clone https://github.com/reinachaturvedi09/ai-resume-tracker.git
   cd ai-resume-tracker
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```
   > **Note:** The app will auto-download this model on first run if it's missing.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** — Navigate to `http://localhost:8501`

---

## 📖 Usage Guide

1. **Paste the Job Description** — Copy the full job posting text into the text area on the left
2. **Upload Your Resume** — Drag-and-drop or browse for a `.pdf` or `.docx` file on the right
3. **Click "Analyze Resume"** — The app will process and display results instantly

### Understanding the Results

| Metric | What It Means |
|---|---|
| **Match Score** | Overall TF-IDF cosine similarity (0–100%) between your resume and the job description |
| **Matched Keywords** | Skills/keywords found in BOTH your resume and the job description |
| **Missing Keywords** | Skills the job requires that are NOT found in your resume |
| **Recommendations** | Actionable steps to improve your resume's ATS compatibility |

---

## 🔧 How It Works (Architecture)

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Input  │────▶│  text extraction │────▶│  text cleaning  │
│  (PDF/DOCX   │     │  (pdfplumber /   │     │  & tokenization │
│   + JD text) │     │   python-docx)   │     │  (regex + spaCy)│
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                        │
                         ┌──────────────────────────────┤
                         ▼                              ▼
                ┌────────────────┐            ┌─────────────────┐
                │  TF-IDF +      │            │  Keyword        │
                │  Cosine        │            │  Extraction &   │
                │  Similarity    │            │  Comparison     │
                │  (scikit-learn)│            │  (spaCy NLP)    │
                └───────┬────────┘            └────────┬────────┘
                        │                              │
                        ▼                              ▼
                ┌─────────────────────────────────────────────┐
                │           AnalysisResult                    │
                │  • match_score   • matched_keywords         │
                │  • missing_keywords   • recommendations     │
                └──────────────────────┬──────────────────────┘
                                       ▼
                              ┌─────────────────┐
                              │  Streamlit UI    │
                              │  (Dashboard)     │
                              └─────────────────┘
```

---

## 🧪 Example

**Job Description excerpt:**
> "Looking for a Python Developer with experience in Django, REST APIs, PostgreSQL, AWS, Docker, and CI/CD pipelines."

**Resume contains:** Python, Django, REST API, PostgreSQL

**Result:**
- Match Score: **72%**
- Matched: `python`, `django`, `rest api`, `postgresql`
- Missing: `aws`, `docker`, `ci/cd`
- Recommendation: "Consider adding these missing keywords: aws, docker, ci/cd."

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Reina Chaturvedi**
- 📧 reinachaturvedi09@gmail.com
- 🐙 GitHub: [reinachaturvedi09](https://github.com/reinachaturvedi09)

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) — Fast, beautiful web apps for data science
- [scikit-learn](https://scikit-learn.org/) — ML library for TF-IDF and similarity
- [spaCy](https://spacy.io/) — Industrial-strength NLP
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text extraction
- [python-docx](https://python-docx.readthedocs.io/) — DOCX parsing
