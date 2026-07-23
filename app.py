"""
AI-Powered Resume Tracker & ATS Optimizer — Streamlit Dashboard
================================================================
Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from utils import analyze_resume

# ──────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Tracker & ATS Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Custom CSS (minimal, professional)
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stMetric label { font-size: 0.95rem !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; }
    .keyword-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px 4px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .matched  { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .missing  { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .tip-box  { background: #fff3cd; padding: 12px 16px; border-radius: 8px;
                border-left: 4px solid #ffc107; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/3d-fluency/94/document.png",
        width=80,
    )
    st.title("AI Resume Tracker")
    st.markdown("### ATS Optimizer")
    st.divider()
    st.markdown(
        """
        **How it works:**
        1. Paste the Job Description
        2. Upload your Resume (PDF / DOCX)
        3. Get an instant ATS match analysis

        ---
        Built with **Streamlit**, **scikit-learn**, and **spaCy**
        """
    )
    st.divider()
    st.caption("v1.0.0  •  Open Source")


# ──────────────────────────────────────────────
#  Main content
# ──────────────────────────────────────────────
st.markdown("# 📄 AI-Powered Resume Tracker & ATS Optimizer")
st.markdown(
    "Upload your resume and paste a job description to see how well your "
    "resume matches — with keyword analysis and actionable recommendations."
)
st.divider()

# ── Inputs ──────────────────────────────────
col_jd, col_resume = st.columns(2)

with col_jd:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=320,
        placeholder="e.g. We are looking for a Python Developer with experience in Django, REST APIs, PostgreSQL, AWS…",
        label_visibility="collapsed",
    )

with col_resume:
    st.subheader("📁 Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

st.divider()

# ── Analyze ─────────────────────────────────
analyze_btn = st.button("🚀  Analyze Resume", type="primary", use_container_width=True)

if analyze_btn:
    if not job_description.strip():
        st.error("⚠️ Please paste a Job Description before analyzing.")
    elif uploaded_file is None:
        st.error("⚠️ Please upload your Resume (PDF or DOCX).")
    else:
        with st.spinner("Analyzing your resume against the job description…"):
            result = analyze_resume(
                resume_bytes=uploaded_file.read(),
                resume_filename=uploaded_file.name,
                job_description=job_description,
            )

        # ── Results ─────────────────────────────
        st.success("✅  Analysis complete!")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Match Score", f"{result.match_score}%",
                  delta="Above avg" if result.match_score >= 60 else "Below avg")
        m2.metric("Keywords Matched", len(result.matched_keywords))
        m3.metric("Keywords Missing", len(result.missing_keywords))

        st.divider()

        # ── Score gauge (progress bar) ──────────
        st.subheader("📊 Match Score")
        st.progress(result.match_score / 100)
        st.caption(f"Overall similarity: **{result.match_score}%**")

        st.divider()

        # ── Keywords ────────────────────────────
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.subheader("✅ Matched Keywords")
            if result.matched_keywords:
                tags_html = "".join(
                    f'<span class="keyword-tag matched">{kw}</span>'
                    for kw in result.matched_keywords
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No keywords matched. Consider adding more relevant skills.")

        with kw_col2:
            st.subheader("❌ Missing Keywords")
            if result.missing_keywords:
                tags_html = "".join(
                    f'<span class="keyword-tag missing">{kw}</span>'
                    for kw in result.missing_keywords
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.success("Great! No obvious keywords are missing.")

        st.divider()

        # ── Recommendations ─────────────────────
        st.subheader("💡 Recommendations")
        for tip in result.recommendations:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        st.divider()

        # ── Text previews (collapsible) ─────────
        with st.expander("📄 Resume Text Preview"):
            st.code(result.resume_text_preview, language=None)

        with st.expander("📋 Job Description Preview"):
            st.code(result.job_text_preview, language=None)


# ── Footer ──────────────────────────────────
st.divider()
st.caption(
    "Built with ❤️ using Streamlit • TF-IDF + Cosine Similarity • spaCy NLP  |  "
    "GitHub: reinachaturvedi09"
)
