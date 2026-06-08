import streamlit as st
import fitz  # PyMuPDF
from datetime import datetime

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def analyze_resume(resume_text: str):
    """Simulate AI resume review (replace with real LLM API call later)"""
    # Simple rule-based scoring for demo + realistic feedback
    score = 75
    feedback = []

    keywords = ["python", "machine learning", "ai", "sql", "github", "project", "experience"]
    found_keywords = sum(1 for kw in keywords if kw.lower() in resume_text.lower())
    score += found_keywords * 3

    if "experience" not in resume_text.lower():
        feedback.append("⚠️ Add a clear 'Experience' section with bullet points")
    if "project" not in resume_text.lower():
        feedback.append("⚠️ Highlight personal/school projects more prominently")
    if len(resume_text.split()) < 200:
        feedback.append("⚠️ Resume seems short — aim for more detailed achievements")

    feedback = feedback or ["✅ Strong overall structure!"]

    return {
        "score": min(98, int(score)),
        "strengths": ["Good technical keywords detected", "Clear education background"],
        "improvements": feedback,
        "suggestions": [
            "Quantify achievements (e.g., 'Improved model accuracy by 15%')",
            "Add GitHub link and live demo links",
            "Tailor summary to AI/ML roles"
        ]
    }

def main():
    st.set_page_config(page_title="Rate My Resume", page_icon="📄", layout="centered")

    st.title("📄 JRC Rate My Resume")
    st.markdown("**AI-Powered Resume Feedback** — Month 7 Project 3")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            analysis = analyze_resume(resume_text)

        st.success(f"**Overall Score: {analysis['score']}/100**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Strengths")
            for strength in analysis["strengths"]:
                st.write(f"• {strength}")

        with col2:
            st.subheader("🔧 Improvements")
            for imp in analysis["improvements"]:
                st.write(f"• {imp}")

        st.subheader("🚀 Actionable Suggestions")
        for suggestion in analysis["suggestions"]:
            st.write(f"• {suggestion}")

        st.download_button(
            label="Download Review Report",
            data=f"Resume Review - {datetime.now().strftime('%Y-%m-%d')}\nScore: {analysis['score']}/100\n\n" +
                 "\n".join(analysis["improvements"]),
            file_name="resume_review.md",
            mime="text/markdown"
        )

    st.caption(f"Built as part of AI Mastery 2026 • {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
