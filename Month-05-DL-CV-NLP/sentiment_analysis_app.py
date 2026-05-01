import streamlit as st
from transformers import pipeline
from datetime import datetime

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def main():
    st.set_page_config(page_title="JRC Sentiment Analyzer", page_icon="😊", layout="centered")
    st.title("😊 JRC Sentiment Analysis Web App")
    st.markdown("**Month 5 Project 2** — Analyze the sentiment of any text using a fine-tuned LLM.")
    st.write("Paste any text (reviews, tweets, comments, emails, etc.) and get instant sentiment analysis.")

    text_input = st.text_area("Enter text to analyze:", height=150, placeholder="Type or paste your text here...")

    col1, _ = st.columns([1, 3])
    with col1:
        analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_button and text_input.strip():
        with st.spinner("Analyzing..."):
            classifier = load_sentiment_model()
            result = classifier(text_input[:512])[0]
            label = result['label']
            score = result['score']
            emoji = "😊" if label == "POSITIVE" else "😞"
            st.success(f"{emoji} **{label}**")
            st.metric("Confidence", f"{score:.1%}")
            st.write("**Your text:**")
            st.info(text_input)
            st.progress(score)

    st.subheader("Try these examples")
    examples = [
        "I absolutely love this product! Best purchase ever.",
        "This is the worst movie I've seen in years.",
        "The food was okay, nothing special.",
        "I'm extremely disappointed with the customer service."
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.text_input = ex

    st.caption(f"Built as part of AI Mastery 2026 • {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
