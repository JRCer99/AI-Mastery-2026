import random
import streamlit as st
from transformers import pipeline
from datetime import datetime

EXAMPLE_POOL = [
    "I absolutely love this product! Best purchase ever.",
    "This is the worst movie I've seen in years.",
    "The food was okay, nothing special.",
    "I'm extremely disappointed with the customer service.",
    "Incredible experience from start to finish — highly recommend!",
    "Waste of money. Broke after two days.",
    "Shipping was fast and the packaging was great.",
    "The app crashes constantly. Totally unusable.",
    "Not bad, but nothing I'd rave about.",
    "Staff were rude and unhelpful. Won't be returning.",
    "Best coffee I've ever had in my life.",
    "Mediocre at best. Expected way more for the price.",
    "Five stars — exceeded every expectation.",
    "I've had better experiences at a gas station.",
    "Pretty good for the price point.",
    "Absolutely terrible. Do not buy this.",
    "Warm, welcoming atmosphere and amazing food.",
    "The instructions were confusing and incomplete.",
    "Works exactly as described. Super happy with it.",
    "Total scam. Looked nothing like the pictures.",
]

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def main():
    st.set_page_config(page_title="JRC Sentiment Analyzer", page_icon="😊", layout="centered")
    st.title("😊 JRC Sentiment Analysis Web App")
    st.markdown("**Month 5 Project 2** — Analyze the sentiment of any text using a fine-tuned LLM.")
    st.write("Paste any text (reviews, tweets, comments, emails, etc.) and get instant sentiment analysis.")

    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""
    if "shown_examples" not in st.session_state:
        st.session_state["shown_examples"] = random.sample(EXAMPLE_POOL, 4)

    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here...",
        key="text_input",
    )

    col1, _ = st.columns([1, 3])
    with col1:
        analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    if analyze_button and text_input.strip():
        with st.spinner("Analyzing..."):
            classifier = load_sentiment_model()
            result = classifier(text_input[:512])[0]
            label = result["label"]
            score = result["score"]
            emoji = "😊" if label == "POSITIVE" else "😞"
            st.success(f"{emoji} **{label}**")
            st.metric("Confidence", f"{score:.1%}")
            st.write("**Your text:**")
            st.info(text_input)
            st.progress(score)

    st.subheader("Try these examples")
    for ex in st.session_state["shown_examples"]:
        col_text, col_btn = st.columns([5, 1])
        with col_text:
            st.code(ex, language=None)
        with col_btn:
            st.write("")  # vertical align
            st.button(
                "Use",
                key=f"ex_{ex[:30]}",
                use_container_width=True,
                on_click=lambda e=ex: st.session_state.update({"text_input": e}),
            )

    def refresh_examples():
        current = set(st.session_state["shown_examples"])
        pool = [e for e in EXAMPLE_POOL if e not in current]
        if len(pool) < 4:
            pool = EXAMPLE_POOL
        st.session_state["shown_examples"] = random.sample(pool, 4)

    st.button("🔀 Generate More Examples", use_container_width=True, on_click=refresh_examples)

    st.caption(f"Built as part of AI Mastery 2026 • {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
