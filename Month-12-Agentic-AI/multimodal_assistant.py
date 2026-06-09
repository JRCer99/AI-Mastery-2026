"""
Autonomous Multimodal Assistant — Month 12 Project 2
Agent that processes text + images to complete tasks.
Streamlit UI: upload image → ask questions → get grounded answers.

MOCK = True  → structured mock response describing image metadata (no API needed)
MOCK = False → real Claude vision API (set ANTHROPIC_API_KEY env var)
"""

import streamlit as st
import base64
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

MOCK = True  # Set False + export ANTHROPIC_API_KEY=... for real Claude vision

# ─── Vision Agent ────────────────────────────────────────────────────────────

def encode_image(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")

def analyze_image(image_bytes: bytes, image_name: str, query: str, img: Image.Image) -> str:
    if MOCK:
        w, h = img.size
        mode = img.mode
        size_kb = len(image_bytes) / 1024
        return (
            f"**[MOCK Vision Analysis]**\n\n"
            f"**Image:** `{image_name}` ({w}×{h}px, {mode}, {size_kb:.1f} KB)\n\n"
            f"**Your question:** {query}\n\n"
            f"**Mock response:** I can see the uploaded image ({w}×{h} pixels). "
            f"In a real deployment with `MOCK=False` and a valid `ANTHROPIC_API_KEY`, "
            f"Claude's vision model would analyze the actual image content and answer your question "
            f"with specific details about what's visible — objects, text, colors, layout, and context.\n\n"
            f"**What this agent can do (live mode):**\n"
            f"- Describe image content in detail\n"
            f"- Answer specific questions about objects, text, or scenes\n"
            f"- Extract text from screenshots or documents (OCR-style)\n"
            f"- Compare multiple images\n"
            f"- Generate captions or alt text\n\n"
            f"*Set `MOCK = False` + `export ANTHROPIC_API_KEY=...` to enable real vision.*"
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        b64 = encode_image(image_bytes)
        suffix = Path(image_name).suffix.lower()
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                      "gif": "image/gif", "webp": "image/webp"}.get(suffix.lstrip("."), "image/jpeg")

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": query},
                ],
            }],
        )
        return response.content[0].text
    except Exception as e:
        return f"[Vision API error] {e}\n\nSet `ANTHROPIC_API_KEY` and ensure `anthropic` is installed."


# ─── App ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Multimodal Assistant", page_icon="👁️", layout="wide")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("👁️ Multimodal Assistant")
        st.caption("🟡 MOCK mode" if MOCK else "🟢 Claude Vision API")
        st.divider()

        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "gif"])
        if uploaded:
            st.session_state.current_image = uploaded
            st.session_state.history = []

        st.divider()
        st.caption("**Capabilities (live mode)**")
        st.caption("• Describe image content")
        st.caption("• Answer questions about scenes")
        st.caption("• Extract text from screenshots")
        st.caption("• Identify objects and colors")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.caption("**To enable real vision:**")
        st.code("export ANTHROPIC_API_KEY=...\n# Then set MOCK = False", language="bash")

    # ── Main ─────────────────────────────────────────────────────────────────
    st.title("👁️ Autonomous Multimodal Assistant")
    st.caption("Upload an image and ask anything about it — powered by Claude Vision")

    if MOCK:
        st.info("MOCK mode active — set `MOCK = False` + `ANTHROPIC_API_KEY` to enable real vision.", icon="🟡")

    if st.session_state.current_image is None:
        st.markdown("### Upload an image to get started")
        st.markdown("Supported formats: JPG, PNG, WebP, GIF")
        st.divider()
        st.caption(f"Month 12 Project 2 — Multimodal Assistant | {datetime.now().strftime('%B %d, %Y')}")
        return

    col_img, col_chat = st.columns([1, 1.5])

    with col_img:
        st.subheader("📸 Image")
        img_file = st.session_state.current_image
        img_bytes = img_file.read()
        img_file.seek(0)
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, use_container_width=True)
        st.caption(f"`{img_file.name}` — {img.size[0]}×{img.size[1]}px, {img.mode}, {len(img_bytes)/1024:.1f} KB")

    with col_chat:
        st.subheader("💬 Ask about this image")

        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("What do you see? What's in this image?")
        if query:
            st.session_state.history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    img_file.seek(0)
                    fresh_bytes = img_file.read()
                    answer = analyze_image(fresh_bytes, img_file.name, query, img)
                st.markdown(answer)
            st.session_state.history.append({"role": "assistant", "content": answer})

        if not st.session_state.history:
            st.markdown("**Try asking:**")
            for hint in ["What is in this image?", "Describe the colors and composition.",
                         "Is there any text visible?", "What mood does this image convey?"]:
                st.markdown(f"- *{hint}*")

    st.divider()
    st.caption(f"Month 12 Project 2 — Multimodal Assistant | {datetime.now().strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()
