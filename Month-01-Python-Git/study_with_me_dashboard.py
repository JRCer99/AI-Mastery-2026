import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "study_sessions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            notes TEXT,
            date TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_session(subject: str, duration_minutes: int, notes: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (subject, duration_minutes, notes, date, created_at) VALUES (?, ?, ?, ?, ?)",
        (subject, duration_minutes, notes, date.today().isoformat(), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_sessions() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY created_at DESC", conn)
    conn.close()
    return df


def main():
    st.set_page_config(page_title="Study With Me", page_icon="📚", layout="wide")
    init_db()

    st.title("📚 Study With Me Dashboard")
    st.markdown("**Track your study sessions — AI Mastery 2026**")

    # --- Log a session ---
    st.subheader("⏱️ Log a Study Session")
    if "duration" not in st.session_state:
        st.session_state.duration = 30

    col1, col2, col3 = st.columns(3)
    with col1:
        subject = st.selectbox("Subject", [
            "Python", "Math/Stats", "Machine Learning", "Deep Learning",
            "Transformers", "MLOps", "LLM Agents", "RAG", "Other"
        ])
    with col2:
        duration = st.slider("Duration (minutes)", min_value=5, max_value=480,
                             value=st.session_state.duration, step=5)
        st.session_state.duration = duration
    with col3:
        notes = st.text_input("Notes (optional)", placeholder="e.g. Finished Chapter 3")

    if st.button("✅ Log Session", use_container_width=True):
        log_session(subject, duration, notes)
        st.success(f"Logged {duration} min of {subject}!")
        st.rerun()

    st.divider()

    # --- Stats ---
    df = get_sessions()

    if df.empty:
        st.info("No sessions logged yet. Start studying!")
        return

    df['date'] = pd.to_datetime(df['date'])
    df['duration_hours'] = df['duration_minutes'] / 60

    total_hours = df['duration_hours'].sum()
    total_sessions = len(df)
    today_minutes = df[df['date'] == pd.Timestamp(date.today())]['duration_minutes'].sum()
    streak = df['date'].dt.date.nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Hours", f"{total_hours:.1f}h")
    col2.metric("Sessions", total_sessions)
    col3.metric("Today", f"{today_minutes} min")
    col4.metric("Days Active", streak)

    st.subheader("📊 Daily Study Hours")
    daily = df.groupby('date')['duration_hours'].sum().reset_index()
    st.bar_chart(daily.set_index('date')['duration_hours'])

    st.subheader("🧠 Time by Subject")
    by_subject = df.groupby('subject')['duration_hours'].sum().sort_values(ascending=False)
    st.bar_chart(by_subject)

    st.subheader("📋 Recent Sessions")
    display = df[['date', 'subject', 'duration_minutes', 'notes']].head(10).copy()
    display.columns = ['Date', 'Subject', 'Minutes', 'Notes']
    st.dataframe(display, use_container_width=True)

    st.caption(f"AI Mastery 2026 • {datetime.now().strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()
