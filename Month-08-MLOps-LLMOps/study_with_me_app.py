import streamlit as st
import sqlite3
import pandas as pd
import time
from datetime import datetime, date, timedelta
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS streak (
            id INTEGER PRIMARY KEY,
            current_streak INTEGER DEFAULT 0,
            last_date TEXT
        )
    """)
    if conn.execute("SELECT COUNT(*) FROM streak").fetchone()[0] == 0:
        conn.execute("INSERT INTO streak (id, current_streak, last_date) VALUES (1, 0, NULL)")
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


def get_streak() -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT current_streak, last_date FROM streak WHERE id=1").fetchone()
    conn.close()
    streak, last_date = row
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    if last_date in (today, yesterday):
        return streak
    return 0


def update_streak():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT current_streak, last_date FROM streak WHERE id=1").fetchone()
    streak, last_date = row
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    if last_date == today:
        pass
    elif last_date == yesterday:
        streak += 1
        conn.execute("UPDATE streak SET current_streak=?, last_date=? WHERE id=1", (streak, today))
    else:
        streak = 1
        conn.execute("UPDATE streak SET current_streak=1, last_date=? WHERE id=1", (today,))
    conn.commit()
    conn.close()


def main():
    st.set_page_config(page_title="Study With Me", page_icon="⏱️", layout="wide")
    init_db()

    for key, val in [("timer_running", False), ("timer_start", None),
                     ("timer_duration", 0), ("timer_subject", ""), ("timer_notes", "")]:
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("⏱️ JRC Study With Me")
    st.markdown("**Focus. Track. Improve.** — AI Mastery 2026")
    st.metric("Current Streak 🔥", f"{get_streak()} days")

    st.divider()

    st.subheader("🎯 Start a Study Session")
    col1, col2, col3 = st.columns(3)
    with col1:
        subject = st.selectbox("Subject", [
            "Python", "Math/Stats", "Machine Learning", "Deep Learning",
            "Transformers", "MLOps", "LLM Agents", "RAG", "Other"
        ])
    with col2:
        duration = st.slider("Duration (minutes)", min_value=5, max_value=180, value=25, step=5)
    with col3:
        notes = st.text_input("Notes (optional)", placeholder="e.g. Finished Chapter 3")

    if not st.session_state.timer_running:
        if st.button("▶ Start Timer", type="primary", use_container_width=True):
            st.session_state.timer_running = True
            st.session_state.timer_start = time.time()
            st.session_state.timer_duration = duration * 60
            st.session_state.timer_subject = subject
            st.session_state.timer_notes = notes
            st.rerun()
    else:
        elapsed = time.time() - st.session_state.timer_start
        remaining = st.session_state.timer_duration - elapsed

        if remaining <= 0:
            st.success("✅ Session Complete! Great work!")
            log_session(
                st.session_state.timer_subject,
                st.session_state.timer_duration // 60,
                st.session_state.timer_notes
            )
            update_streak()
            st.session_state.timer_running = False
            st.rerun()
        else:
            mins, secs = divmod(int(remaining), 60)
            st.info(f"⏳ **{st.session_state.timer_subject}** — Time left: {mins:02d}:{secs:02d}")
            if st.button("⏹ Stop Session", use_container_width=True):
                log_session(
                    st.session_state.timer_subject,
                    max(1, int(elapsed // 60)),
                    st.session_state.timer_notes
                )
                update_streak()
                st.session_state.timer_running = False
                st.rerun()
            time.sleep(1)
            st.rerun()

    st.divider()

    df = get_sessions()
    if df.empty:
        st.info("No sessions logged yet — start your first timer!")
        return

    df['date'] = pd.to_datetime(df['date'])
    df['duration_hours'] = df['duration_minutes'] / 60

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Hours", f"{df['duration_hours'].sum():.1f}h")
    c2.metric("Sessions", len(df))
    c3.metric("Today", f"{df[df['date'] == pd.Timestamp(date.today())]['duration_minutes'].sum()} min")
    c4.metric("Days Active", df['date'].dt.date.nunique())

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📊 Daily Study Hours")
        st.bar_chart(df.groupby('date')['duration_hours'].sum())
    with col_b:
        st.subheader("🧠 Time by Subject")
        st.bar_chart(df.groupby('subject')['duration_hours'].sum().sort_values(ascending=False))

    st.subheader("📋 Recent Sessions")
    display = df[['date', 'subject', 'duration_minutes', 'notes']].head(10).copy()
    display.columns = ['Date', 'Subject', 'Minutes', 'Notes']
    st.dataframe(display, use_container_width=True)

    st.caption(f"AI Mastery 2026 • {datetime.now().strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()
