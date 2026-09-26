"""Streamlit dashboard for real-time vehicle crossing analytics."""

from __future__ import annotations

from sqlalchemy import text

from config import load_settings
from db import EventRepository


def main() -> None:
    """Render a browser dashboard backed by PostgreSQL."""
    import pandas as pd
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh

    settings = load_settings()
    repository = EventRepository(settings.database_url)
    st.set_page_config(page_title="Vehicle Analytics", page_icon="🚦", layout="wide")
    st_autorefresh(interval=2_000, key="dashboard-refresh")
    st.title("🚦 Real-Time Vehicle Analytics")
    st.caption("Live line-crossing counts and recent detections · refreshes every two seconds")

    try:
        repository.initialize()
        with repository.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        total, colors, recent = repository.dashboard_data()
    except Exception as exc:
        st.error(f"Database unavailable: {exc}")
        st.info("Start PostgreSQL with `docker compose up -d postgres`, then refresh this page.")
        st.stop()

    metrics = st.columns(5)
    metrics[0].metric("All vehicles", total)
    for column, color in zip(metrics[1:], ("white", "black", "gray", "other")):
        if color == "other":
            value = sum(count for name, count in colors.items() if name not in {"white", "black", "gray"})
        else:
            value = colors.get(color, 0)
        column.metric(color.title(), value)

    chart, table = st.columns([1, 2])
    with chart:
        st.subheader("Color breakdown")
        if colors:
            st.bar_chart(pd.DataFrame.from_dict(colors, orient="index", columns=["Vehicles"]))
        else:
            st.caption("No vehicles have crossed the counting line yet.")
    with table:
        st.subheader("Recent crossings")
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
        else:
            st.caption("Events appear after the pipeline detects a line crossing.")

    with st.sidebar:
        st.header("Pipeline status")
        st.success("PostgreSQL connected")
        st.write(f"**Stored crossings:** {total}")
        st.code(settings.rtsp_url, language=None)
        st.caption("Run `python pipeline.py` to start inference.")


if __name__ == "__main__":
    main()
