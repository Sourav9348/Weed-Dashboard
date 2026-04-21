"""Shared helpers: data loading, theming, and reusable UI components."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Consistent species color map used everywhere in the app.
SPECIES_COLORS = {
    "Cocklebur": "#E53935",         # red
    "Foxtail": "#1E88E5",           # blue
    "Redroot_Pigweed": "#43A047",   # green
    "Giant_Ragweed": "#FB8C00",     # orange
}

# Architecture family -> color (used across benchmark plots)
FAMILY_COLORS = {
    "YOLOv5": "#5B8DEF",
    "YOLOv8": "#2E7D32",
    "YOLOv9": "#AD1457",
    "YOLOv10": "#6A1B9A",
    "YOLO11": "#EF6C00",
    "YOLO12": "#00838F",
    "YOLO26": "#546E7A",
}


def get_family(model_name: str) -> str:
    """Map 'YOLOv8n' -> 'YOLOv8'."""
    name = str(model_name)
    for fam in FAMILY_COLORS:
        if name.startswith(fam):
            return fam
    return "Other"


def get_size_tier(model_name: str) -> str:
    """Map model suffix n/s/m/t to a size tier."""
    if not model_name:
        return "?"
    last = str(model_name)[-1].lower()
    return {"n": "nano", "t": "tiny", "s": "small", "m": "medium"}.get(last, "?")


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "benchmark_summary.csv")
    df["Family"] = df["Model"].apply(get_family)
    df["Size Tier"] = df["Model"].apply(get_size_tier)
    return df


@st.cache_data
def load_parameter_relations() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "parameter_relations.csv")
    df["Family"] = df["Model"].apply(get_family)
    df["Size Tier"] = df["Model"].apply(get_size_tier)
    return df


@st.cache_data
def load_sessions() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "simulated_sessions.csv")


@st.cache_data
def load_session_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "simulated_session_summary.csv")


@st.cache_data
def load_per_class() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "yolov8n_per_class.csv")


@st.cache_data
def load_confusion_matrix() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "yolov8n_confusion_matrix.csv", index_col=0)


# --- UI helpers ---
def page_header(title: str, subtitle: str, emoji: str = "🌿"):
    st.markdown(
        f"""
        <div style="
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
            color: white;
            margin-bottom: 1.2rem;
        ">
            <div style="font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem;">
                {emoji} {title}
            </div>
            <div style="font-size: 0.95rem; opacity: 0.92;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(col, label: str, value: str, delta: str | None = None, help: str | None = None):
    with col:
        if delta:
            st.metric(label, value, delta, help=help)
        else:
            st.metric(label, value, help=help)


def data_source_note(real: bool, note: str):
    """Clearly mark whether a chart uses real or simulated data."""
    if real:
        st.caption(f"📊 **Data source:** Real benchmark / deployment data. {note}")
    else:
        st.caption(f"🧪 **Data source:** Simulated session data (placeholder until IMU field data is collected). {note}")
