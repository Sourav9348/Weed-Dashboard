"""
Weed Detection Edge Dashboard — Main Page
ASM 591: Agricultural Data Visualization & Edge Computing
Sourav Ranjan Mohapatra | Purdue University | Spring 2026
"""
import streamlit as st

from utils import (
    page_header,
    load_benchmark,
    load_session_summary,
)

st.set_page_config(
    page_title="Weed Detection Edge Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header(
    "Edge-Based Weed Detection Dashboard",
    "Real-time YOLOv8n inference on NVIDIA Jetson with IMU-assisted frame filtering",
    emoji="🌿",
)

# --- Headline KPI row ---
bench = load_benchmark()
sessions = load_session_summary()

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Models benchmarked",
    f"{len(bench)}",
    help="All official Ultralytics YOLO architectures tested on 4Weed",
)
col2.metric(
    "Best test mAP50",
    f"{bench['Test mAP50'].max():.3f}",
    delta=f"+{(bench['Test mAP50'].max() - 0.70):.2f} vs 70% target",
)
col3.metric(
    "Jetson TRT FPS",
    "49.0",
    delta="2.12× PyTorch",
    help="TensorRT FP16 engine on NVIDIA Jetson reComputer",
)
col4.metric(
    "Field sessions logged",
    f"{len(sessions)}",
    help="Simulated sessions for dashboard demo; replaced by field data in final phase",
)

st.markdown("---")

# --- Two-column project summary ---
left, right = st.columns([3, 2])

with left:
    st.subheader("📌 Project in one paragraph")
    st.markdown(
        """
        Traditional weed scouting means a person walking the field, squinting at every plant,
        and radioing numbers back. This project builds a **handheld, offline-capable**
        alternative: a camera and an IMU mounted to an **NVIDIA Jetson reComputer** running a
        TensorRT-optimized **YOLOv8n** model. The model detects and classifies four weed species
        in real time; the IMU filters out blurry/shaky frames before inference. The central
        research question is whether IMU-based frame filtering measurably improves detection
        reliability during manual scouting.
        """
    )

    st.subheader("🎯 Five objectives — current status")
    status_df = [
        ("1. Benchmark YOLO architectures", "Google Colab", "✅ Complete", 100),
        ("2. Optimize selected model via TensorRT", "Edge Device", "✅ Complete", 100),
        ("3. Real-time weed detection on Jetson", "Camera Sensor", "🟡 In Progress", 75),
        ("4. IMU-based frame stability filtering", "IMU Sensor", "⏳ Not Started", 0),
        ("5. Interactive visualization dashboard", "Application", "🟢 This app", 85),
    ]
    import pandas as pd
    status = pd.DataFrame(status_df, columns=["Objective", "Requirement", "Status", "Progress %"])
    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Progress %": st.column_config.ProgressColumn(
                "Progress", min_value=0, max_value=100, format="%d%%"
            )
        },
    )

with right:
    st.subheader("🧭 How to navigate this dashboard")
    st.markdown(
        """
        Use the sidebar to move between pages:

        - **1 · Model Benchmarking** — compare all 21 YOLO variants side-by-side.
        - **2 · Selected Model (YOLOv8n)** — per-class diagnostics: PR curve, confusion matrix.
        - **3 · Jetson Deployment** — PyTorch vs TensorRT performance on edge hardware.
        - **4 · Live Session Replay** — walk through a handheld scouting session with IMU
          stability, detection timeline, and weed counts.
        - **5 · IMU ↔ Confidence Analysis** — does a steadier hand give better predictions?
        """
    )

    st.info(
        "📊 **Real vs simulated data:** Benchmark and Jetson pages use real measurements "
        "from the project. Session & IMU pages use clearly-labeled simulated data as a "
        "placeholder until field collection is complete.",
        icon="ℹ️",
    )

    st.subheader("🔗 Quick links")
    st.markdown(
        """
        - [GitHub repository](https://github.com/Sourav9348/Weed-Dashboard)
        """
    )

st.markdown("---")

# --- Tech stack strip ---
st.subheader("🛠️ Tech stack")
stack_cols = st.columns(6)
tech = [
    ("🐍", "Python 3.11", "Core language"),
    ("🧠", "YOLOv8n", "Ultralytics"),
    ("⚡", "TensorRT", "FP16 engine"),
    ("🖥️", "NVIDIA Jetson", "reComputer"),
    ("📊", "Streamlit", "This dashboard"),
    ("📈", "Plotly", "Interactive charts"),
]
for c, (emoji, name, desc) in zip(stack_cols, tech):
    c.markdown(
        f"""
        <div style='text-align:center; padding:0.6rem; background:#F1F8E9;
                    border-radius:8px; min-height:110px;'>
            <div style='font-size:1.8rem;'>{emoji}</div>
            <div style='font-weight:600; margin-top:0.2rem;'>{name}</div>
            <div style='font-size:0.8rem; color:#5D6D7E;'>{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style='margin-top:2rem; padding:1rem; background:#F8F9FA;
                border-radius:8px; font-size:0.85rem; color:#566573;'>
    <b>ASM 591: Agricultural Data Visualization & Edge Computing · Spring 2026</b><br>
    Sourav Ranjan Mohapatra · sourav@purdue.edu · Purdue University
    </div>
    """,
    unsafe_allow_html=True,
)
