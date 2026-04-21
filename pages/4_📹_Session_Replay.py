"""Page 4 — Live Session Replay.
Scrub through a simulated handheld scouting session with IMU + detection overlays.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    page_header, load_sessions, load_session_summary,
    SPECIES_COLORS, data_source_note,
)

st.set_page_config(page_title="Session Replay · Weed Edge", page_icon="📹", layout="wide")

page_header(
    "Live Session Replay",
    "Scrub through a handheld scouting session: IMU, detections, and confidence in sync",
    emoji="📹",
)

sessions_df = load_sessions()
summary = load_session_summary()

# --- Session picker ---
session_ids = sessions_df["session_id"].unique().tolist()

col_sel, col_range = st.columns([1, 2])
with col_sel:
    sid = st.selectbox(
        "Select session",
        session_ids,
        index=0,
        help="Each session is ~1.5–2 min of simulated walking with handheld Jetson.",
    )

sess = sessions_df[sessions_df["session_id"] == sid].reset_index(drop=True)
total_s = float(sess["time_s"].max())

with col_range:
    t_start, t_end = st.slider(
        "Time window (seconds)",
        0.0, float(np.ceil(total_s)),
        (0.0, float(np.ceil(total_s))),
        step=1.0,
        help="Drag the handles to zoom into a segment of the session.",
    )

window = sess[(sess["time_s"] >= t_start) & (sess["time_s"] <= t_end)].copy()

# --- Session-level KPIs ---
sum_row = summary[summary["session_id"] == sid].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Duration", f"{sum_row['duration_s']:.0f} s")
c2.metric("Total detections", f"{int(sum_row['total_detections']):,}")
c3.metric("Mean stability", f"{sum_row['mean_stability']:.2f}")
c4.metric("Mean FPS", f"{sum_row['mean_fps']:.1f}")
c5.metric(
    "Conf (stable vs unstable)",
    f"{sum_row['avg_conf_stable']:.3f}",
    delta=f"+{(sum_row['avg_conf_stable'] - sum_row['avg_conf_unstable']):.3f}",
    help="Avg detection confidence on stable frames vs unstable frames. Positive delta = filter hypothesis supported.",
)

st.markdown("---")

# --- Three synced time-series panels ---
st.subheader(f"Timeline — {sid}")

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("IMU stability (0 = shaky, 1 = rock-steady)",
                    "Detection confidence per frame",
                    "Inference speed (FPS)"),
    row_heights=[0.35, 0.40, 0.25],
)

# Stability band
fig.add_trace(
    go.Scatter(
        x=window["time_s"], y=window["stability"],
        mode="lines", line=dict(color="#1565C0", width=1.5),
        name="Stability", fill="tozeroy", fillcolor="rgba(21, 101, 192, 0.15)",
    ),
    row=1, col=1,
)

# Detections scatter colored by class
dets = window[window["weed_class"] != ""].copy()
for cls, color in SPECIES_COLORS.items():
    c = dets[dets["weed_class"] == cls]
    if len(c):
        fig.add_trace(
            go.Scatter(
                x=c["time_s"], y=c["confidence"],
                mode="markers",
                name=cls,
                marker=dict(color=color, size=5, opacity=0.75),
                hovertemplate="t=%{x:.1f}s<br>conf=%{y:.2f}<br>class=" + cls + "<extra></extra>",
            ),
            row=2, col=1,
        )

# FPS
fig.add_trace(
    go.Scatter(
        x=window["time_s"], y=window["fps_inst"],
        mode="lines", line=dict(color="#2E7D32", width=1),
        name="FPS", showlegend=False,
    ),
    row=3, col=1,
)
fig.add_hline(y=5, line_dash="dash", line_color="gray", row=3, col=1)

fig.update_yaxes(range=[0, 1.05], row=1, col=1)
fig.update_yaxes(range=[0, 1.0], title_text="Confidence", row=2, col=1)
fig.update_yaxes(title_text="FPS", row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(
    height=620,
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig, use_container_width=True)

data_source_note(
    False,
    f"Zoom shows {len(window):,} of {len(sess):,} frames. Stability is derived from "
    "rolling std of gyroscope magnitude over a 0.5 s window.",
)

st.markdown("---")

# --- Two-column summary: species breakdown + stability histogram ---
left, right = st.columns(2)

with left:
    st.subheader("Detections by species (in window)")
    if len(dets):
        counts = dets["weed_class"].value_counts().reset_index()
        counts.columns = ["class", "count"]
        fig_sp = px.bar(
            counts, x="class", y="count",
            color="class", color_discrete_map=SPECIES_COLORS,
            text="count",
        )
        fig_sp.update_layout(
            xaxis_title="", yaxis_title="Frames with detection",
            showlegend=False, height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sp, use_container_width=True)
    else:
        st.info("No detections in this window — zoom out.")

with right:
    st.subheader("Stability distribution (in window)")
    fig_stab = px.histogram(
        window, x="stability", nbins=30,
        color_discrete_sequence=["#1565C0"],
    )
    fig_stab.add_vline(
        x=window["stability"].median(), line_dash="dash", line_color="red",
        annotation_text=f"Median = {window['stability'].median():.2f}",
    )
    fig_stab.update_layout(
        xaxis_title="Stability score",
        yaxis_title="Frame count",
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_stab, use_container_width=True)

st.markdown("---")

# --- Frame-level inspector ---
st.subheader("🔬 Frame-level inspector")
st.caption("The raw per-frame log the Jetson writes to disk — this is what the dashboard ingests.")

show_detected_only = st.checkbox("Show only frames with a detection", value=True)
table = dets if show_detected_only else window
st.dataframe(
    table[["time_s", "weed_class", "confidence", "stability", "n_boxes", "infer_ms", "fps_inst"]]
    .head(300),
    use_container_width=True, hide_index=True,
    column_config={
        "confidence": st.column_config.ProgressColumn(
            "confidence", min_value=0, max_value=1, format="%.2f"
        ),
        "stability": st.column_config.ProgressColumn(
            "stability", min_value=0, max_value=1, format="%.2f"
        ),
    },
)
