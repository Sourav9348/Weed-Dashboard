"""Page 1 — Benchmark Explorer.
Interactive view of all 21 YOLO models: filter by family/size, sort, and explore.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    page_header,
    load_benchmark,
    load_parameter_relations,
    FAMILY_COLORS,
    data_source_note,
)

st.set_page_config(page_title="Benchmarking · Weed Edge", page_icon="📊", layout="wide")

page_header(
    "Model Benchmarking",
    "21 YOLO models trained with identical hyperparameters on the 4Weed dataset",
    emoji="📊",
)

bench = load_benchmark()
params_df = load_parameter_relations()

# --- Sidebar filters ---
st.sidebar.header("🔎 Filters")
families = sorted(bench["Family"].unique())
picked_fams = st.sidebar.multiselect(
    "Architecture family", families, default=families,
    help="Toggle YOLO generations on/off.",
)
tiers = ["nano", "tiny", "small", "medium"]
picked_tiers = st.sidebar.multiselect(
    "Size tier", tiers, default=tiers,
    help="Nano/tiny = edge-friendly, medium = bigger & slower.",
)
min_map = st.sidebar.slider(
    "Min test mAP50", 0.70, 0.90, 0.70, 0.01,
    help="Filter to models above this accuracy threshold.",
)

filt = bench[
    bench["Family"].isin(picked_fams)
    & bench["Size Tier"].isin(picked_tiers)
    & (bench["Test mAP50"] >= min_map)
].copy()

# --- Top summary strip ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Models shown", f"{len(filt)} / {len(bench)}")
c2.metric("Best test mAP50", f"{filt['Test mAP50'].max():.3f}" if len(filt) else "—")
c3.metric("Smallest model (MB)", f"{filt['Size (MB)'].min():.1f}" if len(filt) else "—")
c4.metric("Fastest train (min)", f"{filt['Train time (min)'].min():.1f}" if len(filt) else "—")

st.markdown("---")

# --- Ranked bar chart ---
st.subheader("Test mAP50 — ranked")

if len(filt):
    ranked = filt.sort_values("Test mAP50", ascending=True)
    fig = px.bar(
        ranked,
        x="Test mAP50",
        y="Model",
        color="Family",
        color_discrete_map=FAMILY_COLORS,
        orientation="h",
        text=ranked["Test mAP50"].apply(lambda v: f"{v:.3f}"),
        hover_data={
            "Params (M)": ":.2f",
            "Size (MB)": ":.1f",
            "Test Precision": ":.3f",
            "Test Recall": ":.3f",
            "Test mAP50": False,
            "Model": False,
            "Family": False,
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(400, 28 * len(ranked)),
        xaxis=dict(range=[0.70, 0.90], title="Test mAP@0.50"),
        yaxis_title="",
        margin=dict(l=20, r=20, t=10, b=10),
    )
    fig.add_vline(x=0.80, line_dash="dash", line_color="gray",
                  annotation_text="80% target", annotation_position="top")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No models match these filters.")

data_source_note(True, "Source: `benchmark_summary.csv` — 21 models × 100 epochs on A100.")

st.markdown("---")

# --- Scatter: accuracy vs size ---
st.subheader("Accuracy vs. model size")

colA, colB = st.columns([3, 2])
with colA:
    fig2 = px.scatter(
        filt,
        x="Size (MB)",
        y="Test mAP50",
        color="Family",
        size="Params (M)",
        color_discrete_map=FAMILY_COLORS,
        hover_name="Model",
        hover_data={"Params (M)": ":.2f", "Size (MB)": ":.1f", "Family": False},
        log_x=True,
        size_max=28,
    )
    fig2.update_layout(
        height=430,
        xaxis_title="Model weight file size (MB, log scale)",
        yaxis_title="Test mAP@0.50",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

with colB:
    st.markdown("**Reading this chart**")
    st.markdown(
        """
        - **Bottom-left** = small *and* accurate → edge-friendly.
        - **Top-right** = large & accurate → best for server.
        - Bubble size = parameter count.
        - **YOLOv8n** (green nano) sits near the Pareto frontier at just 6.3 MB and 3.0 M params → why it was selected.
        """
    )

st.markdown("---")

# --- Full sortable table ---
st.subheader("Full benchmark table")
display_cols = [
    "Model", "Family", "Size Tier", "Params (M)", "Size (MB)",
    "Test mAP50", "Test mAP50-95", "Test Precision", "Test Recall",
    "Train time (min)",
]
st.dataframe(
    filt[display_cols].sort_values("Test mAP50", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Test mAP50": st.column_config.ProgressColumn(
            "Test mAP50", min_value=0.70, max_value=0.90, format="%.3f"
        ),
        "Test mAP50-95": st.column_config.ProgressColumn(
            "Test mAP50-95", min_value=0.40, max_value=0.50, format="%.3f"
        ),
    },
)

# Download button
csv = filt[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv,
    file_name="filtered_benchmark.csv",
    mime="text/csv",
)
