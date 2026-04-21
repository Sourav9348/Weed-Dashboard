"""Page 2 — Selected Model: YOLOv8n.
Per-class performance diagnostics: PR curve summary, confusion matrix, class support.
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
    load_per_class,
    load_confusion_matrix,
    SPECIES_COLORS,
    data_source_note,
)

st.set_page_config(page_title="YOLOv8n · Weed Edge", page_icon="🎯", layout="wide")

page_header(
    "Selected Model — YOLOv8n",
    "Per-class diagnostics on the 4Weed validation set (mAP@0.5 = 0.697)",
    emoji="🎯",
)

per_class = load_per_class()
cm = load_confusion_matrix()

# --- Why this model? ---
with st.expander("📘 Why was YOLOv8n selected?", expanded=False):
    st.markdown(
        """
        Of the 21 YOLO variants benchmarked, YOLOv8n was selected for Jetson deployment because it
        sits on the **accuracy–size–speed Pareto frontier**:

        - **3.01 M parameters / 6.3 MB** — fits comfortably on the Jetson Nano-class device.
        - **Test mAP50 = 0.814** — well above the 70% course target and competitive with larger models.
        - **Mature TensorRT export path** — Ultralytics officially supports ONNX → TensorRT conversion for v8.
        - **Lowest inference time** among nano-class models (6.93 ms per frame on A100).
        """
    )

# --- KPI strip ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall mAP@0.5 (val)", "0.697")
c2.metric("Best class", "Foxtail (AP 0.851)")
c3.metric("Weakest class", "Giant Ragweed (AP 0.410)", delta="Needs more data", delta_color="inverse")
c4.metric("Optimal conf. threshold", "0.347", help="Peak F1 = 0.72 at this threshold")

st.markdown("---")

# --- Per-class AP + sample count side by side ---
st.subheader("Per-class performance")

left, right = st.columns(2)

with left:
    fig_ap = px.bar(
        per_class,
        x="class", y="AP_0_5",
        color="class",
        color_discrete_map=SPECIES_COLORS,
        text=per_class["AP_0_5"].apply(lambda v: f"{v:.3f}"),
    )
    fig_ap.update_traces(textposition="outside")
    fig_ap.update_layout(
        title="Average Precision @ IoU 0.5",
        yaxis=dict(range=[0, 1], title="AP"),
        xaxis_title="",
        showlegend=False,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig_ap.add_hline(y=0.70, line_dash="dash", line_color="gray",
                     annotation_text="70% target", annotation_position="top right")
    st.plotly_chart(fig_ap, use_container_width=True)

with right:
    counts = per_class.melt(
        id_vars="class",
        value_vars=["train_count", "val_count", "test_count"],
        var_name="split", value_name="count",
    )
    counts["split"] = counts["split"].map({
        "train_count": "Train", "val_count": "Val", "test_count": "Test"
    })
    fig_support = px.bar(
        counts, x="class", y="count", color="split",
        color_discrete_map={"Train": "#2E7D32", "Val": "#1E88E5", "Test": "#E53935"},
        text="count",
    )
    fig_support.update_layout(
        title="Sample support per class (618 images total, 70/20/10 split)",
        xaxis_title="", yaxis_title="Images",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_support, use_container_width=True)

st.info(
    "🔍 **Insight:** Giant Ragweed's weak AP (0.410) despite having 108 training images suggests "
    "the problem isn't sample count but visual difficulty — it looks like several other broadleaf "
    "species. Future work: targeted hard-negative mining and color-shift augmentation.",
    icon="💡",
)

data_source_note(True, "Source: YOLOv8n validation metrics reported in Weekly Report 2.")

st.markdown("---")

# --- Confusion matrix ---
st.subheader("Confusion matrix (normalized)")

show_bg = st.checkbox("Include background class", value=True)
cm_plot = cm if show_bg else cm.drop(columns=["background"]).drop(index=["background"])

fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm_plot.values,
        x=cm_plot.columns,
        y=cm_plot.index,
        colorscale="Greens",
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in cm_plot.values],
        texttemplate="%{text}",
        textfont=dict(size=13),
        colorbar=dict(title="Rate"),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Rate: %{z:.2f}<extra></extra>",
    )
)
fig_cm.update_layout(
    xaxis_title="Predicted class",
    yaxis_title="True class",
    height=480,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown(
    """
    **How to read this matrix:**
    - Diagonal values are correct predictions (higher = better).
    - Row "Giant_Ragweed" shows only **0.45** on the diagonal and **0.40 → background** — meaning
      45% of Giant Ragweed instances were classified correctly, and 40% were missed entirely.
    - Foxtail and Redroot Pigweed are the cleanest classes (82–83% correct).
    """
)
