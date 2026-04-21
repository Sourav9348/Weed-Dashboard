"""Page 3 — Jetson Deployment.
Compares PyTorch vs TensorRT FP16 performance on the NVIDIA Jetson reComputer.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import page_header, data_source_note

st.set_page_config(page_title="Jetson Deployment · Weed Edge", page_icon="⚡", layout="wide")

page_header(
    "Jetson TensorRT Deployment",
    "YOLOv8n exported to FP16 TensorRT and benchmarked on NVIDIA Jetson reComputer",
    emoji="⚡",
)

# --- Real measured numbers from Weekly Report 2 ---
backends = pd.DataFrame({
    "Backend": ["PyTorch", "TensorRT (FP16)"],
    "FPS": [23.11, 49.00],
    "ms_per_frame": [43.26, 20.41],
    "Meets 5 FPS target": ["✅ Yes", "✅ Yes"],
})

# --- KPI row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("TensorRT FPS", "49.0", delta="+25.9 vs PyTorch")
c2.metric("TensorRT latency", "20.4 ms", delta="-22.8 ms", delta_color="inverse")
c3.metric("Speedup", "2.12×", delta="112% faster")
c4.metric("Target (≥5 FPS)", "✅ Met", help="Course success criterion for real-time edge inference")

st.markdown("---")

# --- Side-by-side comparison ---
left, right = st.columns(2)

with left:
    st.subheader("Throughput (FPS)")
    fig_fps = px.bar(
        backends, x="Backend", y="FPS",
        color="Backend",
        color_discrete_map={"PyTorch": "#EE6666", "TensorRT (FP16)": "#2E7D32"},
        text="FPS",
    )
    fig_fps.update_traces(texttemplate="%{text:.2f} FPS", textposition="outside")
    fig_fps.update_layout(
        yaxis=dict(title="Frames per second", range=[0, 60]),
        xaxis_title="", showlegend=False, height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig_fps.add_hline(y=5, line_dash="dash", line_color="gray",
                      annotation_text="5 FPS target", annotation_position="bottom right")
    st.plotly_chart(fig_fps, use_container_width=True)

with right:
    st.subheader("Latency (ms per frame)")
    fig_ms = px.bar(
        backends, x="Backend", y="ms_per_frame",
        color="Backend",
        color_discrete_map={"PyTorch": "#EE6666", "TensorRT (FP16)": "#2E7D32"},
        text="ms_per_frame",
    )
    fig_ms.update_traces(texttemplate="%{text:.2f} ms", textposition="outside")
    fig_ms.update_layout(
        yaxis=dict(title="Milliseconds per frame", range=[0, 55]),
        xaxis_title="", showlegend=False, height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_ms, use_container_width=True)

data_source_note(True, "Source: 100-frame Jetson benchmark (10-frame warm-up) — `jetson_benchmark_report.txt`.")

st.markdown("---")

# --- Latency distribution (simulated around the real mean) ---
st.subheader("Latency distribution across 1000 frames")

st.caption(
    "🧪 Simulated distribution around the measured means (σ ≈ 2.5 ms PyTorch, σ ≈ 1.2 ms TensorRT) "
    "to illustrate jitter characteristics. Real percentiles will replace this after a full session."
)

np.random.seed(7)
pt_lat = np.random.normal(43.26, 2.5, 1000).clip(38, 55)
trt_lat = np.random.normal(20.41, 1.2, 1000).clip(15, 28)
dist_df = pd.DataFrame({
    "latency_ms": np.concatenate([pt_lat, trt_lat]),
    "backend": ["PyTorch"] * 1000 + ["TensorRT (FP16)"] * 1000,
})

fig_dist = px.histogram(
    dist_df, x="latency_ms", color="backend", nbins=60,
    color_discrete_map={"PyTorch": "#EE6666", "TensorRT (FP16)": "#2E7D32"},
    opacity=0.75, barmode="overlay",
)
fig_dist.update_layout(
    xaxis_title="Inference latency (ms)",
    yaxis_title="Frame count",
    height=400, legend_title="",
    margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig_dist, use_container_width=True)

colA, colB = st.columns(2)
with colA:
    st.markdown("**PyTorch (FP32)**")
    st.markdown(f"- p50: **43.3 ms** → 23.1 FPS\n- p95: **47.4 ms** → 21.1 FPS\n- Model weights: 6.3 MB, unoptimized eager mode")
with colB:
    st.markdown("**TensorRT (FP16)**")
    st.markdown(f"- p50: **20.4 ms** → 49.0 FPS\n- p95: **22.4 ms** → 44.6 FPS\n- Kernel-fused, half-precision engine")

st.markdown("---")

# --- Deployment pipeline diagram ---
st.subheader("Deployment pipeline")
st.markdown(
    """
    ```
    Colab A100           ONNX Runtime          Jetson reComputer (on-device)
    ───────────          ────────────          ──────────────────────────────
    YOLOv8n.pt    ─▶    best.onnx      ─▶    trtexec --fp16     ─▶    best.engine
    (PyTorch)                                 (TensorRT build)         (49 FPS inference)
    ```
    """
)

with st.expander("🔧 Actual conversion command used"):
    st.code(
        "trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --workspace=4096",
        language="bash",
    )
    st.caption(
        "FP16 halves memory traffic and lets the Jetson's Tensor Cores kick in, "
        "which is where most of the 2.12× speedup comes from."
    )
