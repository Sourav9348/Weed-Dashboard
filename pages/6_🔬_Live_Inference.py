"""Page 6 — Live Inference.
Run YOLOv8n on user-uploaded images, videos, or browser webcam snapshots.
Falls back to base YOLOv8n (COCO) if custom weights are not provided.
"""
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import page_header  # noqa: E402

st.set_page_config(page_title="Live Inference · Weed Edge", page_icon="🔬", layout="wide")

page_header(
    "Live Inference",
    "Run YOLOv8n detection on uploaded images, videos, or live camera snapshots",
    emoji="🔬",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
SAMPLE_IMG_DIR = ROOT / "data" / "sample_images"
SAMPLE_VID_DIR = ROOT / "data" / "sample_videos"

for d in (MODEL_DIR, SAMPLE_IMG_DIR, SAMPLE_VID_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Model loader (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    """Load custom 4Weed model if `models/best.pt` exists; otherwise base YOLOv8n."""
    try:
        from ultralytics import YOLO  # noqa: WPS433
    except ImportError as e:
        return None, f"ultralytics not installed: {e}", False

    custom_path = MODEL_DIR / "best.pt"
    if custom_path.exists():
        try:
            m = YOLO(str(custom_path))
            return m, f"Custom 4Weed model loaded from `models/best.pt`", True
        except Exception as e:
            return None, f"Failed to load custom weights: {e}", False

    # Fallback: base YOLOv8n (auto-downloaded on first use, ~6 MB)
    try:
        m = YOLO("yolov8n.pt")
        msg = (
            "Base YOLOv8n (COCO) loaded — this detects people, cars, animals, etc., "
            "*not* weeds. To enable weed detection, place your trained `best.pt` in "
            "the `models/` folder and restart."
        )
        return m, msg, False
    except Exception as e:
        return None, f"Could not load any YOLO model: {e}", False


model, model_status, is_custom = load_model()

if model is None:
    st.error(model_status)
    st.info(
        "Add `ultralytics>=8.1` and `opencv-python-headless` to `requirements.txt` "
        "and redeploy, or run `pip install -r requirements.txt` locally."
    )
    st.stop()

# Status banner
if is_custom:
    st.success(f"🧠 {model_status}")
else:
    st.warning(f"⚠️ {model_status}")

# ---------------------------------------------------------------------------
# Shared inference controls
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ Inference settings")
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
    help="Lower = more detections (and more false positives).",
)
iou_thresh = st.sidebar.slider(
    "NMS IoU threshold", 0.10, 0.90, 0.45, 0.05,
    help="Higher = allow more overlapping boxes to survive non-maximum suppression.",
)
img_size = st.sidebar.select_slider(
    "Inference image size", options=[320, 416, 512, 640, 800], value=640,
    help="Bigger = more accurate, slower. 640 matches the training resolution.",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image", "🎥 Video", "📷 Camera"])


def _predict(img_bgr_or_rgb: np.ndarray, is_rgb: bool = True):
    """Run YOLO and return (annotated_rgb, results_obj)."""
    results = model.predict(
        img_bgr_or_rgb,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=img_size,
        verbose=False,
    )
    r = results[0]
    plotted_bgr = r.plot()  # Ultralytics returns BGR
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    return plotted_rgb, r


def _detections_to_df(r) -> pd.DataFrame:
    """Turn a YOLO Results object into a tidy DataFrame."""
    if len(r.boxes) == 0:
        return pd.DataFrame(columns=["#", "class", "confidence", "x1", "y1", "x2", "y2"])
    rows = []
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        cls_name = r.names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        rows.append({
            "#": i + 1,
            "class": cls_name,
            "confidence": round(conf, 3),
            "x1": round(xyxy[0], 1), "y1": round(xyxy[1], 1),
            "x2": round(xyxy[2], 1), "y2": round(xyxy[3], 1),
        })
    return pd.DataFrame(rows)


# ============================================================================
# TAB 1 — IMAGE
# ============================================================================
with tab_img:
    st.subheader("Image detection")

    mode = st.radio(
        "Input source", ["Upload your own", "Use a sample from the repo"],
        horizontal=True, key="img_mode",
    )

    pil_img = None
    caption = ""

    if mode == "Upload your own":
        up = st.file_uploader(
            "Drop an image", type=list(e.lstrip(".") for e in IMG_EXTS),
            help="JPG / PNG / BMP / WEBP, up to ~200 MB.",
            key="img_upload",
        )
        if up is not None:
            pil_img = Image.open(up).convert("RGB")
            caption = up.name
    else:
        samples = sorted(f for f in SAMPLE_IMG_DIR.iterdir() if f.suffix.lower() in IMG_EXTS)
        if samples:
            pick = st.selectbox(
                "Pick a sample image",
                options=[f.name for f in samples],
                key="img_sample",
            )
            chosen = SAMPLE_IMG_DIR / pick
            pil_img = Image.open(chosen).convert("RGB")
            caption = pick
        else:
            st.info(
                f"📁 No sample images yet. Drop JPG/PNG files into "
                f"`data/sample_images/` (relative to the app root) and they'll appear here "
                f"automatically."
            )

    if pil_img is not None:
        if st.button("🚀 Run inference", type="primary", key="img_run"):
            with st.spinner("Running YOLOv8n…"):
                annotated, r = _predict(np.array(pil_img))
            c1, c2 = st.columns(2)
            with c1:
                st.image(pil_img, caption=f"Input · {caption}", use_container_width=True)
            with c2:
                st.image(
                    annotated,
                    caption=f"Detections · {len(r.boxes)} boxes",
                    use_container_width=True,
                )

            df = _detections_to_df(r)
            if len(df):
                st.subheader(f"Detection summary ({len(df)} boxes)")
                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "confidence": st.column_config.ProgressColumn(
                            "confidence", min_value=0, max_value=1, format="%.3f",
                        )
                    },
                )

                buf = io.BytesIO()
                Image.fromarray(annotated).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download annotated PNG",
                    data=buf.getvalue(),
                    file_name=f"annotated_{Path(caption).stem or 'image'}.png",
                    mime="image/png",
                )
            else:
                st.info(
                    "No detections above the current confidence threshold. "
                    "Try lowering it in the sidebar."
                )


# ============================================================================
# TAB 2 — VIDEO
# ============================================================================
with tab_vid:
    st.subheader("Video detection")
    st.caption(
        "⚠️ On Streamlit Cloud (CPU-only), inference runs at ~2–5 FPS. "
        "This tab samples up to 90 frames from the video for a quick demo. "
        "On the Jetson with TensorRT FP16, the same model runs at **49 FPS** on the full stream."
    )

    v_mode = st.radio(
        "Input source", ["Upload your own", "Use a sample from the repo"],
        horizontal=True, key="vid_mode",
    )

    active_video = None
    v_label = ""

    if v_mode == "Upload your own":
        v_up = st.file_uploader(
            "Drop a video", type=list(e.lstrip(".") for e in VID_EXTS),
            help="MP4 / AVI / MOV / MKV / WEBM. Keep it short (< 30 s) for a timely demo.",
            key="vid_upload",
        )
        if v_up is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(v_up.name).suffix)
            tmp.write(v_up.read())
            tmp.flush()
            active_video = tmp.name
            v_label = v_up.name
    else:
        v_samples = sorted(f for f in SAMPLE_VID_DIR.iterdir() if f.suffix.lower() in VID_EXTS)
        if v_samples:
            v_pick = st.selectbox(
                "Pick a sample video",
                options=[f.name for f in v_samples],
                key="vid_sample",
            )
            active_video = str(SAMPLE_VID_DIR / v_pick)
            v_label = v_pick
        else:
            st.info(
                "📁 No sample videos yet. Drop MP4 / MOV / AVI files into "
                "`data/sample_videos/` to make them selectable here."
            )

    if active_video is not None:
        max_frames = st.slider(
            "Max frames to process",
            10, 150, 60, 10,
            help="Frames are sampled evenly across the video.",
            key="vid_max",
        )

        if st.button("🚀 Run inference on video", type="primary", key="vid_run"):
            cap = cv2.VideoCapture(active_video)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration = total / src_fps if src_fps else 0
            st.write(
                f"**Source:** `{v_label}` · {total} frames @ {src_fps:.1f} FPS "
                f"({duration:.1f} s)"
            )

            step = max(1, total // max_frames)
            frame_slot = st.empty()
            progress = st.progress(0.0)
            metric_slot = st.empty()

            detections_per_frame = []
            classes_per_frame = []
            idx = 0
            processed = 0

            while processed < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, r = _predict(frame)
                frame_slot.image(
                    annotated,
                    caption=f"Frame {idx} / {total} · {len(r.boxes)} detections",
                    use_container_width=True,
                )
                detections_per_frame.append(len(r.boxes))
                classes_per_frame.append(
                    [r.names.get(int(b.cls[0]), "?") for b in r.boxes]
                )
                processed += 1
                progress.progress(processed / max_frames)
                idx += step
                if idx >= total:
                    break
            cap.release()

            if detections_per_frame:
                total_det = sum(detections_per_frame)
                avg_det = float(np.mean(detections_per_frame))
                metric_slot.success(
                    f"✅ Processed **{processed}** sampled frames · "
                    f"Total detections: **{total_det}** · "
                    f"Avg per frame: **{avg_det:.2f}**"
                )

                # Detections over time
                df_timeline = pd.DataFrame({
                    "sampled_frame": list(range(processed)),
                    "detections": detections_per_frame,
                })
                st.line_chart(df_timeline, x="sampled_frame", y="detections", height=220)

                # Class frequency
                flat = [c for row in classes_per_frame for c in row]
                if flat:
                    cls_counts = pd.Series(flat).value_counts().reset_index()
                    cls_counts.columns = ["class", "count"]
                    st.subheader("Class frequency across sampled frames")
                    st.bar_chart(cls_counts, x="class", y="count", height=260)


# ============================================================================
# TAB 3 — CAMERA
# ============================================================================
with tab_cam:
    st.subheader("Camera detection")

    st.info(
        "**Two modes, depending on where the dashboard is running:**\n\n"
        "- 💻 **Running on your laptop or on Streamlit Cloud** → the widget below captures a "
        "single snapshot from your browser's webcam and runs inference. Great for a quick "
        "spot check during a demo.\n"
        "- 🔌 **Running on the Jetson with a CSI/USB camera attached** → the production "
        "pipeline uses a GStreamer input feeding frames directly to the TensorRT engine at "
        "**49 FPS**. That's a separate script (`05_TRT_Realtime_Detection.py`) — the "
        "recorded demo is on the *Jetson Deployment* page.",
        icon="📷",
    )

    photo = st.camera_input("Take a snapshot with your webcam")

    if photo is not None:
        pil = Image.open(photo).convert("RGB")
        with st.spinner("Running inference…"):
            annotated, r = _predict(np.array(pil))

        c1, c2 = st.columns(2)
        with c1:
            st.image(pil, caption="Snapshot", use_container_width=True)
        with c2:
            st.image(
                annotated,
                caption=f"{len(r.boxes)} detections",
                use_container_width=True,
            )

        df = _detections_to_df(r)
        if len(df):
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No detections in this snapshot.")

    st.markdown("---")
    st.subheader("On-Jetson real-time pipeline (reference code)")
    st.caption(
        "This is the script that runs on the physical device. It reads directly from the "
        "CSI camera via GStreamer and uses the TensorRT engine instead of the PyTorch model."
    )
    st.code(
        '''# 05_TRT_Realtime_Detection.py  (runs ON the Jetson, not in this dashboard)
from ultralytics import YOLO
import cv2

# Load the compiled TensorRT FP16 engine (49 FPS on Jetson)
model = YOLO("best.engine", task="detect")

# GStreamer pipeline for the Jetson's CSI camera
gst_pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
    "width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Jetson CSI · YOLOv8n TensorRT", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
''',
        language="python",
    )
