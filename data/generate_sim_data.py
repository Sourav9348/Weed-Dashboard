"""
Generate realistic simulated IMU + detection session data.
Run once to produce CSVs that ship with the dashboard.
These are clearly labeled SIMULATED so they can be replaced with real field data later.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT = Path(__file__).parent

CLASSES = ["Cocklebur", "Foxtail", "Redroot_Pigweed", "Giant_Ragweed"]
CLASS_BASE_CONF = {"Cocklebur": 0.82, "Foxtail": 0.88, "Redroot_Pigweed": 0.80, "Giant_Ragweed": 0.62}


def synth_session(session_id: str, duration_s: int = 120, fps: int = 30):
    """One scouting session. Emits per-frame rows with IMU + detection."""
    n = duration_s * fps
    t = np.arange(n) / fps

    # --- IMU signal: accel (m/s^2) and gyro (rad/s) ---
    # Walking creates a ~1.5-2.5 Hz step frequency; hand tremor adds ~8-12 Hz.
    step_hz = np.random.uniform(1.8, 2.3)
    ax = 0.35 * np.sin(2 * np.pi * step_hz * t) + 0.15 * np.random.randn(n)
    ay = 0.25 * np.sin(2 * np.pi * step_hz * t + 0.7) + 0.15 * np.random.randn(n)
    az = 9.81 + 0.45 * np.sin(2 * np.pi * step_hz * t + 1.2) + 0.20 * np.random.randn(n)

    gx = 0.15 * np.sin(2 * np.pi * step_hz * t + 0.3) + 0.08 * np.random.randn(n)
    gy = 0.12 * np.sin(2 * np.pi * step_hz * t + 1.8) + 0.08 * np.random.randn(n)
    gz = 0.10 * np.sin(2 * np.pi * step_hz * t + 2.5) + 0.06 * np.random.randn(n)

    # Inject occasional "shake events" (rapid hand jerks)
    n_shake = np.random.randint(4, 9)
    for _ in range(n_shake):
        start = np.random.randint(0, n - fps)
        dur = np.random.randint(fps // 2, fps * 2)
        end = min(start + dur, n)
        ax[start:end] += np.random.randn(end - start) * 1.2
        ay[start:end] += np.random.randn(end - start) * 1.2
        gx[start:end] += np.random.randn(end - start) * 0.8
        gy[start:end] += np.random.randn(end - start) * 0.8

    # --- Stability score (the thing the filter uses) ---
    # Rolling std of gyro magnitude over a 0.5s window, then map to [0, 1].
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    window = max(1, fps // 2)
    gyro_std = pd.Series(gyro_mag).rolling(window, min_periods=1).std().fillna(0).to_numpy()
    # Lower std = more stable. Invert & normalize.
    stab = 1.0 - (gyro_std - gyro_std.min()) / (gyro_std.max() - gyro_std.min() + 1e-9)
    stability = np.clip(stab, 0, 1)

    # --- Detections: not every frame detects. Prob scales with stability ---
    # When stable: ~70% of frames see a weed. When shaking: ~25%.
    det_prob = 0.25 + 0.5 * stability
    has_det = np.random.rand(n) < det_prob

    weed_class = np.full(n, "", dtype=object)
    confidence = np.zeros(n)
    n_boxes = np.zeros(n, dtype=int)
    for i in range(n):
        if has_det[i]:
            cls = np.random.choice(CLASSES, p=[0.28, 0.25, 0.28, 0.19])
            weed_class[i] = cls
            # Stable frames -> confidence ~ base + noise; shaky -> degraded
            base = CLASS_BASE_CONF[cls]
            confidence[i] = np.clip(
                base * (0.55 + 0.45 * stability[i]) + np.random.normal(0, 0.04),
                0.05, 0.99
            )
            n_boxes[i] = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.4, 0.2, 0.15, 0.15, 0.07, 0.03])

    # --- Inference latency (ms): tight around TensorRT spec (~20.4ms) ---
    infer_ms = np.random.normal(20.4, 1.2, n).clip(15, 30)
    fps_inst = 1000.0 / infer_ms

    df = pd.DataFrame({
        "session_id": session_id,
        "time_s": np.round(t, 3),
        "accel_x": np.round(ax, 4),
        "accel_y": np.round(ay, 4),
        "accel_z": np.round(az, 4),
        "gyro_x": np.round(gx, 4),
        "gyro_y": np.round(gy, 4),
        "gyro_z": np.round(gz, 4),
        "stability": np.round(stability, 4),
        "weed_class": weed_class,
        "confidence": np.round(confidence, 4),
        "n_boxes": n_boxes,
        "infer_ms": np.round(infer_ms, 2),
        "fps_inst": np.round(fps_inst, 2),
    })
    return df


def main():
    sessions = []
    for i, (sid, dur) in enumerate([
        ("SESSION_001_NorthField", 90),
        ("SESSION_002_SouthPlot", 120),
        ("SESSION_003_Greenhouse", 75),
    ]):
        df = synth_session(sid, duration_s=dur)
        sessions.append(df)

    full = pd.concat(sessions, ignore_index=True)
    full.to_csv(OUT / "simulated_sessions.csv", index=False)

    # Session-level summary
    summary_rows = []
    for sid, grp in full.groupby("session_id"):
        det = grp[grp["weed_class"] != ""]
        # "Stable" = top 50% stability quantile
        stable_thresh = grp["stability"].median()
        stable_det = det[det["stability"] >= stable_thresh]
        unstable_det = det[det["stability"] < stable_thresh]

        summary_rows.append({
            "session_id": sid,
            "duration_s": grp["time_s"].max(),
            "total_frames": len(grp),
            "total_detections": len(det),
            "avg_conf_stable": round(stable_det["confidence"].mean(), 4) if len(stable_det) else 0,
            "avg_conf_unstable": round(unstable_det["confidence"].mean(), 4) if len(unstable_det) else 0,
            "mean_fps": round(grp["fps_inst"].mean(), 2),
            "mean_stability": round(grp["stability"].mean(), 4),
            "cocklebur": int((det["weed_class"] == "Cocklebur").sum()),
            "foxtail": int((det["weed_class"] == "Foxtail").sum()),
            "redroot_pigweed": int((det["weed_class"] == "Redroot_Pigweed").sum()),
            "giant_ragweed": int((det["weed_class"] == "Giant_Ragweed").sum()),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "simulated_session_summary.csv", index=False)

    # YOLOv8n per-class table from the paper
    per_class = pd.DataFrame({
        "class": ["Cocklebur", "Foxtail", "Redroot_Pigweed", "Giant_Ragweed"],
        "AP_0_5": [0.785, 0.851, 0.744, 0.410],
        "correct_pct": [0.76, 0.82, 0.83, 0.45],
        "train_count": [107, 91, 126, 108],
        "val_count": [34, 27, 32, 31],
        "test_count": [18, 21, 12, 11],
    })
    per_class.to_csv(OUT / "yolov8n_per_class.csv", index=False)

    # Confusion matrix from the paper's report
    cm = pd.DataFrame(
        [[0.76, 0.04, 0.06, 0.02, 0.12],
         [0.03, 0.82, 0.05, 0.01, 0.09],
         [0.04, 0.03, 0.83, 0.02, 0.08],
         [0.05, 0.04, 0.06, 0.45, 0.40],
         [0.05, 0.06, 0.04, 0.12, 0.73]],
        columns=["Cocklebur", "Foxtail", "Redroot_Pigweed", "Giant_Ragweed", "background"],
        index=["Cocklebur", "Foxtail", "Redroot_Pigweed", "Giant_Ragweed", "background"],
    )
    cm.to_csv(OUT / "yolov8n_confusion_matrix.csv")

    print("Wrote:")
    for f in sorted(OUT.glob("*.csv")):
        print(" ", f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
