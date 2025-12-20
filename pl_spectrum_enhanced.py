
# # import streamlit as st
# # st.set_page_config(page_title="PL Ensemble System", layout="wide")

# # import pandas as pd
# # import os
# # import cv2
# # import numpy as np
# # import uuid
# # import tempfile
# # import joblib
# # from datetime import datetime
# # import matplotlib.pyplot as plt
# # from scipy.interpolate import make_interp_spline
# # from sqlalchemy import create_engine, text

# # # sklearn imports
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# # from sklearn.svm import SVR
# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.linear_model import Ridge
# # from sklearn.metrics import r2_score, mean_squared_error

# # # Optional XGBoost
# # try:
# #     from xgboost import XGBRegressor
# #     XG_AVAILABLE = True
# # except Exception:
# #     XG_AVAILABLE = False

# # # ======================
# # # CONFIG
# # # ======================
# # RESULTS_DB = "sqlite:///pl_results.db"
# # engine = create_engine(RESULTS_DB)
# # LOCAL_PLOTS_DIR = "saved_plots"
# # os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)
# # MODEL_PATH = "best_ensemble_model.joblib"
# # DEFAULT_ROI_SIZE = 50

# # # ======================
# # # AUTO DB INIT / FIXER
# # # ======================
# # def init_db():
# #     with engine.connect() as conn:
# #         conn.execute(text("""
# #             CREATE TABLE IF NOT EXISTS results (
# #                 id TEXT PRIMARY KEY,
# #                 timestamp TEXT,
# #                 sample_name TEXT,
# #                 voltage REAL,
# #                 avg_nm REAL,
# #                 peak_nm REAL,
# #                 min_nm REAL,
# #                 max_nm REAL,
# #                 plot_path TEXT,
# #                 ensemble_type TEXT,
# #                 ensemble_r2 REAL
# #             )
# #         """))
# #         conn.commit()

# # def ensure_db_columns():
# #     """Ensures all columns exist even if DB is old."""
# #     with engine.connect() as conn:
# #         existing_cols = pd.read_sql("PRAGMA table_info(results);", conn)["name"].tolist()
# #         missing = []
# #         if "ensemble_type" not in existing_cols:
# #             conn.execute(text("ALTER TABLE results ADD COLUMN ensemble_type TEXT"))
# #             missing.append("ensemble_type")
# #         if "ensemble_r2" not in existing_cols:
# #             conn.execute(text("ALTER TABLE results ADD COLUMN ensemble_r2 REAL"))
# #             missing.append("ensemble_r2")
# #         if missing:
# #             st.warning(f"🔧 Added columns to DB: {', '.join(missing)}")
# #         conn.commit()

# # init_db()
# # ensure_db_columns()

# # # ======================
# # # ENSEMBLE TRAINING
# # # ======================
# # @st.cache_resource
# # def train_and_select_ensemble(csv_path="nm RGB.csv", force_retrain=False):
# #     """Train base models, build ensembles, and select best."""
# #     if os.path.exists(MODEL_PATH) and not force_retrain:
# #         try:
# #             loaded = joblib.load(MODEL_PATH)
# #             st.info("Loaded saved ensemble model.")
# #             return loaded
# #         except Exception:
# #             st.warning("Failed to load model, retraining...")

# #     df = pd.read_csv(csv_path)
# #     X = df.drop(columns=["nm"])
# #     y = df["nm"].values
# #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# #     base_models = {
# #         "rf": RandomForestRegressor(n_estimators=200, random_state=42),
# #         "gbr": GradientBoostingRegressor(random_state=42),
# #         "svr": SVR(kernel="rbf"),
# #         "knn": KNeighborsRegressor(n_neighbors=5)
# #     }
# #     if XG_AVAILABLE:
# #         base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)

# #     val_scores = {}
# #     for name, model in base_models.items():
# #         model.fit(X_train, y_train)
# #         pred = model.predict(X_val)
# #         val_scores[name] = r2_score(y_val, pred)
# #         st.write(f"{name} → R² = {val_scores[name]:.4f}")

# #     # Weighted ensemble
# #     r2_vals = np.array([max(0, s) for s in val_scores.values()])
# #     weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

# #     def weighted_predict(X_input):
# #         return sum(w * m.predict(X_input) for w, m in zip(weights, base_models.values()))

# #     weighted_pred = weighted_predict(X_val)
# #     weighted_r2 = r2_score(y_val, weighted_pred)
# #     st.write(f"Weighted Ensemble R² = {weighted_r2:.4f}")

# #     # Stacking ensemble
# #     estimators = [(n, m) for n, m in base_models.items()]
# #     stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), n_jobs=-1)
# #     stack.fit(X_train, y_train)
# #     stack_pred = stack.predict(X_val)
# #     stack_r2 = r2_score(y_val, stack_pred)
# #     st.write(f"Stacking Ensemble R² = {stack_r2:.4f}")

# #     # Choose best
# #     if stack_r2 >= weighted_r2:
# #         chosen = {"type": "stacking", "model": stack, "r2": stack_r2}
# #         joblib.dump(chosen, MODEL_PATH)
# #         st.success(f"Selected Stacking Ensemble (R²={stack_r2:.3f})")
# #     else:
# #         chosen = {"type": "weighted", "models": base_models, "weights": weights, "r2": weighted_r2}
# #         joblib.dump(chosen, MODEL_PATH)
# #         st.success(f"Selected Weighted Ensemble (R²={weighted_r2:.3f})")
# #     return chosen

# # # ======================
# # # ENSEMBLE PREDICTION
# # # ======================
# # def ensemble_predict(model_obj, X_input):
# #     if model_obj["type"] == "stacking":
# #         return model_obj["model"].predict(X_input)
# #     elif model_obj["type"] == "weighted":
# #         preds = np.zeros(len(X_input))
# #         for w, m in zip(model_obj["weights"], model_obj["models"].values()):
# #             preds += w * m.predict(X_input)
# #         return preds
# #     else:
# #         raise ValueError("Invalid ensemble object")

# # # ======================
# # # ROI DETECTION
# # # ======================
# # def detect_dynamic_roi(frame, roi_size=DEFAULT_ROI_SIZE):
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.GaussianBlur(gray, (9, 9), 0)
# #     _, _, _, maxLoc = cv2.minMaxLoc(gray)
# #     xc, yc = maxLoc
# #     x1, y1 = max(0, xc - roi_size//2), max(0, yc - roi_size//2)
# #     x2, y2 = min(frame.shape[1], xc + roi_size//2), min(frame.shape[0], yc + roi_size//2)
# #     roi = frame[y1:y2, x1:x2]
# #     return roi

# # # ======================
# # # VIDEO FRAME ANALYSIS
# # # ======================
# # def analyze_video_frames(video_path, ensemble_obj):
# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         return {"error": "Cannot open video"}
# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     if fps == 0 or total_frames == 0:
# #         return {"error": "Invalid video file"}

# #     wavelengths, intensities = [], []
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         roi = detect_dynamic_roi(frame)
# #         if roi.size == 0:
# #             continue
# #         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# #         intensity = float(np.mean(gray))
# #         avg_rgb = cv2.resize(roi, (1, 1))[0, 0][::-1]
# #         pred_nm = ensemble_predict(ensemble_obj, np.array(avg_rgb).reshape(1, -1))[0]
# #         wavelengths.append(pred_nm)
# #         intensities.append(intensity)
# #     cap.release()

# #     if not wavelengths:
# #         return {"error": "No frames processed"}

# #     df = pd.DataFrame({"wavelength_nm": wavelengths, "intensity": intensities})
# #     stats = {
# #         "avg": float(np.mean(wavelengths)),
# #         "peak": float(np.max(wavelengths)),
# #         "min": float(np.min(wavelengths)),
# #         "max": float(np.max(wavelengths))
# #     }
# #     return {"data": df, "stats": stats}

# # # ======================
# # # EMISSION SPECTRUM PLOT (WAVELENGTH vs INTENSITY)
# # # ======================
# # def plot_emission_spectrum(df, sample_name):
# #     df_sorted = df.sort_values("wavelength_nm")
# #     df_unique = df_sorted.groupby("wavelength_nm", as_index=False)["intensity"].mean()
# #     x = df_unique["wavelength_nm"].values
# #     y = df_unique["intensity"].values
# #     y = y / np.max(y)
# #     if len(x) > 5:
# #         try:
# #             spline = make_interp_spline(x, y)
# #             x_s = np.linspace(x.min(), x.max(), 400)
# #             y_s = spline(x_s)
# #         except Exception:
# #             x_s, y_s = x, y
# #     else:
# #         x_s, y_s = x, y
# #     peak_idx = np.argmax(y_s)
# #     peak_wavelength = x_s[peak_idx]
# #     fig, ax = plt.subplots(figsize=(8, 5))
# #     ax.plot(x_s, y_s, color="royalblue", lw=2)
# #     ax.scatter(peak_wavelength, y_s[peak_idx], color="red", s=50, label=f"Peak: {peak_wavelength:.1f} nm")
# #     ax.set_xlabel("Wavelength (nm)")
# #     ax.set_ylabel("Normalized Intensity (a.u.)")
# #     ax.set_title(f"Emission Spectrum — {sample_name}")
# #     ax.grid(True, alpha=0.3)
# #     ax.legend()
# #     path = os.path.join(LOCAL_PLOTS_DIR, f"{uuid.uuid4()}.png")
# #     fig.savefig(path, dpi=150, bbox_inches="tight")
# #     st.pyplot(fig)
# #     plt.close(fig)
# #     return path, peak_wavelength

# # # ======================
# # # MAIN APP
# # # ======================
# # def main():
# #     st.title("⚡ Photoluminescence Spectroscopy — Ensemble Model")

# #     ensemble_obj = train_and_select_ensemble()
# #     ensemble_type = ensemble_obj.get("type", "stacking")
# #     ensemble_r2 = float(ensemble_obj.get("r2", np.nan))

# #     st.sidebar.write(f"Model: **{ensemble_type}** | R²={ensemble_r2:.3f}")
# #     if st.sidebar.button(" Retrain Models"):
# #         if os.path.exists(MODEL_PATH):
# #             os.remove(MODEL_PATH)
# #         ensemble_obj = train_and_select_ensemble(force_retrain=True)

# #     menu = ["Analyze New Sample", "View Results"]
# #     choice = st.sidebar.radio("Menu", menu)

# #     if choice == "Analyze New Sample":
# #         sample_name = st.text_input("Sample Name")
# #         uploaded = st.file_uploader("Upload Emission Video", type=["mp4", "avi", "mov"])
# #         voltage = st.slider("Voltage (V)", 0.0, 10.0, 0.0)
# #         if uploaded and sample_name.strip():
# #             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
# #                 tmp.write(uploaded.getbuffer())
# #                 path = tmp.name
# #             st.info("Processing video...")
# #             result = analyze_video_frames(path, ensemble_obj)
# #             os.remove(path)
# #             if "error" in result:
# #                 st.error(result["error"])
# #             else:
# #                 df = result["data"]
# #                 stats = result["stats"]
# #                 st.write(f"Average λ: {stats['avg']:.2f} nm | Range: {stats['min']:.2f}-{stats['max']:.2f} nm")
# #                 st.write(f"Peak λ: {stats['peak']:.2f} nm")
# #                 plot_path, peak_wl = plot_emission_spectrum(df, sample_name)
# #                 entry = {
# #                     "id": str(uuid.uuid4()),
# #                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #                     "sample_name": sample_name,
# #                     "voltage": float(voltage),
# #                     "avg_nm": float(stats["avg"]),
# #                     "peak_nm": float(peak_wl),
# #                     "min_nm": float(stats["min"]),
# #                     "max_nm": float(stats["max"]),
# #                     "plot_path": plot_path,
# #                     "ensemble_type": ensemble_type,
# #                     "ensemble_r2": float(ensemble_r2)
# #                 }
# #                 pd.DataFrame([entry]).to_sql("results", engine, if_exists="append", index=False)
# #                 st.success("Saved successfully ")

# #     elif choice == "View Results":
# #         df = pd.read_sql("SELECT * FROM results", engine)
# #         if df.empty:
# #             st.info("No results found yet.")
# #             return
# #         for col in ["ensemble_type", "ensemble_r2"]:
# #             if col not in df.columns:
# #                 df[col] = "N/A"
# #         samples = sorted(df["sample_name"].dropna().unique().tolist())
# #         selected = st.selectbox("Select Sample", samples)
# #         rows = df[df["sample_name"] == selected]
# #         display_cols = [c for c in ["timestamp", "voltage", "avg_nm", "peak_nm", "min_nm", "max_nm", "ensemble_type"] if c in rows.columns]
# #         st.dataframe(rows[display_cols])
# #         if "plot_path" in rows.columns and os.path.exists(rows.iloc[0]["plot_path"]):
# #             st.image(rows.iloc[0]["plot_path"], caption=f"Spectrum — {selected}")

# # if __name__ == "__main__":
# #     main()

import streamlit as st
st.set_page_config(page_title="PL Ensemble System", layout="wide")

import pandas as pd
import os
import cv2
import numpy as np
import uuid
import tempfile
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sqlalchemy import create_engine, text
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XG_AVAILABLE = True
except Exception:
    XG_AVAILABLE = False

# ======================
# CONFIG
# ======================
RESULTS_DB = "sqlite:///pl_results.db"
engine = create_engine(RESULTS_DB)
LOCAL_PLOTS_DIR = "saved_plots"
os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)
MODEL_PATH = "best_ensemble_model.joblib"
DEFAULT_ROI_SIZE = 50

# ======================
# AUTO DB INIT / FIXER
# ======================
def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                sample_name TEXT,
                voltage REAL,
                avg_nm REAL,
                peak_nm REAL,
                min_nm REAL,
                max_nm REAL,
                plot_path TEXT,
                ensemble_type TEXT,
                ensemble_r2 REAL
            )
        """))
        conn.commit()

def ensure_db_columns():
    with engine.connect() as conn:
        existing_cols = pd.read_sql("PRAGMA table_info(results);", conn)["name"].tolist()
        missing = []
        if "ensemble_type" not in existing_cols:
            conn.execute(text("ALTER TABLE results ADD COLUMN ensemble_type TEXT"))
            missing.append("ensemble_type")
        if "ensemble_r2" not in existing_cols:
            conn.execute(text("ALTER TABLE results ADD COLUMN ensemble_r2 REAL"))
            missing.append("ensemble_r2")
        if missing:
            st.warning(f"🔧 Added columns to DB: {', '.join(missing)}")
        conn.commit()

init_db()
ensure_db_columns()

# ======================
# ENSEMBLE TRAINING
# ======================
@st.cache_resource
def train_and_select_ensemble(csv_path="nm RGB.csv", force_retrain=False):
    """Train base models, build ensembles, and select best."""
    if os.path.exists(MODEL_PATH) and not force_retrain:
        try:
            loaded = joblib.load(MODEL_PATH)
            st.info("Loaded saved ensemble model.")
            return loaded
        except Exception:
            st.warning("Failed to load model, retraining...")

    df = pd.read_csv(csv_path)
    X = df.drop(columns=["nm"])
    y = df["nm"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    kernel = ConstantKernel(1e5, (1e2, 1e8)) * \
    RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-3, 1e3)) + \
    WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e3))

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=42
    )
    base_models = {
        "rf": RandomForestRegressor(n_estimators=200, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42),
        "svr": gpr,
        "knn": KNeighborsRegressor(n_neighbors=5)
    }
    if XG_AVAILABLE:
        base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)

    val_scores = {}
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_scores[name] = r2_score(y_val, pred)
        st.write(f"{name} → R² = {val_scores[name]:.4f}")

    # Weighted ensemble
    r2_vals = np.array([max(0, s) for s in val_scores.values()])
    weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

    def weighted_predict(X_input):
        return sum(w * m.predict(X_input) for w, m in zip(weights, base_models.values()))

    weighted_pred = weighted_predict(X_val)
    weighted_r2 = r2_score(y_val, weighted_pred)
    st.write(f"Weighted Ensemble R² = {weighted_r2:.4f}")

    # Stacking ensemble
    estimators = [(n, m) for n, m in base_models.items()]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), n_jobs=-1)
    stack.fit(X_train, y_train)
    stack_pred = stack.predict(X_val)
    stack_r2 = r2_score(y_val, stack_pred)
    st.write(f"Stacking Ensemble R² = {stack_r2:.4f}")

    # Choose best
    if stack_r2 >= weighted_r2:
        chosen = {"type": "stacking", "model": stack, "r2": stack_r2}
        joblib.dump(chosen, MODEL_PATH)
        st.success(f" Selected Stacking Ensemble (R²={stack_r2:.3f})")
    else:
        chosen = {"type": "weighted", "models": base_models, "weights": weights, "r2": weighted_r2}
        joblib.dump(chosen, MODEL_PATH)
        st.success(f" Selected Weighted Ensemble (R²={weighted_r2:.3f})")
    return chosen

# ======================
# ENSEMBLE PREDICTION
# ======================
def ensemble_predict(model_obj, X_input):
    if model_obj["type"] == "stacking":
        return model_obj["model"].predict(X_input)
    elif model_obj["type"] == "weighted":
        preds = np.zeros(len(X_input))
        for w, m in zip(model_obj["weights"], model_obj["models"].values()):
            preds += w * m.predict(X_input)
        return preds
    else:
        raise ValueError("Invalid ensemble object")

# ======================
# VIDEO FRAME ANALYSIS WITH ATTENTION-LIKE WEIGHTING
# ======================
def analyze_video_frames(video_path, ensemble_obj):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or total_frames == 0:
        return {"error": "Invalid video file"}

    frame_idx = 0
    frame_batch_rgb = []
    frame_batch_intensity = []
    wavelengths, intensities = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w, _ = frame.shape
        grid_size = 8
        patch_h, patch_w = h // grid_size, w // grid_size

        patch_rgbs = []
        patch_weights = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * patch_h, (i + 1) * patch_h
                x1, x2 = j * patch_w, (j + 1) * patch_w
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                intensity = np.mean(gray)
                rgb_mean = cv2.resize(patch, (1, 1))[0, 0][::-1]
                patch_rgbs.append(rgb_mean)
                patch_weights.append(intensity ** 2)

        patch_weights = np.array(patch_weights)
        if patch_weights.sum() == 0:
            continue
        patch_weights /= patch_weights.sum()
        patch_rgbs = np.array(patch_rgbs)
        weighted_rgb = np.average(patch_rgbs, axis=0, weights=patch_weights)

        frame_batch_rgb.append(weighted_rgb)
        frame_batch_intensity.append(np.mean(patch_weights))

        # Aggregate every 26 frames (~1 sec)
        if frame_idx % 26 == 0:
            avg_rgb = np.mean(frame_batch_rgb, axis=0)
            avg_intensity = np.mean(frame_batch_intensity)
            pred_nm = ensemble_predict(ensemble_obj, np.array(avg_rgb).reshape(1, -1))[0]
            wavelengths.append(pred_nm)
            intensities.append(avg_intensity)
            frame_batch_rgb, frame_batch_intensity = [], []

    cap.release()

    if not wavelengths:
        return {"error": "No frames processed"}

    df = pd.DataFrame({"wavelength_nm": wavelengths, "intensity": intensities})
    stats = {
        "avg": float(np.mean(wavelengths)),
        "peak": float(np.max(wavelengths)),
        "min": float(np.min(wavelengths)),
        "max": float(np.max(wavelengths))
    }
    return {"data": df, "stats": stats}

# ======================
# EMISSION SPECTRUM PLOT
# ======================
def plot_emission_spectrum(df, sample_name):
    """
    Build a smooth, physically-meaningful Wavelength vs Intensity plot by:
      1) creating weighted histogram (weights = intensity)
      2) smoothing with a Gaussian filter
      3) detecting peak wavelength(s)
    Input:
      df - DataFrame with columns 'wavelength_nm' and 'intensity'
      sample_name - string for titles/filenames
    Returns:
      path (png), peak_wavelength (float)
    """
    if df.empty:
        raise ValueError("Empty dataframe passed to plot_emission_spectrum")

    # use the raw arrays
    wl = np.asarray(df["wavelength_nm"].values, dtype=float)
    inten = np.asarray(df["intensity"].values, dtype=float)

    # if all wavelengths identical (degenerate), handle separately
    if np.allclose(wl, wl[0]):
        peak_wl = float(wl[0])
        # create a tiny synthetic spectrum for plotting
        x_s = np.linspace(peak_wl - 5, peak_wl + 5, 200)
        y_s = np.exp(-0.5 * ((x_s - peak_wl) / 1.5) ** 2)
        y_s = y_s / np.max(y_s)
    else:
        # choose number of bins based on range, but cap to reasonable limits
        wl_min, wl_max = wl.min(), wl.max()
        wl_range = max(1e-6, wl_max - wl_min)
        # bins: try ~ (range * 2) or between 80 and 400 bins
        bins = int(np.clip(wl_range * 2.0, 80, 400))

        # create bins and compute weighted histogram (weights = intensity)
        hist, bin_edges = np.histogram(wl, bins=bins, range=(wl_min - 0.1*wl_range, wl_max + 0.1*wl_range), weights=inten)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # normalize histogram (avoid divide by zero)
        if hist.max() > 0:
            hist = hist.astype(float) / np.max(hist)
        else:
            hist = hist.astype(float)

        # smooth with gaussian; sigma controls smoothing (tuneable)
        # choose sigma relative to number of bins
        sigma = max(1.0, bins / 150.0)
        y_s = gaussian_filter1d(hist, sigma=sigma)
        x_s = bin_centers

        # if smoothing introduced tiny negative values (rare), clip
        y_s = np.clip(y_s, 0.0, None)
        if y_s.max() > 0:
            y_s = y_s / y_s.max()

        # peak detection: require peaks above 20% of max (adjustable)
        peaks, props = find_peaks(y_s, height=0.2, distance=3)
        if len(peaks) > 0:
            # choose the highest local peak
            peak_idx = peaks[np.argmax(props["peak_heights"])]
            peak_wl = float(x_s[peak_idx])
        else:
            # fallback to global maximum
            peak_idx = int(np.argmax(y_s))
            peak_wl = float(x_s[peak_idx])

    # ----------------- Plotting -----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_s, y_s, color="royalblue", lw=2, label="Intensity (smoothed)")
    # mark peak
    ax.scatter(peak_wl, float(np.interp(peak_wl, x_s, y_s)), color="red", s=50, zorder=5,
               label=f"Peak: {peak_wl:.2f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity (a.u.)")
    ax.set_title(f"Emission Spectrum — {sample_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # save and show
    path = os.path.join(LOCAL_PLOTS_DIR, f"{uuid.uuid4()}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)

    return path, peak_wl

# ======================
# MAIN APP
# ======================
def main():
    st.title("⚡ Photoluminescence Spectroscopy — Ensemble Model with Attention")

    ensemble_obj = train_and_select_ensemble()
    ensemble_type = ensemble_obj.get("type", "stacking")
    ensemble_r2 = float(ensemble_obj.get("r2", np.nan))

    st.sidebar.write(f"Model: **{ensemble_type}** | R²={ensemble_r2:.3f}")
    if st.sidebar.button("🔁 Retrain Models"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        ensemble_obj = train_and_select_ensemble(force_retrain=True)

    menu = ["Analyze New Sample", "View Results"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Analyze New Sample":
        sample_name = st.text_input("Sample Name")
        uploaded = st.file_uploader("Upload Emission Video", type=["mp4", "avi", "mov"])
        voltage = st.slider("Voltage (V)", 0.0, 10.0, 0.0)
        if uploaded and sample_name.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.getbuffer())
                path = tmp.name
            st.info("Processing video...")
            result = analyze_video_frames(path, ensemble_obj)
            os.remove(path)
            if "error" in result:
                st.error(result["error"])
            else:
                df = result["data"]
                stats = result["stats"]
                st.write(f"Average λ: {stats['avg']:.2f} nm | Range: {stats['min']:.2f}-{stats['max']:.2f} nm")
                st.write(f"Peak λ: {stats['peak']:.2f} nm")
                plot_path, peak_wl = plot_emission_spectrum(df, sample_name)
                entry = {
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sample_name": sample_name,
                    "voltage": float(voltage),
                    "avg_nm": float(stats["avg"]),
                    "peak_nm": float(peak_wl),
                    "min_nm": float(stats["min"]),
                    "max_nm": float(stats["max"]),
                    "plot_path": plot_path,
                    "ensemble_type": ensemble_type,
                    "ensemble_r2": float(ensemble_r2)
                }
                pd.DataFrame([entry]).to_sql("results", engine, if_exists="append", index=False)
                st.success("Saved successfully ")

    elif choice == "View Results":
        df = pd.read_sql("SELECT * FROM results", engine)
        if df.empty:
            st.info("No results found yet.")
            return
        samples = sorted(df["sample_name"].dropna().unique().tolist())
        selected = st.selectbox("Select Sample", samples)
        rows = df[df["sample_name"] == selected]
        display_cols = [c for c in ["timestamp", "voltage", "avg_nm", "peak_nm", "min_nm", "max_nm", "ensemble_type"] if c in rows.columns]
        st.dataframe(rows[display_cols])
        if "plot_path" in rows.columns and os.path.exists(rows.iloc[0]["plot_path"]):
            st.image(rows.iloc[0]["plot_path"], caption=f"Spectrum — {selected}")

if __name__ == "__main__":
    main()
# import streamlit as st
# st.set_page_config(page_title="PL Ensemble System", layout="wide")

# import pandas as pd
# import os
# import cv2
# import numpy as np
# import uuid
# import tempfile
# import joblib
# from datetime import datetime
# import matplotlib.pyplot as plt

# from sqlalchemy import create_engine, text
# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import find_peaks

# # Canvas ROI tool
# from streamlit_drawable_canvas import st_canvas
# from PIL import Image

# # ML
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score


# # ======================================
# # CONFIG
# # ======================================
# MODEL_PATH = "best_ensemble_model.joblib"
# RESULTS_DB = "sqlite:///pl_results.db"

# engine = create_engine(RESULTS_DB)


# # ======================================
# # TRAIN ENSEMBLE MODEL
# # ======================================
# def train_and_select_ensemble(csv_path="nm RGB.csv", force_retrain=False):

#     if os.path.exists(MODEL_PATH) and not force_retrain:
#         try:
#             return joblib.load(MODEL_PATH)
#         except:
#             pass

#     df = pd.read_csv(csv_path)

#     X = df.drop(columns=["nm"])
#     y = df["nm"].values

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     base_models = {
#         "rf": RandomForestRegressor(n_estimators=250, random_state=42),
#         "gbr": GradientBoostingRegressor(random_state=42),
#         "svr": SVR(kernel="rbf"),
#         "knn": KNeighborsRegressor(n_neighbors=4),
#     }

#     scores = {}
#     for name, model in base_models.items():
#         model.fit(X_train, y_train)
#         pred = model.predict(X_val)
#         scores[name] = r2_score(y_val, pred)

#     # Weighted ensemble
#     r2_vals = np.array([max(0, s) for s in scores.values()])
#     weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

#     def weighted_predict(X):
#         return sum(w * m.predict(X) for w, m in zip(weights, base_models.values()))

#     weighted_r2 = r2_score(y_val, weighted_predict(X_val))

#     # Stacking
#     estimators = [(n, m) for n, m in base_models.items()]
#     stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
#     stack.fit(X_train, y_train)
#     stack_r2 = r2_score(y_val, stack.predict(X_val))

#     # Choose best
#     if stack_r2 >= weighted_r2:
#         model_obj = {"type": "stacking", "model": stack}
#     else:
#         model_obj = {"type": "weighted", "models": base_models, "weights": weights}

#     joblib.dump(model_obj, MODEL_PATH)

#     return model_obj


# # Load model
# model_obj = train_and_select_ensemble()


# # ======================================
# # ENSEMBLE PREDICT
# # ======================================
# def ensemble_predict(model_obj, X):
#     if model_obj["type"] == "stacking":
#         return model_obj["model"].predict(X)
#     else:
#         preds = np.zeros(len(X))
#         for w, m in zip(model_obj["weights"], model_obj["models"].values()):
#             preds += w * m.predict(X)
#         return preds


# # ======================================
# # VIDEO ANALYSIS WITH MANUAL ROI
# # ======================================
# def analyze_video_frames(video_path, model_obj, roi):
#     x, y, w, h = roi

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None

#     wavelengths = []
#     intensities = []
#     grid = 5  # sampling subdivision inside ROI

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         crop = frame[y:y + h, x:x + w]
#         if crop.size == 0:
#             continue

#         ph, pw = crop.shape[0] // grid, crop.shape[1] // grid

#         patch_rgb = []
#         patch_weights = []

#         for i in range(grid):
#             for j in range(grid):
#                 y1, y2 = i * ph, (i + 1) * ph
#                 x1, x2 = j * pw, (j + 1) * pw

#                 patch = crop[y1:y2, x1:x2]
#                 if patch.size == 0:
#                     continue

#                 gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
#                 intensity = float(np.mean(gray))

#                 rgb = cv2.resize(patch, (1, 1))[0, 0][::-1]

#                 patch_rgb.append(rgb)
#                 patch_weights.append(intensity ** 2)

#         patch_rgb = np.array(patch_rgb)
#         patch_weights = np.array(patch_weights)

#         patch_weights /= np.sum(patch_weights)

#         final_rgb = np.average(patch_rgb, axis=0, weights=patch_weights)
#         nm = ensemble_predict(model_obj, np.array(final_rgb).reshape(1, -1))[0]

#         wavelengths.append(float(nm))
#         intensities.append(float(np.mean(patch_weights)))

#     cap.release()

#     if not wavelengths:
#         return None

#     return {
#         "avg_nm": float(np.mean(wavelengths)),
#         "peak_nm": float(np.max(wavelengths)),
#         "min_nm": float(np.min(wavelengths)),
#         "max_nm": float(np.max(wavelengths)),
#         "wavelength_list": wavelengths,
#     }


# # ======================================
# # STREAMLIT UI
# # ======================================
# def main():
#     st.title("⚡ Photoluminescence — Manual ROI Analyzer")

#     sample_name = st.text_input("Sample Name")
#     uploaded = st.file_uploader("Upload Emission Video", type=["mp4", "avi", "mov"])

#     if uploaded:
#         # Save temporary video
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#             tmp.write(uploaded.getbuffer())
#             video_path = tmp.name

#         # Read first frame
#         cap = cv2.VideoCapture(video_path)
#         ret, frame = cap.read()
#         cap.release()

#         if not ret:
#             st.error("Could not read video.")
#             return

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         rgb_pil = Image.fromarray(rgb)

#         st.subheader("🎯 Draw ROI on First Frame")

#         canvas = st_canvas(
#             stroke_width=3,
#             stroke_color="red",
#             background_image=rgb_pil,
#             height=rgb.shape[0],
#             width=rgb.shape[1],
#             drawing_mode="rect",
#             key="roi_canvas",
#         )

#         roi = None
#         if canvas.json_data and len(canvas.json_data["objects"]) > 0:
#             obj = canvas.json_data["objects"][0]
#             roi = (
#                 int(obj["left"]),
#                 int(obj["top"]),
#                 int(obj["width"]),
#                 int(obj["height"]),
#             )

#             st.success(f"ROI Selected: {roi}")

#         if st.button("Process Video"):
#             if roi is None:
#                 st.error("Please draw ROI first!")
#                 return

#             result = analyze_video_frames(video_path, model_obj, roi)
#             os.remove(video_path)

#             if result is None:
#                 st.error("Could not process video.")
#             else:
#                 st.success("Video Processed!")
#                 st.json(result)

#                 # Plot wavelength line graph
#                 st.line_chart(result["wavelength_list"])


# if __name__ == "__main__":
#     main()
