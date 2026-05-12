import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import gc
import json
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import joblib
import faulthandler

faulthandler.enable(all_threads=True)

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

try:
    import shap as shap_lib
except Exception as e:
    raise ImportError(
        "shap 未安装或不可用。建议：pip install shap 或 conda install -c conda-forge shap\n"
        f"原始错误：{e}"
    )

RAW_DIR = "/mnt/l/RF_Value/RF_Input_03Deg"
AI_PATH = os.path.join(RAW_DIR, "AI_ET0_annual_01D__aligned_03deg.nc")
AI_VAR = "AI"
SOIL_PATH = os.path.join(RAW_DIR, "HWSD2_topsoil_01D_conservative__aligned_03deg.nc")
SOIL_VARS = {"CACO3": "CACO3", "Clay": "Clay", "OC": "OC", "Sand": "Sand"}

IN_DIR_EVENT_RAW = r"/mnt/l/FD_Data_Process/RF_Model/Result_DOD_Value/FD_RF_Input_Data_GATED_BY_DOD_03Deg_FD_Ave"
IN_DIR_ZSCORE_MEAN = r"/mnt/l/FD_Data_Process/RF_Model/Result_DOD_Value/FD_RF_Input_Data_GATED_BY_DOD_03Deg_FD_Ave_ZScore"
IN_DIR_ZSCORE_MAX = r"/mnt/l/FD_Data_Process/RF_Model/Result_DOD_Value/FD_RF_Input_Data_GATED_BY_DOD_03Deg_FD_Max_ZScore"

FILE_SUFFIX = ".nc"
FILE_MUST_CONTAIN_MEAN = "__eventmean_0p3deg_ordered"
FILE_MUST_CONTAIN_MAX = "__eventmax_0p3deg_ordered"

DEDUP_SHARED_VARS = set()
AUTO_DETECT_SHARED_VARS = False

STATIC_VARS = {"AI", "CACO3", "Clay", "OC", "Sand"}
EVENT_META_VARS = set()

KEEP_RAW_KEYWORDS = (
    "pentad_raw",
    "_raw_event_mean",
    "_5d_event_mean",
    "_5d_event_max",
)

DROP_ANOMALY_KEYWORDS = (
    "anomaly",
    "_anom",
)

REMOTE_OUT_DIR = "/mnt/l/FD_Data_Process/RF_Model/Result_DOD_Value/RF/RF_OOF_SHAP_Result_EVENT_RandomKFold_FIXED600_B_WithEventMax_FAST_V2"
LOCAL_OUT_DIR = "/tmp/RF_OOF_SHAP_Result_EVENT_RandomKFold_FIXED600_B_WithEventMax_FAST_V2"

OUT_DIR = LOCAL_OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REMOTE_OUT_DIR, exist_ok=True)

REMOTE_FOLD_MODEL_DIR = os.path.join(REMOTE_OUT_DIR, "fold_models")
LOCAL_FOLD_MODEL_DIR = os.path.join(LOCAL_OUT_DIR, "fold_models")
os.makedirs(REMOTE_FOLD_MODEL_DIR, exist_ok=True)
os.makedirs(LOCAL_FOLD_MODEL_DIR, exist_ok=True)

FOLD_MODEL_DIR = LOCAL_FOLD_MODEL_DIR

CV_JSON = os.path.join(OUT_DIR, "cv_metrics.json")
FOLDS_CSV = os.path.join(OUT_DIR, "random_kfold_folds.csv")
FOLDS_JSON = os.path.join(OUT_DIR, "random_kfold_folds.json")
OOF_PRED = os.path.join(OUT_DIR, "y_oof.npy")
OOF_OBS = os.path.join(OUT_DIR, "y_obs.npy")
SHAP_OOF_NC = os.path.join(OUT_DIR, "shap_maps_rf_OOF.nc")

SAVE_XY_FOR_PLOTTING = True
X_ALL_NPY = os.path.join(OUT_DIR, "X_all.npy")
Y_ALL_NPY = os.path.join(OUT_DIR, "y_all.npy")
FEAT_JSON = os.path.join(OUT_DIR, "feature_names.json")
META_NPZ = os.path.join(OUT_DIR, "event_meta.npz")

META_VARS = {
    "event_id", "lat_idx", "lon_idx", "lat", "lon", "start", "end",
    "duration_days", "dod_event_mean", "duration_steps", "cumulative_deficit_total",
}

SEED = 42
N_SPLITS = 5
KFOLD_SHUFFLE = True

RF_TRAIN_NJOBS = 8
RF_MAX_SAMPLES = None

RF_PARAMS = dict(
    n_estimators=600,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features=0.7,
    bootstrap=True,
    oob_score=False,
    n_jobs=RF_TRAIN_NJOBS,
    random_state=SEED,
)

OOF_SHAP_ENABLE = True
OOF_SHAP_MAX_PER_FOLD = 30000
SHAP_BATCH_SIZE = 2000

TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

LANDMASK_MODE = "finite"
SOIL_EPS = 1e-12

FEATURE_GROUPS = {
    "Meteorology": [
        "wind_speed_5d_event_mean",
        "wind_speed_5d_event_max",
        "blh_5d_event_mean",
        "sshf_5d_event_mean",
        "tp_5d_event_mean",
    ],
    "HydroVeg": [
        "sms_5d_event_mean",
        "EVI_pentad_raw_event_mean",
    ],
    "Snow": [
        "snowc_5d_event_mean",
    ],
    "Surface": ["AI", "CACO3", "Clay", "OC", "Sand"],
}

DOMINANT_GROUP_METHOD = "mean"
MIN_SAMPLES_FOR_DOMINANT = 10
SKIP_IF_EXISTS = True
STRLEN = 128

SHAP_CACHE_DIR = os.path.join(OUT_DIR, "_shap_fold_cache")
os.makedirs(SHAP_CACHE_DIR, exist_ok=True)


def atomic_write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def atomic_save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def ensure_lat_lon(da):
    rename_map = {}
    for d in da.dims:
        dl = d.lower()
        if dl == "latitude":
            rename_map[d] = "lat"
        elif dl == "longitude":
            rename_map[d] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da


def list_event_files(indir, must_contain):
    out = []
    for fn in os.listdir(indir):
        if not fn.lower().endswith(FILE_SUFFIX):
            continue
        if must_contain and (must_contain.lower() not in fn.lower()):
            continue
        out.append(fn)
    out.sort()
    return out


def base_key_from_filename(fn):
    b = os.path.basename(fn)
    if b.lower().endswith("__zscore.nc"):
        return b[:-len("__ZScore.nc")]
    if b.lower().endswith(".nc"):
        return b[:-3]
    return b


def build_file_map(indir, must_contain):
    fns = list_event_files(indir, must_contain)
    mp_ = {}
    for fn in fns:
        mp_[base_key_from_filename(fn)] = os.path.join(indir, fn)
    return mp_


def choose_feature_name(v, key_name, feat_names, dedup_shared_vars):
    if (v not in dedup_shared_vars) and (v in feat_names):
        return f"{v}__{key_name}"
    return v


def is_anomaly_like(name):
    s = name.lower()
    return any(k in s for k in DROP_ANOMALY_KEYWORDS)


def is_raw_event_feature(name):
    s = name.lower()
    if not (s.endswith("_event_mean") or s.endswith("_event_max")):
        return False
    return any(k in s for k in KEEP_RAW_KEYWORDS)


def should_use_predictor(var_name):
    if var_name in {"duration_steps", "cumulative_deficit_total"}:
        return False
    if var_name in STATIC_VARS:
        return True
    if var_name in EVENT_META_VARS:
        return True
    if var_name in META_VARS:
        return False
    if is_anomaly_like(var_name):
        return False
    if is_raw_event_feature(var_name):
        return True
    return False


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"R2": float(r2), "RMSE": float(rmse), "MAE": float(mae)}


def create_feature_group_mapping(feature_names, feature_groups):
    group_names = list(feature_groups.keys())
    feat2g = {}
    for gi, (gn, feats) in enumerate(feature_groups.items()):
        for f in feats:
            if f in feature_names:
                feat2g[f] = gi

    unmapped = [f for f in feature_names if f not in feat2g]
    if unmapped:
        if "Other" not in group_names:
            group_names.append("Other")
        oi = group_names.index("Other")
        for f in unmapped:
            feat2g[f] = oi

    return feat2g, group_names


def aggregate_shap_to_maps(
    shap_values,
    meta,
    feature_names,
    lat_vals,
    lon_vals,
    feature_groups=None,
    group_method="mean",
    min_samples_dominant=10,
):
    nlat, nlon = int(meta["nlat"]), int(meta["nlon"])
    p = len(feature_names)
    pix = meta["la_idx"].astype(np.int64) * nlon + meta["lo_idx"].astype(np.int64)
    n_pix = nlat * nlon

    sum_raw = np.zeros((n_pix, p), dtype=np.float32)
    sum_abs = np.zeros((n_pix, p), dtype=np.float32)
    sum_pos = np.zeros((n_pix, p), dtype=np.float32)

    for j in range(p):
        col = np.asarray(shap_values[:, j], dtype=np.float32)
        sum_raw[:, j] = np.bincount(pix, weights=col, minlength=n_pix)
        sum_abs[:, j] = np.bincount(pix, weights=np.abs(col), minlength=n_pix)
        sum_pos[:, j] = np.bincount(pix, weights=(col > 0).astype(np.float32), minlength=n_pix)
        del col

    cnt = np.bincount(pix, minlength=n_pix).astype(np.int64)
    valid = cnt > 0

    mean_raw = np.full((n_pix, p), np.nan, dtype=np.float32)
    mean_abs = np.full((n_pix, p), np.nan, dtype=np.float32)
    frac_pos = np.full((n_pix, p), np.nan, dtype=np.float32)

    mean_raw[valid] = sum_raw[valid] / cnt[valid, None]
    mean_abs[valid] = sum_abs[valid] / cnt[valid, None]
    frac_pos[valid] = sum_pos[valid] / cnt[valid, None]

    sign_stability = np.full((n_pix, p), np.nan, dtype=np.float32)
    if np.any(valid):
        denom = mean_abs[valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            sign_stability[valid] = np.abs(mean_raw[valid]) / denom
        sign_stability[valid][denom == 0] = np.nan

    sign_stability = np.clip(sign_stability, 0.0, 1.0)

    mean_raw_3d = mean_raw.reshape(nlat, nlon, p)
    mean_abs_3d = mean_abs.reshape(nlat, nlon, p)
    frac_pos_3d = frac_pos.reshape(nlat, nlon, p)
    sign_stability_3d = sign_stability.reshape(nlat, nlon, p)
    n_samples_per_pixel = cnt.reshape(nlat, nlon).astype(np.int32)

    valid_2d = cnt.reshape(nlat, nlon) >= min_samples_dominant
    has_any = np.isfinite(mean_abs_3d).any(axis=2)
    valid_final = valid_2d & has_any

    dom = np.full((nlat, nlon), -1, dtype=np.int16)
    if np.any(valid_final):
        dom[valid_final] = np.nanargmax(mean_abs_3d[valid_final], axis=1).astype(np.int16)

    dom_name = np.full((nlat, nlon), b"", dtype=f"S{STRLEN}")
    if np.any(valid_final):
        fn_s = np.asarray(feature_names, dtype=f"S{STRLEN}")
        dom_name[valid_final] = fn_s[dom[valid_final].astype(int)]

    dom_group = None
    dom_group_name = None
    group_names = None

    if feature_groups is not None:
        feat2g, group_names = create_feature_group_mapping(feature_names, feature_groups)
        n_groups = len(group_names)
        mean_abs_by_group = np.full((nlat, nlon, n_groups), np.nan, dtype=np.float32)

        with np.errstate(invalid="ignore"):
            for g in range(n_groups):
                feats = [i for i, f in enumerate(feature_names) if feat2g.get(f, -1) == g]
                if feats:
                    if group_method == "mean":
                        mean_abs_by_group[:, :, g] = np.nanmean(mean_abs_3d[:, :, feats], axis=2)
                    elif group_method == "max":
                        mean_abs_by_group[:, :, g] = np.nanmax(mean_abs_3d[:, :, feats], axis=2)

        dom_group = np.full((nlat, nlon), -1, dtype=np.int16)
        has_any_group = np.isfinite(mean_abs_by_group).any(axis=2)
        valid_group = valid_final & has_any_group

        if np.any(valid_group):
            dom_group[valid_group] = np.nanargmax(mean_abs_by_group[valid_group], axis=1).astype(np.int16)

        dom_group_name = np.full((nlat, nlon), b"", dtype=f"S{STRLEN}")
        if np.any(valid_group):
            gn_s = np.asarray(group_names, dtype=f"S{STRLEN}")
            dom_group_name[valid_group] = gn_s[dom_group[valid_group].astype(int)]

    ds = xr.Dataset(
        data_vars={
            "mean_shap": (("lat", "lon", "feature"), mean_raw_3d),
            "mean_abs_shap": (("lat", "lon", "feature"), mean_abs_3d),
            "frac_positive_shap": (("lat", "lon", "feature"), frac_pos_3d),
            "sign_stability": (("lat", "lon", "feature"), sign_stability_3d),
            "n_samples": (("lat", "lon"), n_samples_per_pixel),
            "dominant_driver_index": (("lat", "lon"), dom),
            "dominant_driver_name": (("lat", "lon"), dom_name),
        },
        coords={
            "lat": ("lat", lat_vals),
            "lon": ("lon", lon_vals),
            "feature": ("feature", feature_names),
        },
    )

    if dom_group is not None:
        ds["dominant_group_index"] = (("lat", "lon"), dom_group)
        ds["dominant_group_name"] = (("lat", "lon"), dom_group_name)
        ds = ds.assign_coords(group=("group", group_names))

    coverage_stats = {
        "valid_pixel_count": int(valid_final.sum()),
        "total_pixel_count": int(nlat * nlon),
        "valid_pixel_fraction": float(valid_final.sum() / (nlat * nlon)),
        "min_samples_for_dominant": int(min_samples_dominant),
    }

    return ds, coverage_stats


def _nc_safe(v):
    if v is None:
        return "null"
    if isinstance(v, (str, bytes, int, float, bool)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        return str(v)


def sanitize_dataset_for_netcdf(ds):
    ds = ds.copy()
    ds.attrs = {str(k): _nc_safe(v) for k, v in ds.attrs.items()}
    for var in ds.variables:
        if getattr(ds[var], "attrs", None):
            ds[var].attrs = {str(k): _nc_safe(v) for k, v in ds[var].attrs.items()}
    return ds


def _as_int_eventid(arr):
    if arr.dtype.kind in ("S", "U"):
        return np.asarray(arr, dtype=str).astype(np.int64)
    return np.asarray(arr).astype(np.int64)


def align_dataset_to_ref_event_ids(ds, ref_event_ids):
    cur_ids = _as_int_eventid(ds["event_id"].values)
    ref_ids = _as_int_eventid(ref_event_ids)
    ref_map = {int(v): i for i, v in enumerate(ref_ids)}
    cur_map = {int(v): i for i, v in enumerate(cur_ids)}
    common = sorted(set(ref_map.keys()) & set(cur_map.keys()))

    if len(common) == 0:
        raise ValueError("No common event_id between current dataset and reference event table.")

    ref_idx = np.asarray([ref_map[i] for i in common], dtype=np.int64)
    cur_idx = np.asarray([cur_map[i] for i in common], dtype=np.int64)
    ds2 = ds.isel(event=xr.DataArray(cur_idx, dims="event"))

    return ds2, ref_idx, cur_idx, np.asarray(common, dtype=np.int64)


def sync_fold_models_from_remote():
    if not os.path.isdir(REMOTE_FOLD_MODEL_DIR):
        return

    copied = 0
    skipped = 0

    for fn in os.listdir(REMOTE_FOLD_MODEL_DIR):
        if not fn.endswith(".joblib"):
            continue

        src = os.path.join(REMOTE_FOLD_MODEL_DIR, fn)
        dst = os.path.join(LOCAL_FOLD_MODEL_DIR, fn)

        need_copy = True
        if os.path.exists(dst):
            try:
                ss = os.stat(src)
                ds = os.stat(dst)
                if ss.st_size == ds.st_size and int(ss.st_mtime) == int(ds.st_mtime):
                    need_copy = False
            except Exception:
                need_copy = True

        if need_copy:
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

    print(f"  ✅ Sync fold models to local: copied={copied}, skipped={skipped}")


def mirror_local_model_to_remote(fold_model_path_local):
    fn = os.path.basename(fold_model_path_local)
    dst = os.path.join(REMOTE_FOLD_MODEL_DIR, fn)

    try:
        shutil.copy2(fold_model_path_local, dst)
    except Exception as e:
        print(f"  ⚠️ mirror model to remote failed: {fn} | {e}")


def mirror_results_back_to_remote():
    os.makedirs(REMOTE_OUT_DIR, exist_ok=True)

    items = [
        "cv_metrics.json",
        "random_kfold_folds.csv",
        "random_kfold_folds.json",
        "y_oof.npy",
        "y_obs.npy",
        "shap_maps_rf_OOF.nc",
        "X_all.npy",
        "y_all.npy",
        "feature_names.json",
        "event_meta.npz",
        "_shap_fold_cache",
        "fold_models",
    ]

    for name in items:
        src = os.path.join(OUT_DIR, name)
        dst = os.path.join(REMOTE_OUT_DIR, name)

        if not os.path.exists(src):
            continue

        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"  ⚠️ mirror result failed: {name} | {e}")


def _fold_cache_paths(fold_id):
    stem = os.path.join(SHAP_CACHE_DIR, f"fold{fold_id:02d}")
    return {
        "shap": stem + "_shap.npy",
        "la": stem + "_la.npy",
        "lo": stem + "_lo.npy",
        "meta": stem + "_meta.json",
    }


def _load_fold_model(fold_model_path):
    saved = joblib.load(fold_model_path)
    if isinstance(saved, dict) and "model" in saved:
        return saved["model"]
    return saved


def _save_fold_model(model, fold_model_path):
    joblib.dump(model, fold_model_path, compress=0)


def run_oof_shap_inprocess_optimized(
    X_all,
    y_all,
    lat_idx_evt,
    lon_idx_evt,
    feat_names,
    nlat,
    nlon,
    LAT,
    LON,
    max_per_fold=30000,
    batch_size=20000,
):
    p = len(feat_names)

    kf_local = KFold(n_splits=N_SPLITS, shuffle=KFOLD_SHUFFLE, random_state=SEED)
    fold_tests = []

    for fold_id, (_, te_idx) in enumerate(kf_local.split(X_all), start=1):
        te_idx = np.asarray(te_idx, dtype=np.int64)

        if max_per_fold and len(te_idx) > max_per_fold:
            rng = np.random.default_rng(SEED + 1000 + fold_id)
            te_idx = rng.choice(te_idx, size=max_per_fold, replace=False)

        fold_tests.append((fold_id, te_idx))

    n_total = int(sum(len(te) for _, te in fold_tests))

    print("\n" + "=" * 70)
    print("STEP 2: OOF SHAP (in-process TreeSHAP | fold-level checkpoint | optimized)")
    print("=" * 70)
    print(f"  folds={len(fold_tests)} | total_oof_shap_events={n_total:,} | p={p}")
    print(f"  batch_size={batch_size} | shap feature_perturbation=tree_path_dependent")

    _saved_env = {}
    thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

    for k in thread_keys:
        _saved_env[k] = os.environ.get(k)
        os.environ[k] = "1"

    try:
        for fold_id, te_idx in fold_tests:
            paths = _fold_cache_paths(fold_id)
            n_f = int(len(te_idx))

            fold_done = False
            if all(os.path.exists(paths[k]) for k in ("shap", "la", "lo", "meta")):
                try:
                    with open(paths["meta"], "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    if int(meta.get("n_samples", -1)) == n_f and int(meta.get("n_features", -1)) == p:
                        fold_done = True

                except Exception:
                    fold_done = False

            if fold_done:
                print(f"  [Fold {fold_id:02d}] ✅ checkpoint exists, skip")
                continue

            fold_model_path = os.path.join(FOLD_MODEL_DIR, f"rf_fold{fold_id}.joblib")

            print(f"  [Fold {fold_id:02d}] Loading model from {os.path.basename(fold_model_path)} ...")

            t_load = time.time()
            model = _load_fold_model(fold_model_path)

            print(f"  [Fold {fold_id:02d}] Model loaded ({time.time() - t_load:.1f}s) | n_estimators={model.n_estimators}")

            explainer = shap_lib.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent",
            )

            X_te = np.ascontiguousarray(X_all[te_idx], dtype=np.float32)
            shap_fold = np.empty((n_f, p), dtype=np.float32)

            t0 = time.time()

            for s in range(0, n_f, batch_size):
                e = min(s + batch_size, n_f)
                sv = explainer.shap_values(X_te[s:e], check_additivity=False)
                shap_fold[s:e] = np.asarray(sv, dtype=np.float32)

                elapsed = max(time.time() - t0, 1e-9)
                rate = e / elapsed
                remaining = (n_f - e) / rate if rate > 0 else float("inf")

                print(f"    [{fold_id:02d}] batch {s:,}–{e:,}/{n_f:,} | {rate:,.0f} samp/s | eta {remaining / 60:.1f} min")

            atomic_save_npy(paths["shap"], shap_fold)
            atomic_save_npy(paths["la"], np.asarray(lat_idx_evt[te_idx], dtype=np.int32))
            atomic_save_npy(paths["lo"], np.asarray(lon_idx_evt[te_idx], dtype=np.int32))

            atomic_write_json(
                {
                    "fold": int(fold_id),
                    "n_samples": int(n_f),
                    "n_features": int(p),
                    "updated": pd.Timestamp.now().isoformat(),
                    "batch_size": int(batch_size),
                    "max_per_fold": int(max_per_fold) if max_per_fold is not None else None,
                },
                paths["meta"],
            )

            elapsed_total = time.time() - t0

            print(f"  [Fold {fold_id:02d}] ✅ SHAP done | {elapsed_total:.1f}s | avg {n_f / elapsed_total:,.0f} samp/s")

            del model, explainer, shap_fold, X_te
            gc.collect()

    finally:
        for k, v in _saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    shap_list = []
    la_list = []
    lo_list = []

    for fold_id, te_idx in fold_tests:
        paths = _fold_cache_paths(fold_id)

        if not all(os.path.exists(paths[k]) for k in ("shap", "la", "lo", "meta")):
            raise FileNotFoundError(f"Missing SHAP fold cache for fold {fold_id}: {paths}")

        shap_arr = np.load(paths["shap"])
        la_arr = np.load(paths["la"])
        lo_arr = np.load(paths["lo"])

        if shap_arr.shape != (len(te_idx), p):
            raise ValueError(f"Fold {fold_id} SHAP shape mismatch: {shap_arr.shape} vs {(len(te_idx), p)}")

        shap_list.append(np.asarray(shap_arr, dtype=np.float32, order="C"))
        la_list.append(np.asarray(la_arr, dtype=np.int32))
        lo_list.append(np.asarray(lo_arr, dtype=np.int32))

    shap_all = np.concatenate(shap_list, axis=0)
    la_all = np.concatenate(la_list, axis=0)
    lo_all = np.concatenate(lo_list, axis=0)

    meta_oof = {
        "la_idx": la_all,
        "lo_idx": lo_all,
        "nlat": int(nlat),
        "nlon": int(nlon),
    }

    return shap_all, meta_oof


def main():
    print("=" * 88)
    print("RF + OOF-SHAP | LAND-ONLY | Random KFold | IN-PROCESS SHAP (optimized v2 | local cache)")
    print("=" * 88)
    print(f"REMOTE_OUT_DIR = {REMOTE_OUT_DIR}")
    print(f"LOCAL_OUT_DIR  = {LOCAL_OUT_DIR}")
    print(f"REMOTE_FOLD_MODEL_DIR = {REMOTE_FOLD_MODEL_DIR}")
    print(f"LOCAL_FOLD_MODEL_DIR  = {LOCAL_FOLD_MODEL_DIR}")

    print("\nSyncing remote fold models to local ...")
    sync_fold_models_from_remote()

    for pth in [AI_PATH, SOIL_PATH]:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing static file: {pth}")

    print("\nLoading static maps ...")

    ai_ds = xr.open_dataset(AI_PATH)

    if AI_VAR not in ai_ds:
        if len(ai_ds.data_vars) == 1:
            only = list(ai_ds.data_vars)[0]
            ai_da = ai_ds[only]
        else:
            raise KeyError(f"AI var '{AI_VAR}' not found")
    else:
        ai_da = ai_ds[AI_VAR]

    ai_da = ensure_lat_lon(ai_da)

    soil_ds = xr.open_dataset(SOIL_PATH)
    soil_ds = soil_ds.rename(
        {
            d: ("lat" if d.lower() == "latitude" else "lon")
            for d in soil_ds.dims
            if d.lower() in ["latitude", "longitude"]
        }
    ) if any(d.lower() in ["latitude", "longitude"] for d in soil_ds.dims) else soil_ds

    soil = {}

    for k, vname in SOIL_VARS.items():
        if vname not in soil_ds:
            raise KeyError(f"Soil var '{vname}' not found")
        soil[k] = ensure_lat_lon(soil_ds[vname])

    LAT = ai_da["lat"].values
    LON = ai_da["lon"].values
    nlat = len(LAT)
    nlon = len(LON)

    print(f"Static grid: nlat={nlat}, nlon={nlon}")

    c1 = soil["CACO3"].values
    c2 = soil["Clay"].values
    c3 = soil["OC"].values
    c4 = soil["Sand"].values

    finite_any = np.isfinite(c1) | np.isfinite(c2) | np.isfinite(c3) | np.isfinite(c4)

    if LANDMASK_MODE.lower() == "finite":
        landmask_grid = finite_any
    else:
        mag = (
            np.nan_to_num(np.abs(c1))
            + np.nan_to_num(np.abs(c2))
            + np.nan_to_num(np.abs(c3))
            + np.nan_to_num(np.abs(c4))
        )
        landmask_grid = finite_any & (mag > SOIL_EPS)

    print(f"Land fraction: {landmask_grid.mean():.1%}")

    file_map_mean_z = build_file_map(IN_DIR_ZSCORE_MEAN, FILE_MUST_CONTAIN_MEAN)
    file_map_raw = build_file_map(IN_DIR_EVENT_RAW, FILE_MUST_CONTAIN_MEAN)
    file_map_max_z = build_file_map(IN_DIR_ZSCORE_MAX, FILE_MUST_CONTAIN_MAX)

    if not file_map_mean_z:
        raise RuntimeError(f"No EVENTMEAN ZScore files in {IN_DIR_ZSCORE_MEAN}")
    if not file_map_raw:
        raise RuntimeError(f"No EVENTMEAN RAW files in {IN_DIR_EVENT_RAW}")

    common_keys_mean = sorted(set(file_map_mean_z.keys()) & set(file_map_raw.keys()))

    if not common_keys_mean:
        raise RuntimeError("No matched raw/ZScore EVENTMEAN files.")

    print("\nMatched EVENTMEAN files:")
    for k in common_keys_mean:
        print("  -", k)

    print("\nMatched EVENTMAX ZScore files:")
    for k in sorted(file_map_max_z.keys()):
        print("  -", k)

    event_id_ref = None
    meta_ref = None
    X_cols = []
    feat_names = []
    y_raw = None
    shared_vars_loaded = set()

    for k in common_keys_mean:
        fp_z = file_map_mean_z[k]
        fp_r = file_map_raw[k]

        ds_z = xr.open_dataset(fp_z)
        ds_r = xr.open_dataset(fp_r)

        if "event" not in ds_z.sizes:
            raise ValueError(f"ZScore lacks 'event' dim: {fp_z}")
        if "event_id" not in ds_z:
            raise ValueError(f"ZScore lacks event_id: {fp_z}")
        if "dod_event_mean" not in ds_r:
            raise ValueError(f"RAW lacks dod_event_mean: {fp_r}")
        if "event_id" not in ds_r:
            raise ValueError(f"RAW lacks event_id: {fp_r}")

        ev_z = _as_int_eventid(ds_z["event_id"].values)
        ev_r = _as_int_eventid(ds_r["event_id"].values)

        map_z = {int(v): i for i, v in enumerate(ev_z)}
        map_r = {int(v): i for i, v in enumerate(ev_r)}

        common_local = sorted(set(map_z.keys()) & set(map_r.keys()))

        if len(common_local) == 0:
            raise ValueError(f"No common event_id between RAW and ZScore pair\n  RAW={fp_r}\n  Z  ={fp_z}")

        idx_z_local = np.asarray([map_z[i] for i in common_local], dtype=np.int64)
        idx_r_local = np.asarray([map_r[i] for i in common_local], dtype=np.int64)

        ds_z = ds_z.isel(event=xr.DataArray(idx_z_local, dims="event"))
        ds_r = ds_r.isel(event=xr.DataArray(idx_r_local, dims="event"))

        ev = _as_int_eventid(ds_z["event_id"].values)

        if event_id_ref is None:
            event_id_ref = ev.copy()

            required_meta = ["event_id", "lat_idx", "lon_idx", "lat", "lon"]

            for m0 in required_meta:
                if m0 not in ds_z:
                    raise KeyError(f"Missing required meta '{m0}' in {fp_z}")

            meta_ref = {
                k0: ds_z[k0].values
                for k0 in ["event_id", "lat_idx", "lon_idx", "lat", "lon"]
            }

            y_raw = ds_r["dod_event_mean"].values.astype(np.float32, copy=False)

        else:
            ref_ids = _as_int_eventid(event_id_ref)
            cur_ids = ev

            ref_map = {int(v): i for i, v in enumerate(ref_ids)}
            cur_map = {int(v): i for i, v in enumerate(cur_ids)}

            common = sorted(set(ref_map.keys()) & set(cur_map.keys()))

            if len(common) == 0:
                raise ValueError(f"No common event_id with previous files. Problem pair\n  RAW={fp_r}\n  Z={fp_z}")

            ref_idx = np.asarray([ref_map[i] for i in common], dtype=np.int64)
            cur_idx = np.asarray([cur_map[i] for i in common], dtype=np.int64)

            event_id_ref = event_id_ref[ref_idx]

            for kk in meta_ref:
                meta_ref[kk] = meta_ref[kk][ref_idx]

            y_raw = y_raw[ref_idx]

            if X_cols:
                X_cols = [col[ref_idx] for col in X_cols]

            y_cur = ds_r["dod_event_mean"].values[cur_idx].astype(np.float32, copy=False)
            md = float(np.nanmax(np.abs(y_cur - y_raw)))

            if md > 1e-6:
                print(f"  ⚠️ Y differs across raw files for key={k} (maxdiff={md:.2e}); using reference Y")

            ds_z = ds_z.isel(event=xr.DataArray(cur_idx, dims="event"))
            ds_r = ds_r.isel(event=xr.DataArray(cur_idx, dims="event"))

        added = 0

        for v in DEDUP_SHARED_VARS:
            if v in shared_vars_loaded:
                continue
            if v in ds_z and (ds_z[v].dims == ("event",)):
                col = ds_z[v].values.astype(np.float32, copy=False)
                feat_names.append(v)
                X_cols.append(col)
                shared_vars_loaded.add(v)
                added += 1

        for v in ds_z.data_vars:
            if v in META_VARS or v in DEDUP_SHARED_VARS:
                continue

            da = ds_z[v]

            if da.dims != ("event",):
                continue
            if not should_use_predictor(v):
                continue

            name = choose_feature_name(v, k, feat_names, DEDUP_SHARED_VARS)
            col = da.values.astype(np.float32, copy=False)

            X_cols.append(col)
            feat_names.append(name)
            added += 1

        ds_z.close()
        ds_r.close()

        print(f"  ✅ EVENTMEAN: {k} | +{added} feats | total={len(feat_names)} | n_events={len(event_id_ref):,}")

    if file_map_max_z:
        print("\nLoading extra EVENTMAX predictors .")

        for k in sorted(file_map_max_z.keys()):
            fp_zmax = file_map_max_z[k]
            ds_mx = xr.open_dataset(fp_zmax)

            if "event" not in ds_mx.sizes:
                raise ValueError(f"EVENTMAX lacks event dim: {fp_zmax}")
            if "event_id" not in ds_mx:
                raise ValueError(f"EVENTMAX lacks event_id: {fp_zmax}")

            ds_mx, ref_idx, cur_idx, common_ids = align_dataset_to_ref_event_ids(ds_mx, event_id_ref)

            if len(common_ids) < len(event_id_ref):
                event_id_ref = event_id_ref[ref_idx]

                for kk in meta_ref:
                    meta_ref[kk] = meta_ref[kk][ref_idx]

                y_raw = y_raw[ref_idx]

                if X_cols:
                    X_cols = [col[ref_idx] for col in X_cols]

            added = 0

            for v in ds_mx.data_vars:
                if v in META_VARS:
                    continue

                da = ds_mx[v]

                if da.dims != ("event",):
                    continue
                if not should_use_predictor(v):
                    continue

                name = choose_feature_name(v, k, feat_names, DEDUP_SHARED_VARS)

                if name in feat_names:
                    print(f"    - skip duplicate: {name}")
                    continue

                X_cols.append(da.values.astype(np.float32, copy=False))
                feat_names.append(name)
                added += 1

            ds_mx.close()

            print(f"  ✅ EVENTMAX: {k} | +{added} feats | total={len(feat_names)} | n_events={len(event_id_ref):,}")

    lat_idx = np.asarray(meta_ref["lat_idx"]).astype(np.int32, copy=False)
    lon_idx = np.asarray(meta_ref["lon_idx"]).astype(np.int32, copy=False)
    lat_evt_full = np.asarray(meta_ref["lat"]).astype(np.float64, copy=False)
    lon_evt_full = np.asarray(meta_ref["lon"]).astype(np.float64, copy=False)

    if lat_idx.min() < 0 or lat_idx.max() >= nlat or lon_idx.min() < 0 or lon_idx.max() >= nlon:
        raise ValueError(
            f"Event lat_idx/lon_idx out of static grid range. "
            f"lat_idx=[{lat_idx.min()},{lat_idx.max()}] nlat={nlat}, "
            f"lon_idx=[{lon_idx.min()},{lon_idx.max()}] nlon={nlon}"
        )

    ai_s = ai_da.values[lat_idx, lon_idx].astype(np.float32, copy=False)
    caco3_s = soil["CACO3"].values[lat_idx, lon_idx].astype(np.float32, copy=False)
    clay_s = soil["Clay"].values[lat_idx, lon_idx].astype(np.float32, copy=False)
    oc_s = soil["OC"].values[lat_idx, lon_idx].astype(np.float32, copy=False)
    sand_s = soil["Sand"].values[lat_idx, lon_idx].astype(np.float32, copy=False)

    X_cols.extend([ai_s, caco3_s, clay_s, oc_s, sand_s])
    feat_names.extend(["AI", "CACO3", "Clay", "OC", "Sand"])

    feature_name_set = set(feat_names)

    FEATURE_GROUPS_LOCAL = {
        g: [f for f in feats if f in feature_name_set or f in STATIC_VARS]
        for g, feats in FEATURE_GROUPS.items()
    }

    X_all = np.column_stack(X_cols).astype(np.float32, copy=False)
    y_all = y_raw.astype(np.float32, copy=False)

    print(f"\nBuilt EVENT table: n_events={len(y_all):,} | n_features={X_all.shape[1]}")
    print("\nFINAL X FEATURES:")

    for i, f in enumerate(feat_names, 1):
        print(f"{i:02d}  {f}")

    land_evt_full = landmask_grid[lat_idx, lon_idx].astype(bool)
    m = land_evt_full & np.isfinite(y_all) & np.isfinite(X_all).all(axis=1)

    X_all = X_all[m]
    y_all = y_all[m]
    lat_evt = lat_evt_full[m]
    lon_evt = lon_evt_full[m]
    lat_idx_evt = lat_idx[m]
    lon_idx_evt = lon_idx[m]

    print(f"\nAfter LAND-ONLY + dropna: {len(y_all):,}/{len(m):,} ({100 * len(y_all) / len(m):.1f}%)")

    if len(y_all) < 5000:
        raise ValueError(f"Too few events after filtering: {len(y_all)} (<5000). Check NaNs / indices.")

    if SAVE_XY_FOR_PLOTTING:
        print("\nSaving filtered X_all / y_all for plotting/debug .")

        X_save = np.ascontiguousarray(X_all, dtype=np.float32)
        y_save = np.ascontiguousarray(y_all, dtype=np.float32)

        atomic_save_npy(X_ALL_NPY, X_save)
        atomic_save_npy(Y_ALL_NPY, y_save)
        atomic_write_json(list(feat_names), FEAT_JSON)

        tmp_npz = META_NPZ + ".tmp.npz"

        np.savez_compressed(
            tmp_npz,
            lat_idx=lat_idx_evt.astype(np.int32, copy=False),
            lon_idx=lon_idx_evt.astype(np.int32, copy=False),
            nlat=np.int32(nlat),
            nlon=np.int32(nlon),
            lat=np.asarray(LAT),
            lon=np.asarray(LON),
            event_lat=lat_evt.astype(np.float32, copy=False),
            event_lon=lon_evt.astype(np.float32, copy=False),
            landmask_mode=np.array([LANDMASK_MODE], dtype="S32"),
        )

        os.replace(tmp_npz, META_NPZ)

        print(f"  ✅ X_all saved: {X_ALL_NPY} | shape={X_save.shape} dtype={X_save.dtype}")
        print(f"  ✅ y_all saved: {Y_ALL_NPY} | shape={y_save.shape} dtype={y_save.dtype}")
        print(f"  ✅ feature_names saved: {FEAT_JSON} | n={len(feat_names)}")
        print(f"  ✅ event_meta saved: {META_NPZ}")

    if RF_MAX_SAMPLES is not None:
        RF_PARAMS["max_samples"] = float(RF_MAX_SAMPLES)
        print(f"[RF] max_samples={RF_MAX_SAMPLES} enabled (memory saver)")

    print("\n" + "=" * 70)
    print(f"STEP 1: RANDOM KFold CV (n_splits={N_SPLITS}, shuffle={KFOLD_SHUFFLE}, seed={SEED})")
    print(f"        Model: RandomForestRegressor | n_jobs={RF_PARAMS['n_jobs']} | backend=threading")
    print("=" * 70)

    kf = KFold(n_splits=N_SPLITS, shuffle=KFOLD_SHUFFLE, random_state=SEED)

    fold_rows = []
    y_oof = np.full(len(y_all), np.nan, dtype=np.float32)

    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_all, y_all), start=1):
        fold_model_path = os.path.join(FOLD_MODEL_DIR, f"rf_fold{fold_id}.joblib")

        if SKIP_IF_EXISTS and os.path.exists(fold_model_path):
            model = _load_fold_model(fold_model_path)
            pred_te = model.predict(X_all[te_idx])
            y_oof[te_idx] = pred_te.astype(np.float32, copy=False)

            mt = eval_metrics(y_all[te_idx], pred_te)

            mt.update(
                {
                    "fold": int(fold_id),
                    "seed": int(SEED + fold_id * 101),
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                }
            )

            fold_rows.append(mt)

            print(
                f"  [Fold {fold_id:02d}/{N_SPLITS}] ✅ SKIP TRAIN | "
                f"R²={mt['R2']:.3f} RMSE={mt['RMSE']:.3f} MAE={mt['MAE']:.3f} | "
                f"loaded={os.path.basename(fold_model_path)}"
            )

            del model, pred_te
            gc.collect()
            continue

        params_i = dict(RF_PARAMS, random_state=int(SEED + fold_id * 101))
        model = RandomForestRegressor(**params_i)

        with joblib.parallel_backend("threading"):
            model.fit(X_all[tr_idx], y_all[tr_idx])

        pred_te = model.predict(X_all[te_idx])
        y_oof[te_idx] = pred_te.astype(np.float32, copy=False)

        mt = eval_metrics(y_all[te_idx], pred_te)

        mt.update(
            {
                "fold": int(fold_id),
                "seed": int(params_i["random_state"]),
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
            }
        )

        fold_rows.append(mt)

        _save_fold_model(model, fold_model_path)
        mirror_local_model_to_remote(fold_model_path)

        print(
            f"  [Fold {fold_id:02d}/{N_SPLITS}] R²={mt['R2']:.3f} RMSE={mt['RMSE']:.3f} "
            f"MAE={mt['MAE']:.3f} | saved={os.path.basename(fold_model_path)}"
        )

        del model, pred_te
        gc.collect()

    df_fold = pd.DataFrame(fold_rows)

    df_fold.to_csv(FOLDS_CSV, index=False)

    atomic_write_json({"folds": fold_rows}, FOLDS_JSON)
    atomic_save_npy(OOF_PRED, y_oof.astype(np.float32, copy=False))
    atomic_save_npy(OOF_OBS, y_all.astype(np.float32, copy=False))

    r2_mean = float(np.nanmean(df_fold["R2"]))
    r2_std = float(np.nanstd(df_fold["R2"], ddof=1))
    rmse_mean = float(np.nanmean(df_fold["RMSE"]))
    rmse_std = float(np.nanstd(df_fold["RMSE"], ddof=1))
    mae_mean = float(np.nanmean(df_fold["MAE"]))
    mae_std = float(np.nanstd(df_fold["MAE"], ddof=1))

    atomic_write_json(
        {
            "summary": {
                "R2_mean": r2_mean,
                "R2_std": r2_std,
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "n_events": int(len(y_all)),
                "n_features": int(len(feat_names)),
                "rf_params": RF_PARAMS,
                "shap_method": "in-process TreeSHAP (tree_path_dependent)",
                "shap_batch_size": int(SHAP_BATCH_SIZE),
                "predictor_rule": "keep raw event_mean/event_max; drop anomaly/anom; exclude DD/CD",
                "remote_out_dir": REMOTE_OUT_DIR,
                "local_out_dir": LOCAL_OUT_DIR,
            },
            "folds": fold_rows,
        },
        CV_JSON,
    )

    print(
        f"\n📌 KFold CV: R²={r2_mean:.3f}±{r2_std:.3f} | "
        f"RMSE={rmse_mean:.3f}±{rmse_std:.3f} | MAE={mae_mean:.3f}±{mae_std:.3f}"
    )

    if OOF_SHAP_ENABLE:
        if SKIP_IF_EXISTS and os.path.exists(SHAP_OOF_NC):
            print(f"\n✅ SKIP OOF SHAP: exists -> {SHAP_OOF_NC}")
        else:
            shap_oof, meta_oof = run_oof_shap_inprocess_optimized(
                X_all,
                y_all,
                lat_idx_evt,
                lon_idx_evt,
                feat_names,
                nlat,
                nlon,
                LAT,
                LON,
                max_per_fold=OOF_SHAP_MAX_PER_FOLD,
                batch_size=SHAP_BATCH_SIZE,
            )

            print("\nAggregating OOF SHAP to maps.")

            ds_oof, cov_oof = aggregate_shap_to_maps(
                shap_values=shap_oof,
                meta=meta_oof,
                feature_names=feat_names,
                lat_vals=LAT,
                lon_vals=LON,
                feature_groups=FEATURE_GROUPS_LOCAL,
                group_method=DOMINANT_GROUP_METHOD,
                min_samples_dominant=MIN_SAMPLES_FOR_DOMINANT,
            )

            ds_oof.attrs.update(
                {
                    "description": "OOF SHAP | RF | TreeSHAP in-process (tree_path_dependent) | optimized fold checkpoint | local-cache",
                    "creation_date": pd.Timestamp.now().isoformat(),
                    "shap_method": "in-process TreeSHAP (no multiprocessing)",
                    "shap_batch_size": int(SHAP_BATCH_SIZE),
                    "coverage_stats_json": json.dumps(cov_oof, ensure_ascii=False),
                    "out_dir": OUT_DIR,
                    "remote_out_dir": REMOTE_OUT_DIR,
                    "local_out_dir": LOCAL_OUT_DIR,
                    "x_input_dir_mean": IN_DIR_ZSCORE_MEAN,
                    "x_input_dir_max": IN_DIR_ZSCORE_MAX,
                    "y_input_dir": IN_DIR_EVENT_RAW,
                    "y_definition": "dod_event_mean (RAW)",
                    "predictor_rule": "keep raw event_mean/event_max; drop anomaly/anom; exclude DD/CD",
                    "landmask_mode": LANDMASK_MODE,
                }
            )

            ds_oof = sanitize_dataset_for_netcdf(ds_oof)

            encoding = {
                "mean_shap": {"zlib": True, "complevel": 4, "dtype": "float32"},
                "mean_abs_shap": {"zlib": True, "complevel": 4, "dtype": "float32"},
                "frac_positive_shap": {"zlib": True, "complevel": 4, "dtype": "float32"},
                "sign_stability": {"zlib": True, "complevel": 4, "dtype": "float32"},
                "n_samples": {"zlib": True, "complevel": 4, "dtype": "int32"},
                "dominant_driver_index": {"zlib": True, "complevel": 4, "dtype": "int16"},
                "dominant_driver_name": {"zlib": True, "complevel": 4},
            }

            print("\nExporting OOF SHAP NetCDF.")

            tmp_nc = SHAP_OOF_NC + ".tmp.nc"

            ds_oof.to_netcdf(tmp_nc, encoding=encoding)
            os.replace(tmp_nc, SHAP_OOF_NC)

            print(f"  ✅ Saved: {SHAP_OOF_NC}")

    print("\nMirroring local results back to remote ...")

    mirror_results_back_to_remote()

    print("\n✅ COMPLETE")
    print(f"LOCAL OUT_DIR : {OUT_DIR}")
    print(f"REMOTE OUT_DIR: {REMOTE_OUT_DIR}")
    print(f"Saved X_all: {X_ALL_NPY if SAVE_XY_FOR_PLOTTING else '(disabled)'}")
    print(f"OOF SHAP NC: {SHAP_OOF_NC}")


if __name__ == "__main__":
    main()