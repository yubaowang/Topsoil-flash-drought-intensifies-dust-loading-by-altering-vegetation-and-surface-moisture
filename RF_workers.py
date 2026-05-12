#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import joblib

try:
    import shap
except Exception as e:
    raise ImportError(f"shap import failed in worker: {e}")

_G_EXPLAINER = None
_G_X = None
_G_Y = None
_G_P = None
_G_N = None
_G_FEATURE_NAMES = None


def _ensure_2d_shap_output(arr, expected_rows: int, expected_cols: int) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP ndim={arr.ndim}, shape={arr.shape}")

    if arr.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Unexpected SHAP shape={arr.shape}, expected={(expected_rows, expected_cols)}"
        )

    return np.asarray(arr, dtype=np.float32, order="C")


def mp_worker_init_spawn_from_path(
    fold_model_path: str,
    feature_names,
    x_memmap_path: str,
    y_memmap_path: str,
    n_rows: int,
):
    global _G_EXPLAINER, _G_X, _G_Y, _G_P, _G_N, _G_FEATURE_NAMES

    payload = joblib.load(fold_model_path)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    _G_FEATURE_NAMES = list(feature_names)
    _G_P = len(_G_FEATURE_NAMES)
    _G_N = int(n_rows)

    _G_X = np.memmap(
        x_memmap_path,
        dtype="float32",
        mode="r",
        shape=(_G_N, _G_P),
    )

    _G_Y = np.memmap(
        y_memmap_path,
        dtype="float32",
        mode="r+",
        shape=(_G_N, _G_P),
    )

    _G_EXPLAINER = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )


def mp_worker_task(task):
    global _G_EXPLAINER, _G_X, _G_Y, _G_P

    s, e = int(task[0]), int(task[1])
    X_batch = np.asarray(_G_X[s:e], dtype=np.float32)

    sv = _G_EXPLAINER.shap_values(X_batch, check_additivity=False)
    sv = _ensure_2d_shap_output(
        sv,
        expected_rows=e - s,
        expected_cols=_G_P,
    )

    _G_Y[s:e, :] = sv
    _G_Y.flush()

    return s, e