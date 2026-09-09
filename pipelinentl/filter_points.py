#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alternative spatio-temporal filtering for ISS timelapse GCPs (v6).

Drop-in-oriented replacement for pipelinentl/filter_points.py.

Core idea
---------
1. Each fixed image-grid point (sourceX, sourceY) is treated as a track.
2. A LOCAL robust temporal model predicts lon/lat at each frame.
   - actual frame IDs are used as the time axis, so missing IDs remain gaps;
   - observed points are evaluated leave-one-out, so a bad point cannot pull
     its own prediction toward itself.
3. For each frame, temporal residual VECTORS (east/north, km) are compared
   spatially across neighbouring grid points.
   - a coherent displacement shared by nearby points is treated as a frame/
     regional deformation, not as many independent outliers;
   - isolated deviations from that local spatial consensus are outliers.
4. An observed outlier is replaced by:

       temporal_prediction + local_spatial_consensus

5. Missing points are synthesized only for STRUCTURAL tracks: tracks present in
   a strict majority of frames. Sporadic/minority tracks are never extended into
   frames where they were absent.

Outputs
-------
- <mission>-E-<ID>.points files, compatible with the downstream georeferencing.
- temporal_frame_summary.csv
- temporal_track_summary.csv
- spatiotemporal_point_qc.csv
- optional diagnostic plots/overlays.

Important behavioural rules
---------------------------
- Structural tracks (> majority_track_coverage) may have short internal gaps filled
  and observed outliers replaced.
- Minority/sporadic tracks are never synthesized where absent.
- Unresolved observed outliers are dropped by default.
- The final geographic BallTree neighbour filter is enabled by default for an
  additional conservative spatial validity check.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQUATOR = 111.320

REQUIRED_COLUMNS = {"mapX", "mapY", "sourceX", "sourceY"}
BASE_OUTPUT_COLUMNS = [
    "mapX", "mapY", "sourceX", "sourceY", "enable", "dX", "dY", "residual",
]


@dataclass
class FrameInfo:
    pos: int
    frame_id: int
    input_path: Path
    input_name: str
    output_name: str
    input_count: int = 0
    kept_count: int = 0
    replaced_count: int = 0
    filled_count: int = 0
    unresolved_count: int = 0
    post_spatial_removed: int = 0


@dataclass
class TrackModel:
    key: str
    sourceX: float
    sourceY: float
    observed_count: int
    coverage: float
    pred_lon: np.ndarray
    pred_lat: np.ndarray
    temporal_dx_km: np.ndarray
    temporal_dy_km: np.ndarray
    temporal_mag_km: np.ndarray


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

PROGRESS_STEP_PERCENT = 5.0
PROGRESS_MAX_SILENCE_S = 30.0


class ProgressReporter:
    """Small dependency-free progress reporter suitable for pipeline logs.

    It reports at percentage milestones and also after a maximum period of silence,
    so long individual phases still show that the process is alive.
    """

    def __init__(self, label: str, total: int, step_percent: float | None = None) -> None:
        self.label = str(label)
        self.total = max(int(total), 0)
        self.step_percent = float(PROGRESS_STEP_PERCENT if step_percent is None else step_percent)
        self.step_percent = max(self.step_percent, 0.1)
        self.start_time = time.monotonic()
        self.last_print_time = self.start_time
        self.next_percent = 0.0
        self.last_done = -1
        self.update(0, force=True)

    def update(self, done: int, force: bool = False) -> None:
        done = max(0, min(int(done), self.total)) if self.total else max(0, int(done))
        now = time.monotonic()
        if self.total > 0:
            percent = 100.0 * done / self.total
        else:
            percent = 100.0 if done else 0.0

        due_percent = percent + 1e-12 >= self.next_percent
        due_time = (now - self.last_print_time) >= PROGRESS_MAX_SILENCE_S
        finished = self.total > 0 and done >= self.total

        if not (force or due_percent or due_time or finished):
            return
        if done == self.last_done and not force and not due_time:
            return

        elapsed = now - self.start_time
        if self.total > 0:
            print(
                f"[progress] {self.label}: {done}/{self.total} "
                f"({percent:5.1f}%) | elapsed {elapsed:6.1f}s",
                flush=True,
            )
        else:
            print(f"[progress] {self.label}: {done} | elapsed {elapsed:6.1f}s", flush=True)

        self.last_done = done
        self.last_print_time = now
        while self.next_percent <= percent + 1e-12:
            self.next_percent += self.step_percent

    def finish(self) -> None:
        if self.last_done != self.total:
            self.update(self.total, force=True)


def wrap_lon_deg(lon: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(lon, dtype=float)
    out = (arr + 180.0) % 360.0 - 180.0
    if np.isscalar(lon):
        return float(out)
    return out


def extract_id_from_point_filename(name: str) -> Optional[int]:
    m = re.search(r"[A-Za-z0-9]+-E-(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{5,8})", name)
    return int(m.group(1)) if m else None


def make_source_key(source_x: Any, source_y: Any, decimals: int) -> str:
    sx = round(float(source_x), decimals)
    sy = round(float(source_y), decimals)
    return f"{sx:.{decimals}f}|{sy:.{decimals}f}"


def parse_source_key(key: str) -> Tuple[float, float]:
    sx, sy = key.split("|", 1)
    return float(sx), float(sy)


def coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def read_points_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="M")


def robust_scale(values: np.ndarray, floor: float = 1e-9) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(floor)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(1.4826 * mad, float(floor))


def robust_threshold(
    values: np.ndarray,
    mode: str,
    sigma: float,
    absolute_km: float,
    min_threshold_km: float,
) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("inf")

    med = float(np.median(vals))
    scale = robust_scale(vals, floor=0.0)
    sigma_thr = med + float(sigma) * scale

    if mode == "sigma":
        thr = sigma_thr
    elif mode == "absolute":
        thr = float(absolute_km)
    elif mode == "hybrid":
        thr = max(float(absolute_km), sigma_thr)
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")

    return max(float(min_threshold_km), float(thr))


def lonlat_delta_to_km(
    lon_obs: float,
    lat_obs: float,
    lon_pred: float,
    lat_pred: float,
) -> Tuple[float, float]:
    dlon = float(wrap_lon_deg(lon_obs - lon_pred))
    mean_lat = 0.5 * (float(lat_obs) + float(lat_pred))
    dx = dlon * KM_PER_DEG_LON_EQUATOR * math.cos(math.radians(mean_lat))
    dy = (float(lat_obs) - float(lat_pred)) * KM_PER_DEG_LAT
    return dx, dy


def add_km_offset_to_lonlat(
    lon: float,
    lat: float,
    dx_km: float,
    dy_km: float,
) -> Tuple[float, float]:
    new_lat = float(lat) + float(dy_km) / KM_PER_DEG_LAT
    denom = KM_PER_DEG_LON_EQUATOR * max(abs(math.cos(math.radians(new_lat))), 1e-6)
    new_lon = wrap_lon_deg(float(lon) + float(dx_km) / denom)
    return float(new_lon), float(new_lat)


# -----------------------------------------------------------------------------
# Loading and track construction
# -----------------------------------------------------------------------------

def load_timelapse_points(
    input_folder: str,
    start_id: int,
    end_id: int,
    mission: str,
    input_glob: str,
    source_round_decimals: int,
) -> Tuple[List[FrameInfo], pd.DataFrame, List[str]]:
    input_dir = Path(input_folder)
    paths = []
    for p in input_dir.glob(input_glob):
        sid = extract_id_from_point_filename(p.name)
        if sid is not None and start_id <= sid <= end_id:
            paths.append((sid, p))
    paths.sort(key=lambda x: x[0])

    frames: List[FrameInfo] = []
    rows: List[pd.DataFrame] = []
    first_input_columns: List[str] = []

    progress = ProgressReporter("loading .points files", len(paths))
    for pos, (frame_id, path) in enumerate(paths):
        try:
            df = read_points_file(path)
        except Exception as exc:
            print(f"ERROR reading {path}: {exc}")
            progress.update(pos + 1)
            continue

        if not REQUIRED_COLUMNS.issubset(df.columns):
            print(f"WARNING: {path.name} lacks {sorted(REQUIRED_COLUMNS)}; skipped")
            progress.update(pos + 1)
            continue

        if not first_input_columns:
            first_input_columns = list(df.columns)

        df = coerce_numeric_columns(df, REQUIRED_COLUMNS | {"enable", "dX", "dY", "residual"})
        finite = np.isfinite(df[["mapX", "mapY", "sourceX", "sourceY"]].to_numpy(dtype=float)).all(axis=1)
        df = df.loc[finite].copy()

        info = FrameInfo(
            pos=len(frames),
            frame_id=int(frame_id),
            input_path=path,
            input_name=path.name,
            output_name=f"{mission}-E-{frame_id}.points",
            input_count=len(df),
        )
        frames.append(info)

        df["__frame_pos"] = info.pos
        df["__frame_id"] = int(frame_id)
        df["__row_index"] = np.arange(len(df), dtype=int)
        df["__source_key"] = [
            make_source_key(x, y, source_round_decimals)
            for x, y in zip(df["sourceX"], df["sourceY"])
        ]
        rows.append(df)
        progress.update(pos + 1)

    progress.finish()
    if not rows:
        return frames, pd.DataFrame(), first_input_columns

    return frames, pd.concat(rows, ignore_index=True), first_input_columns


def discover_output_columns(first_input_columns: Sequence[str], add_qc_columns: bool) -> List[str]:
    cols = [c for c in first_input_columns if not str(c).startswith("__")]
    for col in BASE_OUTPUT_COLUMNS:
        if col not in cols:
            cols.append(col)
    if add_qc_columns:
        for col in [
            "temporal_status",
            "temporal_residual_km",
            "spatial_residual_km",
            "spatial_consensus_dx_km",
            "spatial_consensus_dy_km",
            "local_presence_fraction",
            "local_presence_count",
            "local_presence_total",
        ]:
            if col not in cols:
                cols.append(col)
    return cols


# -----------------------------------------------------------------------------
# Robust LOCAL temporal model
# -----------------------------------------------------------------------------

def design_matrix(z: np.ndarray, degree: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.column_stack([z ** k for k in range(degree + 1)])


def weighted_lstsq(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
    try:
        sw = np.sqrt(np.clip(w, 1e-12, np.inf))
        Xw = X * sw[:, None]
        yw = y * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return beta
    except Exception:
        return None


def robust_local_poly_predict(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x0: float,
    degree: int,
    min_points: int,
    max_neighbors: int,
    exclude_x: Optional[float] = None,
    huber_k: float = 1.5,
    max_iter: int = 8,
) -> float:
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    if exclude_x is not None:
        mask &= np.abs(x_obs - float(exclude_x)) > 1e-9

    x = x_obs[mask]
    y = y_obs[mask]
    if len(x) < min_points:
        return float("nan")

    order_idx = np.argsort(np.abs(x - float(x0)))
    if max_neighbors > 0:
        order_idx = order_idx[:max(max_neighbors, min_points)]
    x = x[order_idx]
    y = y[order_idx]

    degree = int(min(max(1, degree), len(x) - 1))
    scale_x = max(float(np.max(np.abs(x - x0))), 1.0)
    z = (x - float(x0)) / scale_x
    X = design_matrix(z, degree)

    # Distance weights make the model genuinely local.
    dist = np.abs(z)
    w_dist = 1.0 / (1.0 + dist ** 2)
    w = w_dist.copy()

    beta = weighted_lstsq(X, y, w)
    if beta is None:
        return float("nan")

    for _ in range(max_iter):
        resid = y - X @ beta
        s = robust_scale(resid, floor=1e-12)
        u = np.abs(resid) / (huber_k * s)
        w_huber = np.ones_like(u)
        large = u > 1.0
        w_huber[large] = 1.0 / u[large]
        new_w = w_dist * w_huber
        new_beta = weighted_lstsq(X, y, new_w)
        if new_beta is None:
            break
        if np.allclose(new_beta, beta, rtol=1e-7, atol=1e-10):
            beta = new_beta
            break
        beta = new_beta

    # Because z=0 at x0, prediction is the intercept.
    return float(beta[0])


def unwrap_longitudes_for_track(x: np.ndarray, lon: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    out = np.full(len(lon), np.nan, dtype=float)
    vals = np.asarray(lon, dtype=float)[order]
    finite = np.isfinite(vals)
    if finite.any():
        unwrapped = np.degrees(np.unwrap(np.radians(vals[finite])))
        tmp = np.full(len(vals), np.nan, dtype=float)
        tmp[finite] = unwrapped
        out[order] = tmp
    return out


def build_temporal_models(
    all_df: pd.DataFrame,
    frames: List[FrameInfo],
    temporal_order: int,
    min_track_points: int,
    min_track_coverage: float,
    temporal_neighbors: int,
) -> Dict[str, TrackModel]:
    frame_ids = np.asarray([f.frame_id for f in frames], dtype=float)
    n_frames = len(frames)
    models: Dict[str, TrackModel] = {}

    grouped_tracks = all_df.groupby("__source_key", sort=True)
    progress = ProgressReporter("temporal track models", grouped_tracks.ngroups)
    for track_no, (key, g) in enumerate(grouped_tracks, start=1):
        sx, sy = parse_source_key(key)
        raw_lon = np.full(n_frames, np.nan, dtype=float)
        raw_lat = np.full(n_frames, np.nan, dtype=float)

        for fi, gf in g.groupby("__frame_pos", sort=True):
            row = gf.iloc[0]
            raw_lon[int(fi)] = float(row["mapX"])
            raw_lat[int(fi)] = float(row["mapY"])

        observed = np.isfinite(raw_lon) & np.isfinite(raw_lat)
        n_obs = int(observed.sum())
        coverage = float(n_obs) / max(n_frames, 1)

        pred_lon = np.full(n_frames, np.nan, dtype=float)
        pred_lat = np.full(n_frames, np.nan, dtype=float)

        if n_obs >= min_track_points:
            x_obs = frame_ids[observed]
            lon_unwrapped_full = unwrap_longitudes_for_track(frame_ids, raw_lon)
            lon_obs_u = lon_unwrapped_full[observed]
            lat_obs = raw_lat[observed]

            for i, x0 in enumerate(frame_ids):
                exclude = x0 if observed[i] else None
                plon_u = robust_local_poly_predict(
                    x_obs, lon_obs_u, x0,
                    degree=min(int(temporal_order), 2),
                    min_points=min_track_points,
                    max_neighbors=temporal_neighbors,
                    exclude_x=exclude,
                )
                plat = robust_local_poly_predict(
                    x_obs, lat_obs, x0,
                    degree=min(int(temporal_order), 2),
                    min_points=min_track_points,
                    max_neighbors=temporal_neighbors,
                    exclude_x=exclude,
                )
                if np.isfinite(plon_u) and np.isfinite(plat):
                    pred_lon[i] = float(wrap_lon_deg(plon_u))
                    pred_lat[i] = float(plat)

        dx = np.full(n_frames, np.nan, dtype=float)
        dy = np.full(n_frames, np.nan, dtype=float)
        mag = np.full(n_frames, np.nan, dtype=float)
        for i in np.where(observed & np.isfinite(pred_lon) & np.isfinite(pred_lat))[0]:
            ddx, ddy = lonlat_delta_to_km(raw_lon[i], raw_lat[i], pred_lon[i], pred_lat[i])
            dx[i], dy[i] = ddx, ddy
            mag[i] = float(np.hypot(ddx, ddy))

        models[key] = TrackModel(
            key=key,
            sourceX=sx,
            sourceY=sy,
            observed_count=n_obs,
            coverage=coverage,
            pred_lon=pred_lon,
            pred_lat=pred_lat,
            temporal_dx_km=dx,
            temporal_dy_km=dy,
            temporal_mag_km=mag,
        )
        progress.update(track_no)

    progress.finish()
    return models


# -----------------------------------------------------------------------------
# Frame-level temporal health and targeted repair
# -----------------------------------------------------------------------------

def _lower_trimmed(values: np.ndarray, upper_quantile: float = 0.75) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 2:
        return vals
    q = float(np.quantile(vals, np.clip(upper_quantile, 0.50, 1.0)))
    trimmed = vals[vals <= q]
    return trimmed if len(trimmed) >= 2 else vals


def detect_corrupt_frames(
    frames: List[FrameInfo],
    models: Dict[str, TrackModel],
    min_points: int = 20,
    neighbour_radius: int = 6,
    min_median_km: float = 50.0,
    ratio: float = 3.0,
    sigma: float = 4.0,
    scale_floor_km: float = 5.0,
) -> Dict[int, dict]:
    """Detect coherent whole-frame georeferencing failures.

    A whole bad frame can be spatially self-consistent, so point-level spatial
    consensus cannot reject it.  Here the statistic is the median leave-one-out
    TEMPORAL residual across all valid tracks in the frame.
    """
    n = len(frames)
    med = np.full(n, np.nan, dtype=float)
    p75 = np.full(n, np.nan, dtype=float)
    p90 = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=int)

    for i in range(n):
        vals = np.asarray([
            tr.temporal_mag_km[i]
            for tr in models.values()
            if np.isfinite(tr.temporal_mag_km[i])
        ], dtype=float)
        if len(vals) >= int(min_points):
            counts[i] = len(vals)
            med[i] = float(np.median(vals))
            p75[i] = float(np.percentile(vals, 75))
            p90[i] = float(np.percentile(vals, 90))

    valid = np.isfinite(med) & (counts >= int(min_points))
    global_vals = _lower_trimmed(med[valid], 0.75)
    if len(global_vals):
        global_base = float(np.median(global_vals))
        global_scale = robust_scale(global_vals, floor=float(scale_floor_km))
    else:
        global_base = 0.0
        global_scale = float(scale_floor_km)

    frame_ids = np.asarray([f.frame_id for f in frames], dtype=int)
    positions = np.arange(n, dtype=int)
    out: Dict[int, dict] = {}

    for i, f in enumerate(frames):
        local_idx = (
            valid
            & (positions != i)
            & (np.abs(frame_ids - int(f.frame_id)) <= int(neighbour_radius))
        )
        local_vals = _lower_trimmed(med[local_idx], 0.75)
        if len(local_vals) >= 2:
            local_base = float(np.median(local_vals))
            local_scale = robust_scale(local_vals, floor=float(scale_floor_km))
        else:
            local_base = global_base
            local_scale = global_scale

        baseline = max(global_base, local_base)
        scale = max(float(scale_floor_km), global_scale, local_scale)
        threshold = max(
            float(min_median_km),
            float(ratio) * baseline,
            baseline + float(sigma) * scale,
        )
        corrupt = bool(valid[i] and med[i] > threshold)
        out[f.pos] = {
            "corrupt": corrupt,
            "n_temporal_residuals": int(counts[i]),
            "median_temporal_residual_km": float(med[i]) if np.isfinite(med[i]) else np.nan,
            "p75_temporal_residual_km": float(p75[i]) if np.isfinite(p75[i]) else np.nan,
            "p90_temporal_residual_km": float(p90[i]) if np.isfinite(p90[i]) else np.nan,
            "baseline_km": float(baseline),
            "threshold_km": float(threshold),
        }
    return out


def build_corrupt_frame_repairs(
    all_df: pd.DataFrame,
    frames: List[FrameInfo],
    frame_health: Dict[int, dict],
    temporal_order: int,
    min_track_points: int,
    temporal_neighbors: int,
    allow_extrapolation: bool = False,
) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Predict only corrupt frames from clean neighbouring observations.

    All frames classified corrupt are removed from the temporal anchor set.
    This is intentionally targeted: unlike rebuilding every track at every frame,
    cost scales mainly with number_of_corrupt_frames x number_of_tracks.
    """
    bad_positions = {
        f.pos for f in frames
        if bool(frame_health.get(f.pos, {}).get("corrupt", False))
    }
    if not bad_positions:
        return {}

    frame_ids = np.asarray([f.frame_id for f in frames], dtype=float)
    bad_mask = np.asarray([f.pos in bad_positions for f in frames], dtype=bool)
    repairs: Dict[Tuple[str, int], Tuple[float, float]] = {}
    grouped = all_df.groupby("__source_key", sort=True)
    progress = ProgressReporter("clean predictions for corrupt frames", grouped.ngroups)

    for track_no, (key, g) in enumerate(grouped, start=1):
        raw_lon = np.full(len(frames), np.nan, dtype=float)
        raw_lat = np.full(len(frames), np.nan, dtype=float)
        for fi, gf in g.groupby("__frame_pos", sort=True):
            row = gf.iloc[0]
            raw_lon[int(fi)] = float(row["mapX"])
            raw_lat[int(fi)] = float(row["mapY"])

        observed = np.isfinite(raw_lon) & np.isfinite(raw_lat)
        clean = observed & ~bad_mask
        if int(clean.sum()) < int(min_track_points):
            progress.update(track_no)
            continue

        x = frame_ids[clean]
        lon = raw_lon[clean]
        lat = raw_lat[clean]
        lon_u = np.degrees(np.unwrap(np.radians(lon)))

        for fi in bad_positions:
            x0 = frame_ids[int(fi)]
            if not allow_extrapolation:
                if not (np.any(x < x0) and np.any(x > x0)):
                    continue
            plon_u = robust_local_poly_predict(
                x, lon_u, x0,
                degree=min(int(temporal_order), 2),
                min_points=min_track_points,
                max_neighbors=temporal_neighbors,
                exclude_x=None,
            )
            plat = robust_local_poly_predict(
                x, lat, x0,
                degree=min(int(temporal_order), 2),
                min_points=min_track_points,
                max_neighbors=temporal_neighbors,
                exclude_x=None,
            )
            if np.isfinite(plon_u) and np.isfinite(plat):
                repairs[(str(key), int(fi))] = (
                    float(wrap_lon_deg(plon_u)), float(plat)
                )
        progress.update(track_no)

    progress.finish()
    return repairs

# -----------------------------------------------------------------------------
# Spatial consensus of temporal residual vectors
# -----------------------------------------------------------------------------

def normalize_source_xy(sx: np.ndarray, sy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    cx = float(np.nanmedian(sx))
    cy = float(np.nanmedian(sy))
    scale_x = max(float(np.nanpercentile(sx, 95) - np.nanpercentile(sx, 5)), 1.0)
    scale_y = max(float(np.nanpercentile(sy, 95) - np.nanpercentile(sy, 5)), 1.0)
    return (sx - cx) / scale_x, (sy - cy) / scale_y, cx, cy, scale_x, scale_y


def spatial_basis(xn: np.ndarray, yn: np.ndarray, order: int) -> np.ndarray:
    if int(order) <= 1:
        return np.column_stack([np.ones(len(xn)), xn, yn])
    return np.column_stack([
        np.ones(len(xn)), xn, yn, xn * xn, xn * yn, yn * yn
    ])


def robust_vector_field_fit(
    sx: np.ndarray,
    sy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    order: int = 1,
    max_iter: int = 8,
) -> Optional[dict]:
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(dx) & np.isfinite(dy)
    sx, sy, dx, dy = sx[finite], sy[finite], dx[finite], dy[finite]

    n_params = 3 if int(order) <= 1 else 6
    if len(sx) < max(6, n_params + 2):
        return None

    xn, yn, cx, cy, scale_x, scale_y = normalize_source_xy(sx, sy)
    X = spatial_basis(xn, yn, order)
    w = np.ones(len(sx), dtype=float)

    bx = weighted_lstsq(X, dx, w)
    by = weighted_lstsq(X, dy, w)
    if bx is None or by is None:
        return None

    for _ in range(max_iter):
        rx = dx - X @ bx
        ry = dy - X @ by
        rmag = np.hypot(rx, ry)
        med = float(np.median(rmag))
        scale = robust_scale(rmag, floor=1e-6)
        u = np.maximum(0.0, rmag - med) / (1.5 * scale)
        new_w = np.ones_like(u)
        large = u > 1.0
        new_w[large] = 1.0 / u[large]
        nbx = weighted_lstsq(X, dx, new_w)
        nby = weighted_lstsq(X, dy, new_w)
        if nbx is None or nby is None:
            break
        if np.allclose(nbx, bx, rtol=1e-7, atol=1e-9) and np.allclose(nby, by, rtol=1e-7, atol=1e-9):
            bx, by = nbx, nby
            break
        bx, by = nbx, nby

    return {
        "bx": bx,
        "by": by,
        "cx": cx,
        "cy": cy,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "order": int(order),
    }


def eval_vector_field(model: Optional[dict], sx: float, sy: float) -> Optional[Tuple[float, float]]:
    if model is None:
        return None
    xn = np.asarray([(float(sx) - model["cx"]) / model["scale_x"]])
    yn = np.asarray([(float(sy) - model["cy"]) / model["scale_y"]])
    X = spatial_basis(xn, yn, model["order"])
    # X has one row here. Explicitly extract the scalar to avoid NumPy's
    # deprecated implicit conversion of a 1-D/2-D array to float.
    dx = np.asarray(X @ model["bx"]).reshape(-1).item()
    dy = np.asarray(X @ model["by"]).reshape(-1).item()
    return float(dx), float(dy)


def local_spatial_median(
    query_sx: float,
    query_sy: float,
    sx: np.ndarray,
    sy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    neighbour_count: int,
    exclude_index: Optional[int] = None,
) -> Optional[Tuple[float, float]]:
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(dx) & np.isfinite(dy)
    idx = np.where(finite)[0]
    if exclude_index is not None:
        idx = idx[idx != int(exclude_index)]
    if len(idx) < 3:
        return None

    sxv, syv = sx[idx], sy[idx]
    _, _, cx, cy, scale_x, scale_y = normalize_source_xy(sxv, syv)
    qx = (float(query_sx) - cx) / scale_x
    qy = (float(query_sy) - cy) / scale_y
    xx = (sxv - cx) / scale_x
    yy = (syv - cy) / scale_y
    dist2 = (xx - qx) ** 2 + (yy - qy) ** 2
    order = np.argsort(dist2)
    k = min(max(3, int(neighbour_count)), len(order))
    take = idx[order[:k]]
    return float(np.median(dx[take])), float(np.median(dy[take]))


def build_spatial_consensus(
    all_df: pd.DataFrame,
    frames: List[FrameInfo],
    models: Dict[str, TrackModel],
    spatial_neighbours: int,
    spatial_order: int,
    threshold_mode: str,
    temporal_sigma: float,
    temporal_outlier_km: float,
    temporal_min_threshold_km: float,
) -> Dict[int, dict]:
    per_frame: Dict[int, dict] = {}
    frame_groups = {int(fi): g.copy() for fi, g in all_df.groupby("__frame_pos", sort=False)}
    progress = ProgressReporter("spatial consensus by frame", len(frames))

    for frame_no, frame in enumerate(frames, start=1):
        fi = frame.pos
        g = frame_groups.get(fi, pd.DataFrame())

        rows = []
        for _, row in g.iterrows():
            key = str(row["__source_key"])
            tr = models.get(key)
            if tr is None:
                continue
            dx = tr.temporal_dx_km[fi]
            dy = tr.temporal_dy_km[fi]
            if not (np.isfinite(dx) and np.isfinite(dy)):
                continue
            rows.append((key, float(row["sourceX"]), float(row["sourceY"]), float(dx), float(dy)))

        if not rows:
            per_frame[fi] = {
                "keys": [], "sx": np.array([]), "sy": np.array([]),
                "dx": np.array([]), "dy": np.array([]), "field": None,
                "threshold": float("inf"), "temporal_threshold": float("inf"),
                "consensus": {}, "spatial_residual": {},
            }
            progress.update(frame_no)
            continue

        keys = [r[0] for r in rows]
        sx = np.asarray([r[1] for r in rows], dtype=float)
        sy = np.asarray([r[2] for r in rows], dtype=float)
        dx = np.asarray([r[3] for r in rows], dtype=float)
        dy = np.asarray([r[4] for r in rows], dtype=float)

        field = robust_vector_field_fit(sx, sy, dx, dy, order=spatial_order)
        consensus: Dict[str, Tuple[float, float]] = {}
        spatial_residual: Dict[str, float] = {}

        for j, key in enumerate(keys):
            local = local_spatial_median(
                sx[j], sy[j], sx, sy, dx, dy,
                neighbour_count=spatial_neighbours,
                exclude_index=j,
            )
            affine = eval_vector_field(field, sx[j], sy[j])

            # Local median is preferred because it preserves coherent regional
            # deformations. Robust global field is the fallback.
            c = local if local is not None else affine
            if c is None:
                c = (0.0, 0.0)
            consensus[key] = c
            spatial_residual[key] = float(np.hypot(dx[j] - c[0], dy[j] - c[1]))

        vals = np.asarray(list(spatial_residual.values()), dtype=float)
        threshold = robust_threshold(
            vals,
            mode=threshold_mode,
            sigma=temporal_sigma,
            absolute_km=temporal_outlier_km,
            min_threshold_km=temporal_min_threshold_km,
        )
        temporal_mags = np.hypot(dx, dy)
        temporal_threshold = robust_threshold(
            temporal_mags,
            mode=threshold_mode,
            sigma=temporal_sigma,
            absolute_km=temporal_outlier_km,
            min_threshold_km=temporal_min_threshold_km,
        )

        per_frame[fi] = {
            "keys": keys,
            "sx": sx,
            "sy": sy,
            "dx": dx,
            "dy": dy,
            "field": field,
            "threshold": threshold,
            "temporal_threshold": temporal_threshold,
            "consensus": consensus,
            "spatial_residual": spatial_residual,
        }
        progress.update(frame_no)

    progress.finish()
    return per_frame


def consensus_for_query(frame_spatial: dict, sx: float, sy: float) -> Tuple[float, float]:
    sx_arr = frame_spatial.get("sx", np.array([]))
    sy_arr = frame_spatial.get("sy", np.array([]))
    dx_arr = frame_spatial.get("dx", np.array([]))
    dy_arr = frame_spatial.get("dy", np.array([]))

    local = local_spatial_median(
        sx, sy, sx_arr, sy_arr, dx_arr, dy_arr,
        neighbour_count=12,
        exclude_index=None,
    ) if len(sx_arr) else None
    if local is not None:
        return local
    affine = eval_vector_field(frame_spatial.get("field"), sx, sy)
    if affine is not None:
        return affine
    if len(dx_arr):
        return float(np.nanmedian(dx_arr)), float(np.nanmedian(dy_arr))
    return 0.0, 0.0


# -----------------------------------------------------------------------------
# Gap logic
# -----------------------------------------------------------------------------

def may_fill_missing_frame(
    frame_id: int,
    observed_frame_ids: np.ndarray,
    max_gap_frames: int,
    allow_extrapolation: bool,
) -> bool:
    obs = np.sort(np.asarray(observed_frame_ids, dtype=int))
    if len(obs) == 0:
        return False
    left = obs[obs < int(frame_id)]
    right = obs[obs > int(frame_id)]

    if len(left) and len(right):
        gap = int(right[0] - left[-1] - 1)
        return max_gap_frames < 0 or gap <= int(max_gap_frames)

    if allow_extrapolation:
        nearest = int(np.min(np.abs(obs - int(frame_id))))
        return max_gap_frames < 0 or nearest <= int(max_gap_frames)

    return False


def local_presence_support(
    frame_id: int,
    observed_frame_ids: np.ndarray,
    available_frame_ids: np.ndarray,
    radius_frames: int,
) -> Tuple[int, int, float, bool, bool]:
    """Measure whether a source-grid position is expected *locally* in time.

    The current frame is excluded from the vote.  This is deliberate: an
    isolated false patch in one frame must not vote for its own validity.

    Returns:
        present_count, neighbour_count, fraction, has_left_support, has_right_support
    """
    fid = int(frame_id)
    radius = max(int(radius_frames), 1)
    obs = set(np.asarray(observed_frame_ids, dtype=int).tolist())
    avail = np.asarray(available_frame_ids, dtype=int)

    neighbours = avail[
        (avail != fid)
        & (np.abs(avail - fid) <= radius)
    ]
    total = int(len(neighbours))
    if total == 0:
        return 0, 0, 0.0, False, False

    present = int(sum(int(x) in obs for x in neighbours))
    fraction = float(present) / float(total)

    left_ids = neighbours[neighbours < fid]
    right_ids = neighbours[neighbours > fid]
    has_left = bool(any(int(x) in obs for x in left_ids))
    has_right = bool(any(int(x) in obs for x in right_ids))

    return present, total, fraction, has_left, has_right


# -----------------------------------------------------------------------------
# Output construction
# -----------------------------------------------------------------------------

def row_to_output_dict(row: Optional[pd.Series], output_columns: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in output_columns:
        if row is not None and col in row.index and not str(col).startswith("__"):
            out[col] = row[col]
        elif col == "enable":
            out[col] = 1
        elif col in {"dX", "dY", "residual"}:
            out[col] = 0.0
        else:
            out[col] = np.nan
    return out


def build_outputs(
    all_df: pd.DataFrame,
    frames: List[FrameInfo],
    models: Dict[str, TrackModel],
    spatial: Dict[int, dict],
    frame_health: Dict[int, dict],
    corrupt_repairs: Dict[Tuple[str, int], Tuple[float, float]],
    output_columns: Sequence[str],
    min_track_points: int,
    min_track_coverage: float,
    majority_track_coverage: float,
    local_presence_radius: int,
    local_presence_fraction: float,
    max_gap_frames: int,
    allow_extrapolation: bool,
    fill_missing: bool,
    unresolved_outlier_policy: str,
    add_qc_columns: bool,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    outputs: Dict[int, List[Dict[str, Any]]] = {f.pos: [] for f in frames}
    qc_rows: List[dict] = []

    grouped = {key: g.copy() for key, g in all_df.groupby("__source_key", sort=True)}
    available_frame_ids = np.asarray([f.frame_id for f in frames], dtype=int)
    progress = ProgressReporter("classifying/reconstructing tracks", len(models))

    for track_no, (key, tr) in enumerate(models.items(), start=1):
        track = grouped.get(key, pd.DataFrame())
        rows_by_pos: Dict[int, pd.Series] = {}
        observed_ids: List[int] = []
        for _, row in track.sort_values(["__frame_pos", "__row_index"]).iterrows():
            fi = int(row["__frame_pos"])
            if fi not in rows_by_pos:
                rows_by_pos[fi] = row
                observed_ids.append(int(row["__frame_id"]))

        template_row = next(iter(rows_by_pos.values()), None)
        template = row_to_output_dict(template_row, output_columns)
        template["sourceX"] = tr.sourceX
        template["sourceY"] = tr.sourceY

        observed_ids_arr = np.asarray(observed_ids, dtype=int)
        # Global coverage is kept only as a diagnostic/legacy quantity.
        # Whether a source-grid position is allowed to exist in a particular
        # frame is decided by LOCAL temporal occupancy around that frame.
        model_ready = bool(tr.observed_count >= min_track_points)

        for frame in frames:
            fi = frame.pos
            row = rows_by_pos.get(fi)
            observed = row is not None

            local_count, local_total, local_fraction, local_left, local_right = local_presence_support(
                frame_id=frame.frame_id,
                observed_frame_ids=observed_ids_arr,
                available_frame_ids=available_frame_ids,
                radius_frames=local_presence_radius,
            )
            # "Majority" is strict: exactly 50% is not enough.
            locally_expected = bool(
                local_total > 0
                and local_fraction > float(local_presence_fraction)
            )

            temporal_mag = tr.temporal_mag_km[fi]
            health = frame_health.get(fi, {})
            frame_corrupt = bool(health.get("corrupt", False))

            if frame_corrupt:
                repair = corrupt_repairs.get((key, fi))
                if repair is None:
                    plon = plat = np.nan
                else:
                    plon, plat = repair
            else:
                plon = tr.pred_lon[fi]
                plat = tr.pred_lat[fi]

            frame_sp = spatial.get(fi, {})
            threshold = float(frame_sp.get("threshold", float("inf")))
            temporal_threshold = float(frame_sp.get("temporal_threshold", float("inf")))
            consensus_map = frame_sp.get("consensus", {})
            spatial_res_map = frame_sp.get("spatial_residual", {})

            if frame_corrupt:
                # A corrupt whole frame must never validate/repair itself with
                # its own internally coherent spatial displacement.
                cdx, cdy = 0.0, 0.0
            elif key in consensus_map:
                cdx, cdy = consensus_map[key]
            else:
                cdx, cdy = consensus_for_query(frame_sp, tr.sourceX, tr.sourceY)

            spatial_res = float(spatial_res_map.get(key, np.nan))
            # A point is replaced only when it is inconsistent in BOTH senses:
            #   1) it departs from its local temporal prediction, and
            #   2) it also departs from the displacement shared by nearby grid points.
            # This protects coherent regional motion and also protects a good point
            # sitting next to a coherently shifted region.
            temporal_bad = bool(
                observed
                and np.isfinite(temporal_mag)
                and np.isfinite(temporal_threshold)
                and temporal_mag > temporal_threshold
            )

            # A temporally inconsistent observation is allowed to survive only
            # when there is POSITIVE spatial evidence that nearby tracks share
            # the same displacement. Missing/undefined spatial consensus is not
            # treated as evidence in favour of keeping it.
            spatially_supported = bool(
                np.isfinite(spatial_res)
                and np.isfinite(threshold)
                and spatial_res <= threshold
            )
            # Presence topology is an independent validity test.
            # A coherent false block can fool the spatial residual test, so an
            # observation is forbidden if its source-grid position is absent
            # from the majority of neighbouring frames.
            topology_outlier = bool(observed and not locally_expected)
            is_outlier = bool(
                topology_outlier
                or frame_corrupt
                or (temporal_bad and not spatially_supported)
            )

            alt_lon = alt_lat = np.nan
            if np.isfinite(plon) and np.isfinite(plat):
                alt_lon, alt_lat = add_km_offset_to_lonlat(plon, plat, cdx, cdy)

            status = "missing"
            emitted = False
            corrected = False

            if observed and not is_outlier:
                out = row_to_output_dict(row, output_columns)
                status = "kept"
                frame.kept_count += 1
                emitted = True

            elif observed and is_outlier:
                if topology_outlier:
                    # Never replace an observation in a source-grid location
                    # that is not locally expected to exist.  This removes
                    # isolated coherent false patches such as the 327046 case.
                    status = "dropped_local_presence_outlier"
                    frame.unresolved_count += 1
                elif (
                    model_ready
                    and locally_expected
                    and np.isfinite(alt_lon)
                    and np.isfinite(alt_lat)
                ):
                    out = row_to_output_dict(row, output_columns)
                    out["mapX"] = float(alt_lon)
                    out["mapY"] = float(alt_lat)
                    out["enable"] = 1
                    out["dX"] = 0.0
                    out["dY"] = 0.0
                    out["residual"] = 0.0
                    status = "replaced_corrupt_frame" if frame_corrupt else "replaced_outlier"
                    frame.replaced_count += 1
                    emitted = True
                    corrected = True
                elif unresolved_outlier_policy == "keep":
                    out = row_to_output_dict(row, output_columns)
                    status = "kept_unresolved_outlier"
                    frame.unresolved_count += 1
                    emitted = True
                else:
                    status = "dropped_unresolved_outlier"
                    frame.unresolved_count += 1

            elif not observed and fill_missing:
                can_fill_track = (
                    model_ready
                    and locally_expected
                    and local_left
                    and local_right
                    and may_fill_missing_frame(
                        frame.frame_id,
                        observed_ids_arr,
                        max_gap_frames=max_gap_frames,
                        allow_extrapolation=allow_extrapolation,
                    )
                )
                if can_fill_track and np.isfinite(alt_lon) and np.isfinite(alt_lat):
                    out = dict(template)
                    out["mapX"] = float(alt_lon)
                    out["mapY"] = float(alt_lat)
                    out["sourceX"] = float(tr.sourceX)
                    out["sourceY"] = float(tr.sourceY)
                    out["enable"] = 1
                    out["dX"] = 0.0
                    out["dY"] = 0.0
                    out["residual"] = 0.0
                    status = "filled_missing"
                    frame.filled_count += 1
                    emitted = True
                    corrected = True

            if emitted:
                if add_qc_columns:
                    out["temporal_status"] = status
                    out["temporal_residual_km"] = float(temporal_mag) if np.isfinite(temporal_mag) else np.nan
                    out["spatial_residual_km"] = float(spatial_res) if np.isfinite(spatial_res) else np.nan
                    out["spatial_consensus_dx_km"] = float(cdx)
                    out["spatial_consensus_dy_km"] = float(cdy)
                    out["local_presence_fraction"] = float(local_fraction)
                    out["local_presence_count"] = int(local_count)
                    out["local_presence_total"] = int(local_total)
                outputs[fi].append(out)

            qc_rows.append({
                "frame_id": int(frame.frame_id),
                "frame_pos": int(fi),
                "source_key": key,
                "sourceX": float(tr.sourceX),
                "sourceY": float(tr.sourceY),
                "observed": bool(observed),
                "status": status,
                "corrected": bool(corrected),
                "temporal_pred_lon": float(plon) if np.isfinite(plon) else np.nan,
                "temporal_pred_lat": float(plat) if np.isfinite(plat) else np.nan,
                "temporal_residual_km": float(temporal_mag) if np.isfinite(temporal_mag) else np.nan,
                "spatial_consensus_dx_km": float(cdx),
                "spatial_consensus_dy_km": float(cdy),
                "spatial_residual_km": float(spatial_res) if np.isfinite(spatial_res) else np.nan,
                "local_presence_fraction": float(local_fraction),
                "local_presence_count": int(local_count),
                "local_presence_total": int(local_total),
                "locally_expected": bool(locally_expected),
                "topology_outlier": bool(topology_outlier),
                "frame_corrupt": bool(frame_corrupt),
                "frame_median_temporal_residual_km": health.get("median_temporal_residual_km", np.nan),
                "frame_corruption_baseline_km": health.get("baseline_km", np.nan),
                "frame_corruption_threshold_km": health.get("threshold_km", np.nan),
                "frame_temporal_threshold_km": float(temporal_threshold),
                "frame_spatial_threshold_km": float(threshold),
                "alternative_lon": float(alt_lon) if np.isfinite(alt_lon) else np.nan,
                "alternative_lat": float(alt_lat) if np.isfinite(alt_lat) else np.nan,
            })

        progress.update(track_no)

    progress.finish()
    out_dfs: Dict[int, pd.DataFrame] = {}
    for frame in frames:
        df = pd.DataFrame(outputs[frame.pos], columns=list(output_columns))
        if not df.empty:
            df = df.assign(
                __sort_y=-pd.to_numeric(df["sourceY"], errors="coerce"),
                __sort_x=pd.to_numeric(df["sourceX"], errors="coerce"),
            ).sort_values(["__sort_y", "__sort_x"]).drop(columns=["__sort_y", "__sort_x"])
        out_dfs[frame.pos] = df

    return out_dfs, pd.DataFrame(qc_rows)


# -----------------------------------------------------------------------------
# Optional old-style final spatial filter
# -----------------------------------------------------------------------------

def spatial_neighbor_mask(df: pd.DataFrame, radius_km: float) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    if len(df) < 2:
        return mask
    coords_deg = df[["mapY", "mapX"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(coords_deg).all(axis=1)
    idx = np.where(finite)[0]
    if len(idx) < 2:
        return mask
    coords_rad = np.radians(coords_deg[idx])
    tree = BallTree(coords_rad, metric="haversine")
    neigh = tree.query_radius(coords_rad, r=float(radius_km) / EARTH_RADIUS_KM)
    mask[idx] = np.array([len(n) > 1 for n in neigh], dtype=bool)
    return mask


def apply_optional_post_spatial_filter(
    out_dfs: Dict[int, pd.DataFrame],
    frames: List[FrameInfo],
    radius_km: float,
    enabled: bool,
) -> Dict[int, pd.DataFrame]:
    if not enabled:
        return out_dfs
    result: Dict[int, pd.DataFrame] = {}
    progress = ProgressReporter("final spatial validity filter", len(frames))
    for frame_no, frame in enumerate(frames, start=1):
        df = out_dfs.get(frame.pos, pd.DataFrame())
        if df.empty:
            result[frame.pos] = df
            progress.update(frame_no)
            continue
        mask = spatial_neighbor_mask(df, radius_km)
        frame.post_spatial_removed = int((~mask).sum())
        result[frame.pos] = df.loc[mask].copy()
        progress.update(frame_no)
    progress.finish()
    return result


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def parse_reference_frame_positions(spec: str, frames: List[FrameInfo]) -> List[int]:
    if not spec:
        return []
    n = len(frames)
    out: List[int] = []
    id_to_pos = {f.frame_id: f.pos for f in frames}
    for token in [t.strip().lower() for t in spec.split(",") if t.strip()]:
        if token == "first":
            pos = 0
        elif token in {"mid", "middle", "center"}:
            pos = n // 2
        elif token == "last":
            pos = n - 1
        else:
            try:
                v = int(token)
            except ValueError:
                continue
            if v in id_to_pos:
                pos = id_to_pos[v]
            else:
                pos = v
        if 0 <= pos < n and pos not in out:
            out.append(pos)
    return out


def find_image_for_frame(image_dir: Path, mission: str, frame_id: int) -> Optional[Path]:
    stem = f"{mission}-E-{frame_id}"
    for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png", ".tif", ".tiff"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    matches = sorted(image_dir.glob(f"*{frame_id}*"))
    return matches[0] if matches else None


def make_diagnostic_plots(
    frames: List[FrameInfo],
    qc: pd.DataFrame,
    plot_dir: Optional[str],
    image_dir: Optional[str],
    mission: str,
    plot_reference_frames: str,
) -> None:
    if not plot_dir:
        return
    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Counts per frame.
    x = [f.frame_id for f in frames]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, [f.kept_count for f in frames], label="kept")
    ax.plot(x, [f.replaced_count for f in frames], label="replaced outliers")
    ax.plot(x, [f.filled_count for f in frames], label="filled missing")
    ax.plot(x, [f.unresolved_count for f in frames], label="unresolved")
    ax.set_xlabel("frame ID")
    ax.set_ylabel("n points")
    ax.set_title("Spatio-temporal filtering per frame")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "spatiotemporal_frame_counts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not image_dir:
        return
    img_dir = Path(image_dir)
    if not img_dir.exists():
        return

    positions = parse_reference_frame_positions(plot_reference_frames, frames)
    for pos in positions:
        frame = frames[pos]
        img_path = find_image_for_frame(img_dir, mission, frame.frame_id)
        if img_path is None:
            continue
        try:
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        except Exception:
            continue

        q = qc[qc["frame_pos"] == pos].copy()
        fig, ax = plt.subplots(figsize=(13, 8.5))
        ax.imshow(img, origin="upper")

        groups = [
            ("kept", "o", 20),
            ("replaced_outlier", "D", 62),
            ("filled_missing", "s", 54),
            ("kept_unresolved_outlier", "x", 60),
            ("dropped_unresolved_outlier", "x", 60),
        ]
        for status, marker, size in groups:
            qq = q[q["status"] == status]
            if qq.empty:
                continue
            ax.scatter(
                qq["sourceX"], -qq["sourceY"],
                s=size, marker=marker, label=f"{status} ({len(qq)})", alpha=0.85,
            )

        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.set_title(
            f"Spatio-temporal QC | {frame.output_name}\n"
            "Outliers are deviations from LOCAL spatial consensus, not simply from the temporal curve"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(out / f"spatiotemporal_overlay_{frame.frame_id}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def filter_and_rename_points(
    input_folder: str,
    output_folder: str,
    radius_km: float,
    start_id: int,
    end_id: int,
    mission: str,
    input_glob: str = "*_real.points",
    source_round_decimals: int = 3,
    disable_temporal: bool = False,
    temporal_order: int = 2,
    temporal_threshold_mode: str = "sigma",
    temporal_outlier_km: float = 80.0,
    temporal_sigma: float = 3.0,
    temporal_min_threshold_km: float = 1.0,
    min_track_points: int = 6,
    min_track_coverage: float = 0.20,
    majority_track_coverage: float = 0.50,
    local_presence_radius: int = 4,
    local_presence_fraction: float = 0.50,
    frame_corruption_detection: bool = True,
    frame_health_radius: int = 6,
    frame_health_min_points: int = 20,
    frame_corrupt_min_median_km: float = 50.0,
    frame_corrupt_ratio: float = 3.0,
    frame_corrupt_sigma: float = 4.0,
    frame_health_scale_floor_km: float = 5.0,
    max_gap_frames: int = 8,
    allow_extrapolation: bool = False,
    fill_missing: bool = True,
    temporal_neighbors: int = 15,
    spatial_neighbours: int = 12,
    spatial_order: int = 1,
    unresolved_outlier_policy: str = "drop",
    post_spatial_filter: bool = True,
    add_qc_columns: bool = False,
    plot_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    plot_reference_frames: str = "",
) -> None:
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, all_df, first_input_columns = load_timelapse_points(
        input_folder=input_folder,
        start_id=start_id,
        end_id=end_id,
        mission=mission,
        input_glob=input_glob,
        source_round_decimals=source_round_decimals,
    )

    print(f"Frames loaded: {len(frames)}")
    print(f"Point observations loaded: {len(all_df)}")
    print(f"Tracks: {0 if all_df.empty else all_df['__source_key'].nunique()}")

    if not frames or all_df.empty:
        print("WARNING: no valid points to process")
        return

    output_columns = discover_output_columns(first_input_columns, add_qc_columns)

    if disable_temporal:
        # Simple passthrough compatible with old --disable_temporal behaviour.
        for frame in frames:
            g = all_df[all_df["__frame_pos"] == frame.pos].copy()
            cols = [c for c in output_columns if c in g.columns]
            out = g[cols].copy()
            for c in output_columns:
                if c not in out.columns:
                    if c == "enable":
                        out[c] = 1
                    elif c in {"dX", "dY", "residual"}:
                        out[c] = 0.0
                    else:
                        out[c] = np.nan
            out = out[output_columns]
            out.to_csv(out_dir / frame.output_name, index=False)
        print("Temporal filtering disabled: files copied/renamed")
        return

    print("\n[stage 1/7] Building local temporal models...", flush=True)
    models = build_temporal_models(
        all_df=all_df,
        frames=frames,
        temporal_order=temporal_order,
        min_track_points=min_track_points,
        min_track_coverage=min_track_coverage,
        temporal_neighbors=temporal_neighbors,
    )

    print("\n[stage 2/7] Detecting coherent whole-frame temporal failures...", flush=True)
    if frame_corruption_detection:
        frame_health = detect_corrupt_frames(
            frames=frames,
            models=models,
            min_points=frame_health_min_points,
            neighbour_radius=frame_health_radius,
            min_median_km=frame_corrupt_min_median_km,
            ratio=frame_corrupt_ratio,
            sigma=frame_corrupt_sigma,
            scale_floor_km=frame_health_scale_floor_km,
        )
    else:
        frame_health = {f.pos: {"corrupt": False} for f in frames}

    bad_ids = [
        int(f.frame_id) for f in frames
        if bool(frame_health.get(f.pos, {}).get("corrupt", False))
    ]
    if bad_ids:
        print(f"   Corrupt frames detected: {len(bad_ids)} -> {bad_ids}", flush=True)
    else:
        print("   No corrupt frames detected.", flush=True)

    print("\n[stage 3/7] Building clean repair predictions only for corrupt frames...", flush=True)
    corrupt_repairs = build_corrupt_frame_repairs(
        all_df=all_df,
        frames=frames,
        frame_health=frame_health,
        temporal_order=temporal_order,
        min_track_points=min_track_points,
        temporal_neighbors=temporal_neighbors,
        allow_extrapolation=allow_extrapolation,
    )
    print(f"   Clean corrupt-frame predictions: {len(corrupt_repairs)}", flush=True)

    print("\n[stage 4/7] Estimating spatial consensus in each frame...", flush=True)
    spatial = build_spatial_consensus(
        all_df=all_df,
        frames=frames,
        models=models,
        spatial_neighbours=spatial_neighbours,
        spatial_order=spatial_order,
        threshold_mode=temporal_threshold_mode,
        temporal_sigma=temporal_sigma,
        temporal_outlier_km=temporal_outlier_km,
        temporal_min_threshold_km=temporal_min_threshold_km,
    )

    print("\n[stage 5/7] Classifying outliers and reconstructing structural tracks...", flush=True)
    out_dfs, qc = build_outputs(
        all_df=all_df,
        frames=frames,
        models=models,
        spatial=spatial,
        frame_health=frame_health,
        corrupt_repairs=corrupt_repairs,
        output_columns=output_columns,
        min_track_points=min_track_points,
        min_track_coverage=min_track_coverage,
        majority_track_coverage=majority_track_coverage,
        local_presence_radius=local_presence_radius,
        local_presence_fraction=local_presence_fraction,
        max_gap_frames=max_gap_frames,
        allow_extrapolation=allow_extrapolation,
        fill_missing=fill_missing,
        unresolved_outlier_policy=unresolved_outlier_policy,
        add_qc_columns=add_qc_columns,
    )

    print("\n[stage 6/7] Applying final spatial validity filter...", flush=True)
    out_dfs = apply_optional_post_spatial_filter(
        out_dfs, frames, radius_km=radius_km, enabled=post_spatial_filter
    )

    print("\n[stage 7/7] Writing filtered .points and QC tables...", flush=True)
    write_progress = ProgressReporter("writing filtered .points", len(frames))
    for frame_no, frame in enumerate(frames, start=1):
        df = out_dfs.get(frame.pos, pd.DataFrame(columns=output_columns))
        df.to_csv(out_dir / frame.output_name, index=False)
        # Keep detailed lines only for frames where something changed; progress
        # already reports the routine all-good frames without flooding the log.
        if (frame.replaced_count or frame.filled_count or frame.unresolved_count or frame.post_spatial_removed):
            print(
                f"QC {frame.output_name}: input={frame.input_count}, "
                f"kept={frame.kept_count}, replaced={frame.replaced_count}, "
                f"filled={frame.filled_count}, unresolved={frame.unresolved_count}, "
                f"post_removed={frame.post_spatial_removed}, final={len(df)}"
            )
        write_progress.update(frame_no)

    write_progress.finish()
    # QC tables.
    qc.to_csv(out_dir / "spatiotemporal_point_qc.csv", index=False)

    frame_summary = pd.DataFrame([
        {
            "frame_pos": f.pos,
            "frame_id": f.frame_id,
            "input_name": f.input_name,
            "output_name": f.output_name,
            "input_count": f.input_count,
            "kept": f.kept_count,
            "replaced_outliers": f.replaced_count,
            "filled_missing": f.filled_count,
            "unresolved_outliers": f.unresolved_count,
            "post_spatial_removed": f.post_spatial_removed,
            "frame_corrupt": bool(frame_health.get(f.pos, {}).get("corrupt", False)),
            "frame_temporal_residual_count": frame_health.get(f.pos, {}).get("n_temporal_residuals", 0),
            "frame_median_temporal_residual_km": frame_health.get(f.pos, {}).get("median_temporal_residual_km", np.nan),
            "frame_p75_temporal_residual_km": frame_health.get(f.pos, {}).get("p75_temporal_residual_km", np.nan),
            "frame_p90_temporal_residual_km": frame_health.get(f.pos, {}).get("p90_temporal_residual_km", np.nan),
            "frame_corruption_baseline_km": frame_health.get(f.pos, {}).get("baseline_km", np.nan),
            "frame_corruption_threshold_km": frame_health.get(f.pos, {}).get("threshold_km", np.nan),
            "temporal_threshold_km": spatial.get(f.pos, {}).get("temporal_threshold", np.nan),
            "spatial_threshold_km": spatial.get(f.pos, {}).get("threshold", np.nan),
        }
        for f in frames
    ])
    frame_summary.to_csv(out_dir / "temporal_frame_summary.csv", index=False)

    track_summary = pd.DataFrame([
        {
            "source_key": tr.key,
            "sourceX": tr.sourceX,
            "sourceY": tr.sourceY,
            "n_observed": tr.observed_count,
            "coverage": tr.coverage,
            "median_temporal_residual_km": float(np.nanmedian(tr.temporal_mag_km)) if np.isfinite(tr.temporal_mag_km).any() else np.nan,
            "p95_temporal_residual_km": float(np.nanpercentile(tr.temporal_mag_km, 95)) if np.isfinite(tr.temporal_mag_km).any() else np.nan,
        }
        for tr in models.values()
    ])
    track_summary.to_csv(out_dir / "temporal_track_summary.csv", index=False)

    make_diagnostic_plots(
        frames=frames,
        qc=qc,
        plot_dir=plot_dir,
        image_dir=image_dir,
        mission=mission,
        plot_reference_frames=plot_reference_frames,
    )

    print("\nAlternative spatio-temporal filtering completed.")


# -----------------------------------------------------------------------------
# CLI - accepts the arguments currently used by timelapse_pipeline.py
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust local spatio-temporal filtering of ISS GCP tracks"
    )
    p.add_argument("--input_folder", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--radius_km", type=float, default=80.0)
    p.add_argument("--start_id", type=int, required=True)
    p.add_argument("--end_id", type=int, required=True)
    p.add_argument("--mission", required=True)
    p.add_argument("--input_glob", default="*_real.points")
    p.add_argument("--source_round_decimals", type=int, default=3)

    p.add_argument("--disable_temporal", action="store_true")
    p.add_argument("--temporal_order", type=int, default=2)
    p.add_argument("--temporal_threshold_mode", choices=["sigma", "hybrid", "absolute"], default="sigma")
    p.add_argument("--temporal_sigma", type=float, default=3.0)
    p.add_argument("--temporal_outlier_km", type=float, default=80.0)
    p.add_argument("--temporal_min_threshold_km", type=float, default=1.0)
    p.add_argument("--temporal_max_iter", type=int, default=4, help="Legacy accepted; local IRLS uses its own iterations")
    p.add_argument("--min_track_points", type=int, default=6)
    p.add_argument("--min_track_coverage", type=float, default=0.20)
    p.add_argument("--majority_track_coverage", type=float, default=0.50,
                   help="Legacy/global coverage diagnostic; local occupancy now controls whether a point may exist in each frame")
    p.add_argument("--local_presence_radius", type=int, default=4,
                   help="Temporal half-window, in frame IDs, used to decide whether a source-grid position is locally allowed")
    p.add_argument("--local_presence_fraction", type=float, default=0.50,
                   help="Strict local neighbour fraction required for a source-grid position to be allowed; default >50%%")
    p.add_argument("--frame_corruption_detection", action=argparse.BooleanOptionalAction, default=True,
                   help="Detect coherent whole-frame temporal failures (default: enabled)")
    p.add_argument("--frame_health_radius", type=int, default=6)
    p.add_argument("--frame_health_min_points", type=int, default=20)
    p.add_argument("--frame_corrupt_min_median_km", type=float, default=50.0)
    p.add_argument("--frame_corrupt_ratio", type=float, default=3.0)
    p.add_argument("--frame_corrupt_sigma", type=float, default=4.0)
    p.add_argument("--frame_health_scale_floor_km", type=float, default=5.0)
    p.add_argument("--max_gap_frames", type=int, default=8)
    p.add_argument("--allow_extrapolation", action="store_true")
    p.add_argument("--fill_missing", action="store_true", default=True,
                   help="Fill missing points only for structural tracks present in the majority of frames (default: enabled)")
    p.add_argument("--no_fill_missing", action="store_true",
                   help="Disable filling of missing points, even for structural majority tracks")

    # New controls.
    p.add_argument("--temporal_neighbors", type=int, default=15,
                   help="Maximum temporal observations used for each local prediction")
    p.add_argument("--spatial_neighbours", type=int, default=12,
                   help="Nearby grid tracks used to estimate local residual consensus")
    p.add_argument("--spatial_order", type=int, choices=[1, 2], default=1,
                   help="Fallback robust spatial vector-field order")
    p.add_argument("--unresolved_outlier_policy", choices=["keep", "drop"], default="drop")

    # Compatibility with old pipeline flags.
    p.add_argument("--pre_spatial_filter", action="store_true", help="Accepted for compatibility; not used")
    p.add_argument("--no_pre_spatial_filter", action="store_true", help="Accepted for compatibility")
    p.add_argument("--post_spatial_filter", action="store_true",
                   help="Enable old final geographic neighbour deletion")
    p.add_argument("--no_post_spatial_filter", action="store_true",
                   help="Explicitly disable final geographic neighbour deletion")

    p.add_argument("--add_qc_columns", action="store_true")
    p.add_argument("--plot_dir", default=None)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--plot_reference_frames", default="")
    p.add_argument("--diagnostic_inset", type=float, default=0.25,
                   help="Legacy accepted for compatibility")
    p.add_argument("--min_diagnostic_observations", type=int, default=20,
                   help="Legacy accepted for compatibility")
    p.add_argument("--progress-step-percent", type=float, default=5.0,
                   help="Progress reporting interval in percent (default: 5)")

    return p.parse_args()


def main() -> None:
    global PROGRESS_STEP_PERCENT
    args = parse_args()
    PROGRESS_STEP_PERCENT = max(float(args.progress_step_percent), 0.1)
    post_spatial = not bool(args.no_post_spatial_filter)

    filter_and_rename_points(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        radius_km=args.radius_km,
        start_id=args.start_id,
        end_id=args.end_id,
        mission=args.mission,
        input_glob=args.input_glob,
        source_round_decimals=args.source_round_decimals,
        disable_temporal=args.disable_temporal,
        temporal_order=args.temporal_order,
        temporal_threshold_mode=args.temporal_threshold_mode,
        temporal_outlier_km=args.temporal_outlier_km,
        temporal_sigma=args.temporal_sigma,
        temporal_min_threshold_km=args.temporal_min_threshold_km,
        min_track_points=args.min_track_points,
        min_track_coverage=args.min_track_coverage,
        majority_track_coverage=args.majority_track_coverage,
        local_presence_radius=args.local_presence_radius,
        local_presence_fraction=args.local_presence_fraction,
        frame_corruption_detection=args.frame_corruption_detection,
        frame_health_radius=args.frame_health_radius,
        frame_health_min_points=args.frame_health_min_points,
        frame_corrupt_min_median_km=args.frame_corrupt_min_median_km,
        frame_corrupt_ratio=args.frame_corrupt_ratio,
        frame_corrupt_sigma=args.frame_corrupt_sigma,
        frame_health_scale_floor_km=args.frame_health_scale_floor_km,
        max_gap_frames=args.max_gap_frames,
        allow_extrapolation=args.allow_extrapolation,
        fill_missing=not bool(args.no_fill_missing),
        temporal_neighbors=args.temporal_neighbors,
        spatial_neighbours=args.spatial_neighbours,
        spatial_order=args.spatial_order,
        unresolved_outlier_policy=args.unresolved_outlier_policy,
        post_spatial_filter=post_spatial,
        add_qc_columns=args.add_qc_columns,
        plot_dir=args.plot_dir,
        image_dir=args.image_dir,
        plot_reference_frames=args.plot_reference_frames,
    )


if __name__ == "__main__":
    main()
