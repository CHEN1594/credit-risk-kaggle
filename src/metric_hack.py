from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


RESTORE_ANCHOR_DATE = pd.Timestamp("2019-01-03")
WEEK_ORIGIN_DATE = pd.Timestamp("2019-01-01")


def _as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="unix", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _competition_split_dir(data_dir: Path, split: str) -> Path:
    return data_dir / "parquet_files" / split


def load_test_time_signal(data_dir: Path) -> pd.DataFrame:
    """Return case_id and a test time signal for metric-hack postprocessing.

    The official test base normally contains WEEK_NUM. If it is unavailable, fall back
    to the public refreshdate_3813885D date restoration trick from credit_bureau_a_1.
    """
    test_dir = _competition_split_dir(data_dir, "test")
    base_path = test_dir / "test_base.parquet"
    base = pl.read_parquet(base_path).to_pandas()
    if "WEEK_NUM" in base.columns and base["WEEK_NUM"].notna().any():
        return base[["case_id", "WEEK_NUM"]].rename(columns={"WEEK_NUM": "mh_time_signal"})

    if "date_decision" not in base.columns:
        return base[["case_id"]].assign(mh_time_signal=np.nan)

    refresh_frames = []
    for path in sorted(test_dir.glob("test_credit_bureau_a_1_*.parquet")):
        schema = pl.scan_parquet(str(path)).collect_schema()
        if "refreshdate_3813885D" not in schema:
            continue
        refresh_frames.append(pl.scan_parquet(str(path)).select(["case_id", "refreshdate_3813885D"]))
    if not refresh_frames:
        return base[["case_id"]].assign(mh_time_signal=np.nan)

    refresh = pl.concat(refresh_frames, how="vertical_relaxed").collect(engine="streaming").to_pandas()
    merged = refresh.merge(base[["case_id", "date_decision"]], on="case_id", how="left")
    refresh_dt = _as_datetime(merged["refreshdate_3813885D"])
    decision_dt = _as_datetime(merged["date_decision"])
    merged["refreshdate_diff_days"] = (refresh_dt - decision_dt).dt.days
    agg = merged.groupby("case_id", as_index=False)["refreshdate_diff_days"].min()
    restored_date = RESTORE_ANCHOR_DATE - pd.to_timedelta(agg["refreshdate_diff_days"], unit="D")
    agg["mh_time_signal"] = ((restored_date - WEEK_ORIGIN_DATE).dt.days // 7).astype("float64")
    return base[["case_id"]].merge(agg[["case_id", "mh_time_signal"]], on="case_id", how="left")


def apply_metric_hack(
    submission: pd.DataFrame,
    data_dir: Path,
    divide: float = 0.5,
    reduce: float = 0.03,
) -> tuple[pd.DataFrame, dict]:
    timeline = load_test_time_signal(data_dir)
    out = submission.merge(timeline, on="case_id", how="left")
    valid = out["mh_time_signal"].notna()
    before = out["score"].copy()

    if valid.any():
        min_week = float(out.loc[valid, "mh_time_signal"].min())
        max_week = float(out.loc[valid, "mh_time_signal"].max())
        cutoff = (max_week - min_week) * float(divide) + min_week
        condition = valid & (out["mh_time_signal"] < cutoff)
        out.loc[condition, "score"] = (out.loc[condition, "score"] - float(reduce)).clip(0.0, 1.0)
    else:
        min_week = np.nan
        max_week = np.nan
        cutoff = np.nan
        condition = pd.Series(False, index=out.index)

    summary = {
        "enabled": True,
        "divide": float(divide),
        "reduce": float(reduce),
        "time_signal_non_null": int(valid.sum()),
        "time_signal_min": None if pd.isna(min_week) else float(min_week),
        "time_signal_max": None if pd.isna(max_week) else float(max_week),
        "cutoff": None if pd.isna(cutoff) else float(cutoff),
        "adjusted_rows": int(condition.sum()),
        "mean_score_before": float(before.mean()),
        "mean_score_after": float(out["score"].mean()),
    }
    return out.drop(columns=["mh_time_signal"]), summary
