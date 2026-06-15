from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype


@dataclass
class PreprocessState:
    feature_cols: list[str]
    category_maps: dict[str, list[object]]
    fill_values: dict[str, float]
    dropped_columns: dict[str, list[str]]


DROP_ALWAYS = {"case_id", "target"}


def reduce_mem_usage(df: pd.DataFrame, use_float16: bool = False) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("int8")
        elif is_float_dtype(df[col]):
            df[col] = df[col].astype("float16" if use_float16 else "float32")
        elif is_integer_dtype(df[col]) and col != "case_id":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def fit_preprocessor(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    max_missing: float = 0.92,
    max_cat_unique: int = 200,
    min_unique: int = 2,
    keep_cols: set[str] | None = None,
    use_float16: bool = False,
) -> PreprocessState:
    keep_cols = keep_cols or {"WEEK_NUM"}
    candidate_cols = [col for col in df.columns if col not in DROP_ALWAYS]
    dropped: dict[str, list[str]] = {
        "missing": [],
        "constant": [],
        "high_cardinality": [],
    }
    selected: list[str] = []

    for col in candidate_cols:
        if col in keep_cols:
            selected.append(col)
            continue
        missing = float(df[col].isna().mean())
        if missing > max_missing:
            dropped["missing"].append(col)
            continue
        nunique = int(df[col].nunique(dropna=True))
        if nunique < min_unique:
            dropped["constant"].append(col)
            continue
        if (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])) and nunique > max_cat_unique:
            dropped["high_cardinality"].append(col)
            continue
        selected.append(col)

    category_maps: dict[str, list[object]] = {}
    fill_values: dict[str, float] = {}

    for col in selected:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            cat = df[col].astype("category")
            category_maps[col] = list(cat.cat.categories)
            df[col] = cat.cat.codes.astype("int32")
        elif is_float_dtype(df[col]) or is_integer_dtype(df[col]):
            fill_values[col] = float(df[col].median()) if df[col].notna().any() else 0.0

    reduce_mem_usage(df, use_float16=use_float16)
    return PreprocessState(
        feature_cols=selected,
        category_maps=category_maps,
        fill_values=fill_values,
        dropped_columns=dropped,
    )


def apply_preprocessor(df: pd.DataFrame, state: PreprocessState, use_float16: bool = False) -> pd.DataFrame:
    missing = [col for col in state.feature_cols if col not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    df = df[state.feature_cols].copy()

    for col, categories in state.category_maps.items():
        if col in df.columns:
            mapper = {value: idx for idx, value in enumerate(categories)}
            df[col] = df[col].map(mapper).fillna(-1).astype("int32")

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes.astype("int32")
        elif col in state.fill_values:
            df[col] = df[col].fillna(state.fill_values[col])

    reduce_mem_usage(df, use_float16=use_float16)
    return df


def summarize_drops(state: PreprocessState) -> dict[str, int]:
    return {key: len(value) for key, value in state.dropped_columns.items()} | {"kept": len(state.feature_cols)}
