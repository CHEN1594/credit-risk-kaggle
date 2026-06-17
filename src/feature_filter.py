from __future__ import annotations

import polars as pl


def filter_polars_columns(
    df: pl.DataFrame,
    *,
    max_missing: float,
    max_cat_unique: int,
    min_unique: int = 2,
) -> tuple[pl.DataFrame, list[str], dict[str, list[str]]]:
    keep = {"case_id", "target", "WEEK_NUM"}
    dropped: dict[str, list[str]] = {"missing": [], "constant": [], "high_cardinality": []}
    selected: list[str] = []
    n_rows = len(df)
    for col, dtype in zip(df.columns, df.dtypes):
        if col in keep:
            selected.append(col)
            continue
        missing = df[col].null_count() / n_rows
        if missing > max_missing:
            dropped["missing"].append(col)
            continue
        nunique = df[col].n_unique()
        if nunique < min_unique:
            dropped["constant"].append(col)
            continue
        if dtype in (pl.String, pl.Categorical) and nunique > max_cat_unique:
            dropped["high_cardinality"].append(col)
            continue
        selected.append(col)
    return df.select(selected), selected, dropped


def select_polars_columns(df: pl.DataFrame, selected: list[str]) -> pl.DataFrame:
    available = [col for col in selected if col in df.columns]
    return df.select(available)


def hash_string_columns(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype in (pl.String, pl.Categorical):
            encoded = pl.col(col).hash(seed=seed).mod(2_147_483_647).cast(pl.Int32)
            exprs.append(pl.when(pl.col(col).is_null()).then(pl.lit(-1)).otherwise(encoded).alias(col))
    return df.with_columns(exprs) if exprs else df


def drop_experiment_columns(
    df: pl.DataFrame,
    *,
    exclude_prefixes: list[str] | None = None,
    exclude_aggs: list[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    protected = {"case_id", "target", "WEEK_NUM"}
    exclude_prefixes = exclude_prefixes or []
    exclude_aggs = exclude_aggs or []
    dropped: dict[str, list[str]] = {"prefix": [], "agg": []}

    for col in df.columns:
        if col in protected:
            continue
        if any(col.startswith(prefix) for prefix in exclude_prefixes):
            dropped["prefix"].append(col)
            continue
        if any(col.endswith(f"__{agg}") for agg in exclude_aggs):
            dropped["agg"].append(col)

    to_drop = sorted(set(dropped["prefix"] + dropped["agg"]))
    if not to_drop:
        return df, dropped
    return df.drop(to_drop), dropped
