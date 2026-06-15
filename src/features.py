from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


SPECIAL_COLUMNS = {"case_id", "num_group1", "num_group2"}


@dataclass(frozen=True)
class FeaturePreset:
    depth0_groups: tuple[str, ...]
    aggregate_groups: tuple[str, ...]


PRESETS: dict[str, FeaturePreset] = {
    "static": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(),
    ),
    "baseline": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(
            "person_1",
            "person_2",
            "applprev_1",
            "applprev_2",
            "debitcard_1",
            "deposit_1",
            "other_1",
            "tax_registry_a_1",
            "tax_registry_b_1",
            "tax_registry_c_1",
            "credit_bureau_b_1",
            "credit_bureau_b_2",
        ),
    ),
    "medium": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(
            "person_1",
            "person_2",
            "applprev_1",
            "applprev_2",
            "debitcard_1",
            "deposit_1",
            "other_1",
            "tax_registry_a_1",
            "tax_registry_b_1",
            "tax_registry_c_1",
            "credit_bureau_a_1",
            "credit_bureau_b_1",
            "credit_bureau_b_2",
        ),
    ),
    "full": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(
            "person_1",
            "person_2",
            "applprev_1",
            "applprev_2",
            "debitcard_1",
            "deposit_1",
            "other_1",
            "tax_registry_a_1",
            "tax_registry_b_1",
            "tax_registry_c_1",
            "credit_bureau_a_1",
            "credit_bureau_a_2",
            "credit_bureau_b_1",
            "credit_bureau_b_2",
        ),
    ),
}


def split_dir(data_dir: Path, split: str) -> Path:
    return data_dir / "parquet_files" / split


def files_for_group(data_dir: Path, split: str, group: str) -> list[Path]:
    paths = sorted(split_dir(data_dir, split).glob(f"{split}_{group}*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found for split={split!r}, group={group!r}")
    return paths


def scan_group(data_dir: Path, split: str, group: str) -> pl.LazyFrame:
    paths = files_for_group(data_dir, split, group)
    frames = [pl.scan_parquet(str(path)) for path in paths]
    schemas = [frame.collect_schema() for frame in frames]
    target_schema: dict[str, pl.DataType] = {}

    for schema in schemas:
        for col, dtype in schema.items():
            if col not in target_schema or target_schema[col] == pl.Null:
                target_schema[col] = dtype
    for key in ("case_id", "num_group1", "num_group2"):
        if key in target_schema:
            target_schema[key] = pl.Int64
    for col, dtype in list(target_schema.items()):
        if dtype == pl.Null:
            target_schema[col] = pl.Utf8

    normalized = []
    for frame, schema in zip(frames, schemas):
        exprs = []
        for col, target_dtype in target_schema.items():
            if col in schema and schema[col] != target_dtype:
                exprs.append(pl.col(col).cast(target_dtype, strict=False).alias(col))
        normalized.append(frame.with_columns(exprs) if exprs else frame)
    return pl.concat(normalized, how="vertical_relaxed")


def normalize_dates(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    exprs = []
    for name, dtype in schema.items():
        if name.endswith("D") or name == "date_decision":
            if dtype.is_numeric():
                continue
            exprs.append(
                pl.col(name)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, strict=False)
                .cast(pl.Int32)
                .alias(name)
            )
    return lf.with_columns(exprs) if exprs else lf


def load_base(data_dir: Path, split: str) -> pl.LazyFrame:
    return normalize_dates(scan_group(data_dir, split, "base"))


def load_depth0(data_dir: Path, split: str, group: str) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(data_dir, split, group))
    return lf.unique(subset=["case_id"], keep="last")


def aggregate_group(data_dir: Path, split: str, group: str) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(data_dir, split, group))
    schema = lf.collect_schema()
    aggs: list[pl.Expr] = [pl.len().alias(f"{group}__row_count")]

    for col, dtype in schema.items():
        if col in SPECIAL_COLUMNS:
            continue
        out = f"{group}__{col}"
        if dtype.is_numeric() or dtype == pl.Boolean or col.endswith("D"):
            base = pl.col(col).cast(pl.Float64, strict=False)
            aggs.extend(
                [
                    base.mean().alias(f"{out}__mean"),
                    base.max().alias(f"{out}__max"),
                    base.min().alias(f"{out}__min"),
                    base.std().alias(f"{out}__std"),
                ]
            )
        else:
            aggs.append(pl.col(col).n_unique().alias(f"{out}__nunique"))

    return lf.group_by("case_id").agg(aggs)


def build_features(
    data_dir: Path,
    split: str,
    preset_name: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Available: {sorted(PRESETS)}")
    preset = PRESETS[preset_name]

    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{split}_{preset_name}_features.parquet"
        if use_cache and cache_path.exists():
            return pl.read_parquet(cache_path)

    lf = load_base(data_dir, split)
    for group in preset.depth0_groups:
        lf = lf.join(load_depth0(data_dir, split, group), on="case_id", how="left")
    for group in preset.aggregate_groups:
        lf = lf.join(aggregate_group(data_dir, split, group), on="case_id", how="left")

    df = lf.collect(engine="streaming")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
    return df
