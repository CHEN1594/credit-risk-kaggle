from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


SPECIAL_COLUMNS = {"case_id", "num_group1", "num_group2"}


@dataclass(frozen=True)
class FeaturePreset:
    depth0_groups: tuple[str, ...]
    aggregate_groups: tuple[str, ...]
    lite_large_groups: tuple[str, ...] = ()


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
    "a2lite": FeaturePreset(
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
        lite_large_groups=("credit_bureau_a_2",),
    ),
    "a2core": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(
            "person_1",
            "applprev_1",
            "tax_registry_a_1",
            "credit_bureau_a_1",
        ),
        lite_large_groups=("credit_bureau_a_2",),
    ),
    "a2ultra": FeaturePreset(
        depth0_groups=("static_0", "static_cb_0"),
        aggregate_groups=(
            "person_1",
            "applprev_1",
            "tax_registry_a_1",
        ),
        lite_large_groups=("credit_bureau_a_1", "credit_bureau_a_2"),
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


LITE_LARGE_NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "credit_bureau_a_1": (
        "dpdmax_139P",
        "dpdmax_757P",
        "dpdmaxdateyear_596T",
        "overdueamountmax_155A",
        "overdueamountmax_35A",
        "numberofoverdueinstlmax_1151L",
        "numberofoverdueinstlmax_1039L",
        "numberofinstls_320L",
        "numberofinstls_229L",
        "residualamount_856A",
        "residualamount_488A",
        "totaloutstanddebtvalue_39A",
        "totaloutstanddebtvalue_668A",
        "debtoutstand_525A",
        "debtoverdue_47A",
    ),
    "credit_bureau_a_2": (
        "pmts_dpd_1073P",
        "pmts_dpd_303P",
        "pmts_overdue_1140A",
        "pmts_overdue_1152A",
        "pmts_month_158T",
        "pmts_month_706T",
        "pmts_year_1139T",
        "pmts_year_507T",
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


def filter_cases(lf: pl.LazyFrame, case_ids: pl.LazyFrame | None) -> pl.LazyFrame:
    if case_ids is None:
        return lf
    return lf.join(case_ids, on="case_id", how="semi")


def load_depth0(
    data_dir: Path,
    split: str,
    group: str,
    case_ids: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(data_dir, split, group))
    lf = filter_cases(lf, case_ids)
    return lf.unique(subset=["case_id"], keep="last")


def aggregate_group(
    data_dir: Path,
    split: str,
    group: str,
    case_ids: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(data_dir, split, group))
    lf = filter_cases(lf, case_ids)
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


def aggregate_large_group_lite(
    data_dir: Path,
    split: str,
    group: str,
    case_ids: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Aggregate very large groups file-by-file using only cheap composable stats.

    This avoids the high memory peak of grouping all credit_bureau_a_2 rows at once.
    """
    selected = LITE_LARGE_NUMERIC_COLUMNS[group]
    partials: list[pl.LazyFrame] = []
    for path in files_for_group(data_dir, split, group):
        lf = pl.scan_parquet(str(path))
        schema = lf.collect_schema()
        available = [col for col in selected if col in schema]
        lf = lf.select(["case_id", *available]).with_columns(pl.col("case_id").cast(pl.Int64))
        lf = filter_cases(lf, case_ids)
        aggs: list[pl.Expr] = [pl.len().alias(f"{group}__row_count")]
        for col in available:
            base = pl.col(col).cast(pl.Float64, strict=False)
            prefix = f"{group}__{col}"
            aggs.extend(
                [
                    base.sum().alias(f"{prefix}__sum"),
                    base.count().alias(f"{prefix}__count"),
                    base.max().alias(f"{prefix}__max"),
                    base.min().alias(f"{prefix}__min"),
                ]
            )
        partials.append(lf.group_by("case_id").agg(aggs))

    partial = pl.concat(partials, how="vertical_relaxed")
    schema = partial.collect_schema()
    final_aggs: list[pl.Expr] = [pl.col(f"{group}__row_count").sum().alias(f"{group}__row_count")]
    for col in selected:
        prefix = f"{group}__{col}"
        sum_col = f"{prefix}__sum"
        count_col = f"{prefix}__count"
        max_col = f"{prefix}__max"
        min_col = f"{prefix}__min"
        if sum_col not in schema:
            continue
        final_aggs.extend(
            [
                (pl.col(sum_col).sum() / pl.col(count_col).sum()).alias(f"{prefix}__mean"),
                pl.col(max_col).max().alias(f"{prefix}__max"),
                pl.col(min_col).min().alias(f"{prefix}__min"),
            ]
        )
    return partial.group_by("case_id").agg(final_aggs)


def aggregate_large_group_lite_eager(
    data_dir: Path,
    split: str,
    group: str,
    temp_dir: Path,
    case_ids: pl.DataFrame | None = None,
) -> pl.DataFrame:
    selected = LITE_LARGE_NUMERIC_COLUMNS[group]
    temp_dir.mkdir(parents=True, exist_ok=True)
    partial_paths: list[Path] = []
    case_ids_lf = case_ids.lazy() if case_ids is not None else None

    for idx, path in enumerate(files_for_group(data_dir, split, group)):
        schema = pl.scan_parquet(str(path)).collect_schema()
        available = [col for col in selected if col in schema]
        lf = pl.scan_parquet(str(path)).select(["case_id", *available])
        lf = lf.with_columns(pl.col("case_id").cast(pl.Int64))
        lf = filter_cases(lf, case_ids_lf)
        aggs: list[pl.Expr] = [pl.len().alias(f"{group}__row_count")]
        for col in available:
            base = pl.col(col).cast(pl.Float64, strict=False)
            prefix = f"{group}__{col}"
            aggs.extend(
                [
                    base.sum().alias(f"{prefix}__sum"),
                    base.count().alias(f"{prefix}__count"),
                    base.max().alias(f"{prefix}__max"),
                    base.min().alias(f"{prefix}__min"),
                ]
            )
        partial = lf.group_by("case_id").agg(aggs).collect(engine="streaming")
        out_path = temp_dir / f"{split}_{group}_partial_{idx}.parquet"
        partial.write_parquet(out_path)
        partial_paths.append(out_path)
        del partial

    partial_lf = pl.scan_parquet([str(path) for path in partial_paths])
    schema = partial_lf.collect_schema()
    final_aggs: list[pl.Expr] = [pl.col(f"{group}__row_count").sum().alias(f"{group}__row_count")]
    for col in selected:
        prefix = f"{group}__{col}"
        sum_col = f"{prefix}__sum"
        count_col = f"{prefix}__count"
        max_col = f"{prefix}__max"
        min_col = f"{prefix}__min"
        if sum_col not in schema:
            continue
        final_aggs.extend(
            [
                (pl.col(sum_col).sum() / pl.col(count_col).sum()).alias(f"{prefix}__mean"),
                pl.col(max_col).max().alias(f"{prefix}__max"),
                pl.col(min_col).min().alias(f"{prefix}__min"),
            ]
        )
    return partial_lf.group_by("case_id").agg(final_aggs).collect(engine="streaming")


def build_features(
    data_dir: Path,
    split: str,
    preset_name: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    sample_rows: int = 0,
) -> pl.DataFrame:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Available: {sorted(PRESETS)}")
    preset = PRESETS[preset_name]

    cache_path = None
    if cache_dir is not None and not sample_rows:
        cache_path = cache_dir / f"{split}_{preset_name}_features.parquet"
        if use_cache and cache_path.exists():
            return pl.read_parquet(cache_path)

    lf = load_base(data_dir, split)
    if sample_rows:
        lf = lf.sort("case_id").head(sample_rows)
    case_ids = lf.select("case_id")
    for group in preset.depth0_groups:
        lf = lf.join(load_depth0(data_dir, split, group, case_ids), on="case_id", how="left")
    for group in preset.aggregate_groups:
        lf = lf.join(aggregate_group(data_dir, split, group, case_ids), on="case_id", how="left")
    df = lf.collect(engine="streaming")
    if preset.lite_large_groups:
        case_ids_df = df.select("case_id")
        temp_root = (cache_path.parent if cache_path is not None else Path("outputs/features")) / "_tmp_lite"
        for group in preset.lite_large_groups:
            lite = aggregate_large_group_lite_eager(
                data_dir,
                split,
                group,
                temp_root / f"{split}_{group}",
                case_ids=case_ids_df if sample_rows else None,
            )
            df = df.join(lite, on="case_id", how="left")
    if cache_path is not None and not sample_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
    return df
