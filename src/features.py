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
    feature_set: str = "none",
) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(data_dir, split, group))
    lf = filter_cases(lf, case_ids)
    schema = lf.collect_schema()
    components = feature_set_components(feature_set)
    if "last" in components and "num_group1" in schema:
        lf = lf.sort(["case_id", "num_group1"])
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
            if "last" in components:
                aggs.append(base.last().alias(f"{out}__last"))
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


def _safe_ratio_expr(numerator: str, denominator: str, alias: str) -> pl.Expr:
    den = pl.col(denominator).cast(pl.Float64, strict=False)
    num = pl.col(numerator).cast(pl.Float64, strict=False)
    return pl.when(den.abs() > 1e-6).then(num / den).otherwise(None).alias(alias)


RATIO_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("annuity_780A", "credamount_770A", "derived__annuity_to_credamount"),
    ("annuitynextmonth_57A", "annuity_780A", "derived__next_annuity_to_annuity"),
    ("currdebt_22A", "credamount_770A", "derived__currdebt_to_credamount"),
    ("disbursedcredamount_1113A", "credamount_770A", "derived__disbursed_to_credamount"),
    ("downpmt_116A", "credamount_770A", "derived__downpmt_to_credamount"),
    ("maxannuity_159A", "annuity_780A", "derived__maxannuity_to_annuity"),
    ("lastapprcredamount_781A", "credamount_770A", "derived__lastappr_to_credamount"),
    ("lastrejectcredamount_222A", "credamount_770A", "derived__lastreject_to_credamount"),
    ("avgpmtlast12m_4525200A", "annuity_780A", "derived__avgpmt_to_annuity"),
    ("avgoutstandbalancel6m_4187114A", "credamount_770A", "derived__avgoutstand_to_credamount"),
    (
        "credit_bureau_a_1__overdueamount_31A__max",
        "credit_bureau_a_1__outstandingamount_354A__max",
        "derived__bureau_overdue_to_outstanding_max",
    ),
    (
        "credit_bureau_a_1__overdueamount_659A__max",
        "credit_bureau_a_1__outstandingamount_362A__max",
        "derived__bureau_overdue2_to_outstanding2_max",
    ),
    ("applprev_1__annuity_853A__mean", "annuity_780A", "derived__prev_annuity_mean_to_current"),
    ("applprev_1__credamount_590A__mean", "credamount_770A", "derived__prev_credamount_mean_to_current"),
)


STABLE_RANGE_MAX_COLUMNS: tuple[str, ...] = (
    "credit_bureau_a_1__numberofoverdueinstlmax_1151L__max",
    "credit_bureau_a_1__lastupdate_388D__max",
    "credit_bureau_a_1__dateofrealrepmt_138D__max",
    "credit_bureau_a_1__dpdmax_757P__max",
    "credit_bureau_a_1__overdueamountmaxdateyear_994T__max",
    "credit_bureau_a_1__dpdmaxdateyear_896T__max",
    "credit_bureau_a_1__dateofcredend_353D__max",
    "credit_bureau_a_1__dateofcredstart_181D__max",
    "credit_bureau_a_1__nominalrate_498L__max",
    "credit_bureau_a_1__totalamount_6A__max",
    "credit_bureau_a_1__overdueamountmax_35A__max",
    "credit_bureau_a_1__numberofinstls_229L__max",
    "credit_bureau_a_1__overdueamountmax2_398A__max",
    "credit_bureau_a_1__dpdmax_139P__max",
    "credit_bureau_a_1__overdueamountmax_155A__max",
    "credit_bureau_a_1__numberofoverdueinstlmax_1039L__max",
    "credit_bureau_a_1__overdueamountmax2_14A__max",
    "tax_registry_c_1__pmtamount_36A__max",
    "credit_bureau_a_1__dpdmaxdatemonth_442T__max",
    "credit_bureau_a_1__overdueamountmaxdatemonth_284T__max",
    "credit_bureau_a_1__monthlyinstlamount_674A__max",
    "applprev_1__maxdpdtolerance_577P__max",
    "credit_bureau_a_1__numberofoverdueinstlmaxdat_148D__max",
)


def feature_set_components(feature_set: str) -> set[str]:
    if feature_set in ("", "none", "v5"):
        return set()
    if feature_set == "light1":
        return {"missing", "counts", "ratios", "ranges"}
    components = set(feature_set.split("+"))
    allowed = {"missing", "counts", "ratios", "ranges", "ranges_stable", "last"}
    unknown = components - allowed
    if unknown:
        raise ValueError(
            "Unknown feature_set components {!r}. Available: none, light1, or + combinations of {}".format(
                sorted(unknown), sorted(allowed)
            )
        )
    return components


def derived_output_columns(feature_set: str) -> list[str]:
    components = feature_set_components(feature_set)
    outputs: list[str] = []
    if "missing" in components:
        outputs.extend(["derived__null_count", "derived__amount_null_count"])
    if "ratios" in components:
        outputs.extend(alias for _, _, alias in RATIO_PAIRS)
    if "ranges_stable" in components:
        outputs.extend(f"{col[:-5]}__range" for col in STABLE_RANGE_MAX_COLUMNS)
    return outputs


def required_raw_columns_for_outputs(output_columns: list[str], feature_set: str) -> list[str]:
    required: set[str] = set(output_columns)
    components = feature_set_components(feature_set)
    outputs = set(output_columns)
    if "ratios" in components:
        for numerator, denominator, alias in RATIO_PAIRS:
            if alias in outputs:
                required.add(numerator)
                required.add(denominator)
    if "ranges_stable" in components:
        for col in STABLE_RANGE_MAX_COLUMNS:
            alias = f"{col[:-5]}__range"
            if alias in outputs:
                required.add(col)
                required.add(f"{col[:-5]}__min")
    return list(required)


def apply_derived_features(df: pl.DataFrame, feature_set: str) -> pl.DataFrame:
    components = feature_set_components(feature_set)
    if not components:
        return df

    protected = {"case_id", "target"}
    feature_cols = [col for col in df.columns if col not in protected]
    exprs: list[pl.Expr] = []

    if "missing" in components and feature_cols:
        exprs.append(
            pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int16) for col in feature_cols]).alias(
                "derived__null_count"
            )
        )
        amount_cols = [col for col in feature_cols if col.endswith("A")]
        if amount_cols:
            exprs.append(
                pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int16) for col in amount_cols]).alias(
                    "derived__amount_null_count"
                )
            )

    if "counts" in components:
        for col in [col for col in df.columns if col.endswith("__row_count")]:
            base = pl.col(col).cast(pl.Float64, strict=False)
            exprs.extend(
                [
                    (base > 0).cast(pl.Int8).alias(f"{col}__has_record"),
                    base.log1p().alias(f"{col}__log1p"),
                ]
            )

    if "ratios" in components:
        for numerator, denominator, alias in RATIO_PAIRS:
            if numerator in df.columns and denominator in df.columns:
                exprs.append(_safe_ratio_expr(numerator, denominator, alias))

    if "ranges_stable" in components:
        for col in STABLE_RANGE_MAX_COLUMNS:
            min_col = f"{col[:-5]}__min"
            if col in df.columns and min_col in df.columns:
                exprs.append(
                    (
                        pl.col(col).cast(pl.Float64, strict=False)
                        - pl.col(min_col).cast(pl.Float64, strict=False)
                    ).alias(f"{col[:-5]}__range")
                )

    if "ranges" in components:
        range_keywords = ("amount", "debt", "dpd", "annuity", "cred")
        range_exprs = []
        for col in df.columns:
            if not col.endswith("__max"):
                continue
            min_col = f"{col[:-5]}__min"
            if min_col not in df.columns:
                continue
            lowered = col.lower()
            if not any(keyword in lowered for keyword in range_keywords):
                continue
            range_exprs.append(
                (
                    pl.col(col).cast(pl.Float64, strict=False)
                    - pl.col(min_col).cast(pl.Float64, strict=False)
                ).alias(f"{col[:-5]}__range")
            )
            if len(range_exprs) >= 80:
                break
        exprs.extend(range_exprs)

    return df.with_columns(exprs) if exprs else df


def build_features(
    data_dir: Path,
    split: str,
    preset_name: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    sample_rows: int = 0,
    feature_set: str = "none",
    output_columns: list[str] | None = None,
) -> pl.DataFrame:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Available: {sorted(PRESETS)}")
    preset = PRESETS[preset_name]

    cache_path = None
    if cache_dir is not None and not sample_rows:
        cache_path = cache_dir / f"{split}_{preset_name}_{feature_set}_features.parquet"
        if use_cache and cache_path.exists():
            return pl.read_parquet(cache_path)

    lf = load_base(data_dir, split)
    if sample_rows:
        lf = lf.sort("case_id").head(sample_rows)
    case_ids = lf.select("case_id")
    for group in preset.depth0_groups:
        lf = lf.join(load_depth0(data_dir, split, group, case_ids), on="case_id", how="left")
    for group in preset.aggregate_groups:
        lf = lf.join(aggregate_group(data_dir, split, group, case_ids, feature_set), on="case_id", how="left")
    if output_columns is not None:
        wanted = set(required_raw_columns_for_outputs(output_columns, feature_set))
        wanted.update(["case_id", "target", "WEEK_NUM"])
        schema = lf.collect_schema()
        lf = lf.select([col for col in schema.names() if col in wanted])
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
    df = apply_derived_features(df, feature_set)
    if output_columns is not None:
        wanted = set(output_columns)
        wanted.update(["case_id", "target"])
        df = df.select([col for col in df.columns if col in wanted])
    if cache_path is not None and not sample_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
    return df
