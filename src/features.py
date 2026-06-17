from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


SPECIAL_COLUMNS = {"case_id", "num_group1", "num_group2", "__decision_date"}


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

TWOSTAGE_LARGE_GROUPS = {"credit_bureau_a_2"}
TWOSTAGE_OVERDUE_COLUMNS = {
    "pmts_dpd_1073P",
    "pmts_dpd_303P",
    "pmts_overdue_1140A",
    "pmts_overdue_1152A",
}
TWOSTAGE_ACTIVE_COLUMNS = {
    "pmts_dpd_1073P",
    "pmts_overdue_1140A",
    "pmts_month_158T",
    "pmts_year_1139T",
}
TWOSTAGE_CLOSED_COLUMNS = {
    "pmts_dpd_303P",
    "pmts_overdue_1152A",
    "pmts_month_706T",
    "pmts_year_507T",
}

EX2_BUREAU_A1_EXTRA_COLUMNS = (
    "annualeffectiverate_199L",
    "annualeffectiverate_63L",
    "contractsum_5085717L",
    "credlmt_230A",
    "credlmt_935A",
    "nominalrate_281L",
    "nominalrate_498L",
    "numberofcontrsvalue_258L",
    "numberofcontrsvalue_358L",
    "numberofinstls_229L",
    "numberofinstls_320L",
    "numberofoutstandinstls_520L",
    "numberofoutstandinstls_59L",
    "numberofoverdueinstlmax_1039L",
    "numberofoverdueinstlmax_1151L",
    "numberofoverdueinstls_725L",
    "numberofoverdueinstls_834L",
)
EX2_PERSON1_FIRST_COLUMNS = (
    "empl_employedtotal_800L",
    "empl_employedfrom_271D",
    "empl_industry_691L",
    "familystate_447L",
    "incometype_1044T",
    "sex_738L",
    "housetype_905L",
    "housingtype_772L",
    "isreference_387L",
    "birth_259D",
)
EX2_PERSON2_FIRST_LAST_COLUMNS = (
    "empls_economicalst_849M",
    "empls_employedfrom_796D",
    "empls_employer_name_740M",
)


def ex2_relevant_column(col: str) -> bool:
    return col not in SPECIAL_COLUMNS and (col[-1:] in ("T", "L", "M", "D", "P", "A") or "num_group" in col)


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


def is_date_column(col: str) -> bool:
    return col.endswith("D") or col == "date_decision"


def column_family(col: str) -> str:
    lowered = col.lower()
    if any(token in lowered for token in ("dpd", "overdue", "pastdue")) or col.endswith("P"):
        return "delinquency"
    if any(
        token in lowered
        for token in (
            "amount",
            "amt",
            "annuity",
            "balance",
            "cred",
            "debt",
            "income",
            "price",
            "sum",
            "turnover",
            "pmtamount",
        )
    ) or col.endswith("A"):
        return "amount"
    if any(token in lowered for token in ("num", "cnt", "count", "instl", "term", "month", "year")) or col.endswith("L"):
        return "count_term"
    if "rate" in lowered or "eir" in lowered or "interest" in lowered:
        return "rate"
    if is_date_column(col):
        return "date"
    return "other"


def numeric_expr(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64, strict=False)


def custom_numeric_ops(
    group: str,
    col: str,
    dtype: pl.DataType,
    components: set[str] | None = None,
) -> tuple[str, ...]:
    components = components or set()
    family = column_family(col)
    if dtype == pl.Boolean:
        return ("max", "last")

    if group.startswith("person_"):
        if "custom_person_narrow_aggs" in components:
            if "income" in col.lower():
                return ("max",)
            if is_date_column(col):
                return ("max", "min")
            if family in {"count_term", "amount"}:
                return ("max", "min")
            return ("max",)
        if "income" in col.lower():
            return ("max", "first", "last")
        if is_date_column(col):
            return ("first", "last", "max", "min")
        return ("first", "last")

    if group in {"deposit_1", "debitcard_1"}:
        return ("max", "min")

    if group == "other_1":
        return ("first",)

    if group.startswith("tax_registry_"):
        if "custom_tax_narrow_aggs" in components:
            if family == "amount":
                return ("max", "min", "mean", "std")
            if family == "date":
                return ("max", "min")
            return ("max", "min")
        return ("max", "min", "first", "last", "mean", "std")

    if group == "credit_bureau_a_1":
        if "custom_bureau_a1_narrow_aggs" in components:
            if family == "delinquency":
                return ("max", "mean", "std")
            if family in {"amount", "rate", "count_term"}:
                return ("max", "min", "mean", "std")
            if family == "date":
                return ("max", "min")
            return ("max", "min")
        if family in {"amount", "rate", "count_term", "delinquency"}:
            return ("max", "min", "mean", "std", "sum", "median", "first")
        if family == "date":
            return ("max", "min", "first", "last")
        return ("max", "min")

    if group == "applprev_1":
        if "custom_applprev_narrow_aggs" in components:
            if family == "delinquency":
                return ("max", "mean", "std")
            if family in {"amount", "count_term", "rate"}:
                return ("max", "min", "mean", "std")
            if family == "date":
                return ("max", "min")
            return ("max", "min")
        if family in {"amount", "count_term", "delinquency"}:
            return ("max", "min", "mean", "std", "sum", "median")
        if family == "date":
            return ("max", "min", "first", "last")
        return ("max", "min")

    if group == "applprev_2":
        if "custom_applprev_narrow_aggs" in components:
            return ("nunique",)
        return ("first", "last", "nunique")

    if group.startswith("credit_bureau_b_"):
        if "custom_bureau_b_narrow_aggs" in components:
            if family == "delinquency":
                return ("max", "mean", "std")
            if family in {"amount", "count_term", "rate"}:
                return ("max", "min", "mean", "std")
            if family == "date":
                return ("max", "min")
            return ("max", "min")
        if family in {"amount", "count_term", "delinquency"}:
            return ("max", "min", "mean", "std", "sum")
        if family == "date":
            return ("max", "min", "first", "last")
        return ("max", "min")

    return ("max", "min")


def custom_categorical_ops(group: str, col: str, components: set[str] | None = None) -> tuple[str, ...]:
    components = components or set()
    if group == "other_1":
        return ("first",)
    if group.startswith("person_"):
        if "custom_person_narrow_aggs" in components:
            return ("nunique",)
        return ("first", "last")
    if group in {"deposit_1", "debitcard_1"}:
        return ("first", "last", "nunique")
    if group.startswith("tax_registry_"):
        if "custom_tax_narrow_aggs" in components:
            return ("nunique",)
        return ("first", "last", "nunique")
    if group in {"applprev_1", "applprev_2"}:
        if "custom_applprev_narrow_aggs" in components:
            return ("nunique",)
        return ("first", "last", "nunique")
    if group == "credit_bureau_a_1" and "custom_bureau_a1_narrow_aggs" in components:
        return ("nunique",)
    if group.startswith("credit_bureau_a_") or group.startswith("credit_bureau_b_"):
        if group.startswith("credit_bureau_b_") and "custom_bureau_b_narrow_aggs" in components:
            return ("nunique",)
        return ("first", "last", "nunique")
    return ("nunique",)


def append_agg_ops(aggs: list[pl.Expr], base: pl.Expr, out: str, ops: tuple[str, ...]) -> None:
    for op in ops:
        if op == "mean":
            aggs.append(base.mean().alias(f"{out}__mean"))
        elif op == "max":
            aggs.append(base.max().alias(f"{out}__max"))
        elif op == "min":
            aggs.append(base.min().alias(f"{out}__min"))
        elif op == "std":
            aggs.append(base.std().alias(f"{out}__std"))
        elif op == "sum":
            aggs.append(base.sum().alias(f"{out}__sum"))
        elif op == "median":
            aggs.append(base.median().alias(f"{out}__median"))
        elif op == "first":
            aggs.append(base.first().alias(f"{out}__first"))
        elif op == "last":
            aggs.append(base.last().alias(f"{out}__last"))
        elif op == "nunique":
            aggs.append(base.n_unique().alias(f"{out}__nunique"))
        else:
            raise ValueError(f"Unknown aggregation op: {op}")


def ex2_base_expr(col: str, schema: pl.Schema) -> pl.Expr:
    base = numeric_expr(col)
    if is_date_column(col) and "__decision_date" in schema:
        return numeric_expr("__decision_date") - base
    return base


def ex2_general_aggs(group: str, schema: pl.Schema, include_extended_la: bool = True) -> list[pl.Expr]:
    aggs: list[pl.Expr] = []
    cols = [col for col in schema.names() if ex2_relevant_column(col)]
    for col in cols:
        append_agg_ops(aggs, ex2_base_expr(col, schema), f"{group}__{col}", ("max", "min"))
    if include_extended_la:
        for col in [col for col in cols if col[-1:] in ("L", "A")]:
            append_agg_ops(aggs, ex2_base_expr(col, schema), f"{group}__{col}", ("mean", "std", "sum", "median"))
    return aggs


def ex2_aggregate_exprs(group: str, schema: pl.Schema) -> list[pl.Expr]:
    aggs: list[pl.Expr] = [pl.len().alias(f"{group}__row_count")]

    if group in {"credit_bureau_b_1", "credit_bureau_b_2", "applprev_2"}:
        return aggs

    if group in {"deposit_1", "debitcard_1"}:
        for col in [col for col in schema.names() if ex2_relevant_column(col)]:
            append_agg_ops(aggs, ex2_base_expr(col, schema), f"{group}__{col}", ("max", "min"))
        return aggs

    if group == "person_1":
        for col in EX2_PERSON1_FIRST_COLUMNS:
            if col not in schema:
                continue
            base = ex2_base_expr(col, schema) if is_date_column(col) or schema[col].is_numeric() else pl.col(col)
            append_agg_ops(aggs, base, f"{group}__{col}", ("first",))
        if "mainoccupationinc_384A" in schema:
            income = numeric_expr("mainoccupationinc_384A")
            aggs.append(income.max().alias(f"{group}__mainoccupationinc_384A__max"))
            if "incometype_1044T" in schema:
                aggs.append(
                    income.filter(pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias(
                        f"{group}__mainoccupationinc_384A__selfemployed_max"
                    )
                )
        return aggs

    if group == "person_2":
        for col in EX2_PERSON2_FIRST_LAST_COLUMNS:
            if col not in schema:
                continue
            base = ex2_base_expr(col, schema) if is_date_column(col) or schema[col].is_numeric() else pl.col(col)
            append_agg_ops(aggs, base, f"{group}__{col}", ("first", "last"))
        return aggs

    if group == "other_1":
        for col in schema.names():
            if col in SPECIAL_COLUMNS or "num_group" in col:
                continue
            base = numeric_expr(col) if schema[col].is_numeric() or schema[col] == pl.Boolean else pl.col(col)
            append_agg_ops(aggs, base, f"{group}__{col}", ("first",))
        return aggs

    if group == "tax_registry_a_1":
        for col in [col for col in schema.names() if ex2_relevant_column(col)]:
            append_agg_ops(
                aggs,
                ex2_base_expr(col, schema),
                f"{group}__{col}",
                ("max", "min", "last", "first", "mean", "std"),
            )
        if "amount_4527230A" in schema:
            amount = numeric_expr("amount_4527230A")
            aggs.append((amount.max() - amount.min()).alias(f"{group}__amount_4527230A__gap"))
        return aggs

    if group == "credit_bureau_a_1":
        for col in [col for col in schema.names() if ex2_relevant_column(col)]:
            append_agg_ops(aggs, ex2_base_expr(col, schema), f"{group}__{col}", ("max", "min"))
        for col in EX2_BUREAU_A1_EXTRA_COLUMNS:
            if col in schema:
                append_agg_ops(
                    aggs,
                    ex2_base_expr(col, schema),
                    f"{group}__{col}",
                    ("mean", "std", "sum", "median", "first"),
                )
        return aggs

    return aggs + ex2_general_aggs(group, schema)


def use_custom_aggs_for_group(components: set[str], group: str) -> bool:
    if "ex2_data" in components:
        return True
    if "table_custom_aggs" in components:
        return True
    if "custom_person_aggs" in components and group.startswith("person_"):
        return True
    if "custom_person_narrow_aggs" in components and group.startswith("person_"):
        return True
    if "custom_deposit_debitcard_aggs" in components and group in {"deposit_1", "debitcard_1"}:
        return True
    if "custom_bureau_a1_aggs" in components and group == "credit_bureau_a_1":
        return True
    if "custom_bureau_a1_narrow_aggs" in components and group == "credit_bureau_a_1":
        return True
    if "custom_applprev_aggs" in components and group.startswith("applprev_"):
        return True
    if "custom_applprev_narrow_aggs" in components and group.startswith("applprev_"):
        return True
    if "custom_tax_aggs" in components and group.startswith("tax_registry_"):
        return True
    if "custom_tax_narrow_aggs" in components and group.startswith("tax_registry_"):
        return True
    if "custom_bureau_b_narrow_aggs" in components and group.startswith("credit_bureau_b_"):
        return True
    if "custom_other_aggs" in components and group == "other_1":
        return True
    return False


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
    decision_dates: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    lf = normalize_dates(scan_group(data_dir, split, group))
    lf = filter_cases(lf, case_ids)
    schema = lf.collect_schema()
    components = feature_set_components(feature_set)
    custom_aggs = use_custom_aggs_for_group(components, group)
    if custom_aggs and decision_dates is not None:
        lf = lf.join(decision_dates, on="case_id", how="left")
        schema = lf.collect_schema()
    if "last" in components and "num_group1" in schema:
        lf = lf.sort(["case_id", "num_group1"])
    elif custom_aggs and "num_group1" in schema:
        lf = lf.sort(["case_id", "num_group1"])
    if "ex2_data" in components:
        return lf.group_by("case_id").agg(ex2_aggregate_exprs(group, schema))
    aggs: list[pl.Expr] = [pl.len().alias(f"{group}__row_count")]

    for col, dtype in schema.items():
        if col in SPECIAL_COLUMNS:
            continue
        out = f"{group}__{col}"
        if custom_aggs:
            if dtype.is_numeric() or dtype == pl.Boolean or is_date_column(col):
                base = numeric_expr(col)
                if is_date_column(col) and "__decision_date" in schema:
                    base = numeric_expr("__decision_date") - numeric_expr(col)
                append_agg_ops(aggs, base, out, custom_numeric_ops(group, col, dtype, components))
                if group.startswith("tax_registry_") and column_family(col) == "amount":
                    aggs.append((base.max() - base.min()).alias(f"{out}__gap"))
                if is_date_column(col) and (
                    "table_custom_aggs" in components
                    or "custom_bureau_a1_narrow_aggs" in components
                    or "custom_applprev_narrow_aggs" in components
                    or "custom_bureau_b_narrow_aggs" in components
                    or "custom_tax_narrow_aggs" in components
                ):
                    aggs.append((base.max() - base.min()).alias(f"{out}__gap"))
            else:
                append_agg_ops(aggs, pl.col(col), out, custom_categorical_ops(group, col, components))
        elif dtype.is_numeric() or dtype == pl.Boolean or col.endswith("D"):
            base = numeric_expr(col)
            append_agg_ops(aggs, base, out, ("mean", "max", "min", "std"))
            if "last" in components:
                append_agg_ops(aggs, base, out, ("last",))
        else:
            append_agg_ops(aggs, pl.col(col), out, ("nunique",))

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


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def aggregate_large_group_twostage_eager(
    data_dir: Path,
    split: str,
    group: str,
    temp_dir: Path,
    case_ids: pl.DataFrame | None = None,
) -> pl.DataFrame:
    selected = LITE_LARGE_NUMERIC_COLUMNS[group]
    contract_dir = temp_dir / "contract_partials"
    case_dir = temp_dir / "case_partials"
    contract_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    case_ids_lf = case_ids.lazy() if case_ids is not None else None
    case_paths: list[Path] = []

    for idx, path in enumerate(files_for_group(data_dir, split, group)):
        schema = pl.scan_parquet(str(path)).collect_schema()
        available = [col for col in selected if col in schema]
        lf = pl.scan_parquet(str(path)).select(["case_id", "num_group1", "num_group2", *available])
        lf = lf.with_columns(
            [
                pl.col("case_id").cast(pl.Int64),
                pl.col("num_group1").cast(pl.Int64, strict=False),
                pl.col("num_group2").cast(pl.Int64, strict=False),
            ]
        )
        lf = filter_cases(lf, case_ids_lf)

        contract_aggs: list[pl.Expr] = [
            pl.len().alias(f"{group}__contract__payment_count"),
            pl.col("num_group2").count().alias(f"{group}__contract__num_group2_count"),
            pl.col("num_group2").max().alias(f"{group}__contract__num_group2_max"),
            pl.col("num_group2").min().alias(f"{group}__contract__num_group2_min"),
        ]
        overdue_flags: list[pl.Expr] = []
        dpd_gt0_exprs: list[pl.Expr] = []
        dpd_gt30_exprs: list[pl.Expr] = []
        dpd_gt60_exprs: list[pl.Expr] = []
        dpd_gt90_exprs: list[pl.Expr] = []
        overdue_amt_gt0_exprs: list[pl.Expr] = []
        active_payment_exprs: list[pl.Expr] = []
        closed_payment_exprs: list[pl.Expr] = []
        active_dpd_gt30_exprs: list[pl.Expr] = []
        closed_dpd_gt30_exprs: list[pl.Expr] = []
        active_overdue_amt_gt0_exprs: list[pl.Expr] = []
        closed_overdue_amt_gt0_exprs: list[pl.Expr] = []
        for col in available:
            base = pl.col(col).cast(pl.Float64, strict=False)
            prefix = f"{group}__contract__{col}"
            contract_aggs.extend(
                [
                    base.sum().alias(f"{prefix}__sum"),
                    base.count().alias(f"{prefix}__count"),
                    base.mean().alias(f"{prefix}__mean"),
                    base.max().alias(f"{prefix}__max"),
                    base.min().alias(f"{prefix}__min"),
                    base.std().alias(f"{prefix}__std"),
                    (base.max() - base.min()).alias(f"{prefix}__range"),
                ]
            )
            if col in TWOSTAGE_OVERDUE_COLUMNS:
                overdue_flags.append(base.fill_null(0) > 0)
            if col in TWOSTAGE_ACTIVE_COLUMNS:
                active_payment_exprs.append(base.is_not_null())
            if col in TWOSTAGE_CLOSED_COLUMNS:
                closed_payment_exprs.append(base.is_not_null())
            if col in {"pmts_dpd_1073P", "pmts_dpd_303P"}:
                clean = base.fill_null(0)
                dpd_gt0_exprs.append(clean > 0)
                dpd_gt30_exprs.append(clean > 30)
                dpd_gt60_exprs.append(clean > 60)
                dpd_gt90_exprs.append(clean > 90)
                if col == "pmts_dpd_1073P":
                    active_dpd_gt30_exprs.append(clean > 30)
                if col == "pmts_dpd_303P":
                    closed_dpd_gt30_exprs.append(clean > 30)
            if col in {"pmts_overdue_1140A", "pmts_overdue_1152A"}:
                overdue_amt_gt0_exprs.append(base.fill_null(0) > 0)
                if col == "pmts_overdue_1140A":
                    active_overdue_amt_gt0_exprs.append(base.fill_null(0) > 0)
                if col == "pmts_overdue_1152A":
                    closed_overdue_amt_gt0_exprs.append(base.fill_null(0) > 0)
        if overdue_flags:
            contract_aggs.append(
                pl.any_horizontal(overdue_flags).max().cast(pl.Int8).alias(f"{group}__contract__has_overdue")
            )
        else:
            contract_aggs.append(pl.lit(None).cast(pl.Int8).alias(f"{group}__contract__has_overdue"))
        if dpd_gt0_exprs:
            contract_aggs.extend(
                [
                    pl.sum_horizontal([expr.cast(pl.Int32) for expr in dpd_gt0_exprs]).sum().alias(
                        f"{group}__contract__dpd_gt0_payment_count"
                    ),
                    pl.sum_horizontal([expr.cast(pl.Int32) for expr in dpd_gt30_exprs]).sum().alias(
                        f"{group}__contract__dpd_gt30_payment_count"
                    ),
                    pl.sum_horizontal([expr.cast(pl.Int32) for expr in dpd_gt60_exprs]).sum().alias(
                        f"{group}__contract__dpd_gt60_payment_count"
                    ),
                    pl.sum_horizontal([expr.cast(pl.Int32) for expr in dpd_gt90_exprs]).sum().alias(
                        f"{group}__contract__dpd_gt90_payment_count"
                    ),
                    pl.any_horizontal(dpd_gt30_exprs).max().cast(pl.Int8).alias(
                        f"{group}__contract__has_dpd_gt30"
                    ),
                    pl.any_horizontal(dpd_gt60_exprs).max().cast(pl.Int8).alias(
                        f"{group}__contract__has_dpd_gt60"
                    ),
                    pl.any_horizontal(dpd_gt90_exprs).max().cast(pl.Int8).alias(
                        f"{group}__contract__has_dpd_gt90"
                    ),
                ]
            )
        if overdue_amt_gt0_exprs:
            contract_aggs.append(
                pl.sum_horizontal([expr.cast(pl.Int32) for expr in overdue_amt_gt0_exprs]).sum().alias(
                    f"{group}__contract__overdue_amt_gt0_payment_count"
                )
            )
        if active_payment_exprs:
            active_payment_flag = pl.any_horizontal(active_payment_exprs)
            contract_aggs.extend(
                [
                    active_payment_flag.cast(pl.Int32).sum().alias(
                        f"{group}__contract__active_payment_count"
                    ),
                    active_payment_flag.max().cast(pl.Int8).alias(f"{group}__contract__has_active_payment"),
                ]
            )
        if closed_payment_exprs:
            closed_payment_flag = pl.any_horizontal(closed_payment_exprs)
            contract_aggs.extend(
                [
                    closed_payment_flag.cast(pl.Int32).sum().alias(
                        f"{group}__contract__closed_payment_count"
                    ),
                    closed_payment_flag.max().cast(pl.Int8).alias(f"{group}__contract__has_closed_payment"),
                ]
            )
        if active_dpd_gt30_exprs:
            contract_aggs.append(
                pl.sum_horizontal([expr.cast(pl.Int32) for expr in active_dpd_gt30_exprs]).sum().alias(
                    f"{group}__contract__active_dpd_gt30_payment_count"
                )
            )
        if closed_dpd_gt30_exprs:
            contract_aggs.append(
                pl.sum_horizontal([expr.cast(pl.Int32) for expr in closed_dpd_gt30_exprs]).sum().alias(
                    f"{group}__contract__closed_dpd_gt30_payment_count"
                )
            )
        if active_overdue_amt_gt0_exprs:
            contract_aggs.append(
                pl.sum_horizontal([expr.cast(pl.Int32) for expr in active_overdue_amt_gt0_exprs]).sum().alias(
                    f"{group}__contract__active_overdue_amt_gt0_payment_count"
                )
            )
        if closed_overdue_amt_gt0_exprs:
            contract_aggs.append(
                pl.sum_horizontal([expr.cast(pl.Int32) for expr in closed_overdue_amt_gt0_exprs]).sum().alias(
                    f"{group}__contract__closed_overdue_amt_gt0_payment_count"
                )
            )
        if {"pmts_year_1139T", "pmts_month_158T"}.issubset(available):
            active_period = (
                pl.col("pmts_year_1139T").cast(pl.Float64, strict=False) * 12
                + pl.col("pmts_month_158T").cast(pl.Float64, strict=False)
            )
            active_bad = pl.col("pmts_dpd_1073P").cast(pl.Float64, strict=False).fill_null(0) > 30
            contract_aggs.extend(
                [
                    active_period.max().alias(f"{group}__contract__active_period_max"),
                    active_period.min().alias(f"{group}__contract__active_period_min"),
                    (active_period.max() - active_period.min()).alias(f"{group}__contract__active_period_span"),
                    pl.when(active_bad).then(active_period).otherwise(None).max().alias(
                        f"{group}__contract__active_dpd_gt30_period_max"
                    ),
                ]
            )
        if {"pmts_year_507T", "pmts_month_706T"}.issubset(available):
            closed_period = (
                pl.col("pmts_year_507T").cast(pl.Float64, strict=False) * 12
                + pl.col("pmts_month_706T").cast(pl.Float64, strict=False)
            )
            closed_bad = pl.col("pmts_dpd_303P").cast(pl.Float64, strict=False).fill_null(0) > 30
            contract_aggs.extend(
                [
                    closed_period.max().alias(f"{group}__contract__closed_period_max"),
                    closed_period.min().alias(f"{group}__contract__closed_period_min"),
                    (closed_period.max() - closed_period.min()).alias(f"{group}__contract__closed_period_span"),
                    pl.when(closed_bad).then(closed_period).otherwise(None).max().alias(
                        f"{group}__contract__closed_dpd_gt30_period_max"
                    ),
                ]
            )

        contracts = lf.group_by(["case_id", "num_group1"]).agg(contract_aggs).collect(engine="streaming")
        contract_path = contract_dir / f"{split}_{group}_contract_partial_{idx}.parquet"
        contracts.write_parquet(contract_path)
        del contracts, lf

        contract_lf = pl.scan_parquet(str(contract_path))
        contract_schema = contract_lf.collect_schema()
        case_aggs: list[pl.Expr] = [
            pl.len().alias(f"{group}__case_partial__contract_count"),
            pl.col(f"{group}__contract__payment_count").sum().alias(f"{group}__case_partial__payment_count_sum"),
            pl.col(f"{group}__contract__payment_count").max().alias(f"{group}__case_partial__payment_count_max"),
            pl.col(f"{group}__contract__payment_count").mean().alias(f"{group}__case_partial__payment_count_mean"),
            pl.col(f"{group}__contract__has_overdue").sum().alias(
                f"{group}__case_partial__overdue_contract_count"
            ),
        ]
        semantic_contract_cols = [
            "dpd_gt0_payment_count",
            "dpd_gt30_payment_count",
            "dpd_gt60_payment_count",
            "dpd_gt90_payment_count",
            "overdue_amt_gt0_payment_count",
            "has_dpd_gt30",
            "has_dpd_gt60",
            "has_dpd_gt90",
            "active_payment_count",
            "closed_payment_count",
            "has_active_payment",
            "has_closed_payment",
            "active_dpd_gt30_payment_count",
            "closed_dpd_gt30_payment_count",
            "active_overdue_amt_gt0_payment_count",
            "closed_overdue_amt_gt0_payment_count",
            "active_period_max",
            "active_period_min",
            "active_period_span",
            "active_dpd_gt30_period_max",
            "closed_period_max",
            "closed_period_min",
            "closed_period_span",
            "closed_dpd_gt30_period_max",
        ]
        for suffix in semantic_contract_cols:
            contract_col = f"{group}__contract__{suffix}"
            if contract_col in contract_schema:
                if suffix.endswith("_period_max") or suffix.endswith("_period_span"):
                    case_aggs.append(pl.col(contract_col).max().alias(f"{group}__case_partial__{suffix}_max"))
                elif suffix.endswith("_period_min"):
                    case_aggs.append(pl.col(contract_col).min().alias(f"{group}__case_partial__{suffix}_min"))
                else:
                    case_aggs.append(pl.col(contract_col).sum().alias(f"{group}__case_partial__{suffix}_sum"))
        for col in selected:
            prefix = f"{group}__contract__{col}"
            sum_col = f"{prefix}__sum"
            count_col = f"{prefix}__count"
            max_col = f"{prefix}__max"
            min_col = f"{prefix}__min"
            mean_col = f"{prefix}__mean"
            std_col = f"{prefix}__std"
            range_col = f"{prefix}__range"
            out = f"{group}__{col}"
            if sum_col not in contract_schema:
                continue
            case_aggs.extend(
                [
                    pl.col(sum_col).sum().alias(f"{out}__sum_sum"),
                    pl.col(count_col).sum().alias(f"{out}__count_sum"),
                    pl.col(max_col).max().alias(f"{out}__contract_max"),
                    pl.col(min_col).min().alias(f"{out}__contract_min"),
                    pl.col(mean_col).mean().alias(f"{out}__contract_mean_mean"),
                    pl.col(mean_col).max().alias(f"{out}__contract_mean_max"),
                    pl.col(std_col).mean().alias(f"{out}__contract_std_mean"),
                    pl.col(range_col).max().alias(f"{out}__contract_range_max"),
                ]
            )

        case_partial = contract_lf.group_by("case_id").agg(case_aggs).collect(engine="streaming")
        case_path = case_dir / f"{split}_{group}_case_partial_{idx}.parquet"
        case_partial.write_parquet(case_path)
        case_paths.append(case_path)
        contract_path.unlink(missing_ok=True)
        del case_partial, contract_lf

    case_lf = pl.scan_parquet([str(path) for path in case_paths])
    case_schema = case_lf.collect_schema()
    final_aggs: list[pl.Expr] = [
        pl.col(f"{group}__case_partial__contract_count").sum().alias(f"{group}__contract_count"),
        pl.col(f"{group}__case_partial__payment_count_sum").sum().alias(f"{group}__payment_count"),
        pl.col(f"{group}__case_partial__payment_count_max").max().alias(f"{group}__payment_count_contract_max"),
        pl.col(f"{group}__case_partial__payment_count_mean").mean().alias(
            f"{group}__payment_count_contract_mean"
        ),
        pl.col(f"{group}__case_partial__overdue_contract_count").sum().alias(
            f"{group}__overdue_contract_count"
        ),
    ]
    semantic_case_cols = [
        "dpd_gt0_payment_count",
        "dpd_gt30_payment_count",
        "dpd_gt60_payment_count",
        "dpd_gt90_payment_count",
        "overdue_amt_gt0_payment_count",
        "has_dpd_gt30",
        "has_dpd_gt60",
        "has_dpd_gt90",
        "active_payment_count",
        "closed_payment_count",
        "has_active_payment",
        "has_closed_payment",
        "active_dpd_gt30_payment_count",
        "closed_dpd_gt30_payment_count",
        "active_overdue_amt_gt0_payment_count",
        "closed_overdue_amt_gt0_payment_count",
        "active_period_max",
        "active_period_min",
        "active_period_span",
        "active_dpd_gt30_period_max",
        "closed_period_max",
        "closed_period_min",
        "closed_period_span",
        "closed_dpd_gt30_period_max",
    ]
    for suffix in semantic_case_cols:
        sum_col = f"{group}__case_partial__{suffix}_sum"
        max_col = f"{group}__case_partial__{suffix}_max"
        min_col = f"{group}__case_partial__{suffix}_min"
        if sum_col in case_schema:
            final_aggs.append(pl.col(sum_col).sum().alias(f"{group}__{suffix}"))
        elif max_col in case_schema:
            final_aggs.append(pl.col(max_col).max().alias(f"{group}__{suffix}"))
        elif min_col in case_schema:
            final_aggs.append(pl.col(min_col).min().alias(f"{group}__{suffix}"))
    for col in selected:
        out = f"{group}__{col}"
        sum_col = f"{out}__sum_sum"
        count_col = f"{out}__count_sum"
        max_col = f"{out}__contract_max"
        min_col = f"{out}__contract_min"
        mean_mean_col = f"{out}__contract_mean_mean"
        mean_max_col = f"{out}__contract_mean_max"
        std_mean_col = f"{out}__contract_std_mean"
        range_max_col = f"{out}__contract_range_max"
        if sum_col not in case_schema:
            continue
        count_sum = pl.col(count_col).sum()
        final_aggs.extend(
            [
                pl.col(sum_col).sum().alias(f"{out}__sum"),
                count_sum.alias(f"{out}__count"),
                _safe_div(pl.col(sum_col).sum(), count_sum).alias(f"{out}__mean"),
                pl.col(max_col).max().alias(f"{out}__max"),
                pl.col(min_col).min().alias(f"{out}__min"),
                pl.col(mean_mean_col).mean().alias(f"{out}__contract_mean_mean"),
                pl.col(mean_max_col).max().alias(f"{out}__contract_mean_max"),
                pl.col(std_mean_col).mean().alias(f"{out}__contract_std_mean"),
                pl.col(range_max_col).max().alias(f"{out}__contract_range_max"),
            ]
        )

    return case_lf.group_by("case_id").agg(final_aggs).collect(engine="streaming")


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
    allowed = {
        "missing",
        "counts",
        "ratios",
        "ranges",
        "ranges_stable",
        "last",
        "a2_twostage",
        "a2_delinquency",
        "a2_delinquency_stable",
        "a2_payment_mix",
        "a2_temporal",
        "a2_dpd30_contract",
        "table_custom_aggs",
        "custom_person_aggs",
        "custom_person_narrow_aggs",
        "custom_deposit_debitcard_aggs",
        "custom_bureau_a1_aggs",
        "custom_bureau_a1_narrow_aggs",
        "custom_applprev_aggs",
        "custom_applprev_narrow_aggs",
        "custom_tax_aggs",
        "custom_tax_narrow_aggs",
        "custom_bureau_b_narrow_aggs",
        "custom_other_aggs",
        "ex2_data",
        "semantic_v2",
        "semantic_v2_stable",
    }
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
    if "a2_twostage" in components:
        group = "credit_bureau_a_2"
        outputs.extend(
            [
                f"{group}__contract_count",
                f"{group}__payment_count",
                f"{group}__payment_count_contract_max",
                f"{group}__payment_count_contract_mean",
                f"{group}__overdue_contract_count",
            ]
        )
        for col in LITE_LARGE_NUMERIC_COLUMNS[group]:
            prefix = f"{group}__{col}"
            outputs.extend(
                [
                    f"{prefix}__sum",
                    f"{prefix}__count",
                    f"{prefix}__mean",
                    f"{prefix}__max",
                    f"{prefix}__min",
                    f"{prefix}__contract_mean_mean",
                    f"{prefix}__contract_mean_max",
                    f"{prefix}__contract_std_mean",
                    f"{prefix}__contract_range_max",
                ]
            )
    if "a2_delinquency" in components:
        group = "credit_bureau_a_2"
        outputs.extend(
            [
                f"{group}__dpd_gt0_payment_count",
                f"{group}__dpd_gt30_payment_count",
                f"{group}__dpd_gt60_payment_count",
                f"{group}__dpd_gt90_payment_count",
                f"{group}__overdue_amt_gt0_payment_count",
                f"{group}__has_dpd_gt30",
                f"{group}__has_dpd_gt60",
                f"{group}__has_dpd_gt90",
                "derived__a2_overdue_contract_rate",
                "derived__a2_dpd_gt0_payment_rate",
                "derived__a2_dpd_gt30_payment_rate",
                "derived__a2_dpd_gt60_payment_rate",
                "derived__a2_dpd_gt90_payment_rate",
                "derived__a2_overdue_amt_gt0_payment_rate",
                "derived__a2_dpd_gt30_contract_rate",
                "derived__a2_dpd_gt60_contract_rate",
                "derived__a2_dpd_gt90_contract_rate",
                "derived__a2_payment_per_contract",
            ]
        )
    if "a2_delinquency_stable" in components:
        group = "credit_bureau_a_2"
        outputs.extend(
            [
                f"{group}__has_dpd_gt30",
                f"{group}__has_dpd_gt60",
                f"{group}__has_dpd_gt90",
                f"{group}__dpd_gt30_payment_count",
                f"{group}__dpd_gt60_payment_count",
                f"{group}__dpd_gt90_payment_count",
                "derived__a2_dpd_gt30_contract_rate",
                "derived__a2_dpd_gt60_contract_rate",
                "derived__a2_dpd_gt90_contract_rate",
                "derived__a2_payment_per_contract",
            ]
        )
    if "a2_dpd30_contract" in components:
        group = "credit_bureau_a_2"
        outputs.extend(
            [
                f"{group}__has_dpd_gt30",
                "derived__a2_dpd_gt30_contract_rate",
            ]
        )
    if "a2_payment_mix" in components:
        group = "credit_bureau_a_2"
        outputs.extend(
            [
                f"{group}__active_payment_count",
                f"{group}__closed_payment_count",
                f"{group}__has_active_payment",
                f"{group}__has_closed_payment",
                f"{group}__active_dpd_gt30_payment_count",
                f"{group}__closed_dpd_gt30_payment_count",
                f"{group}__active_overdue_amt_gt0_payment_count",
                f"{group}__closed_overdue_amt_gt0_payment_count",
                "derived__a2_active_payment_share",
                "derived__a2_closed_payment_share",
                "derived__a2_active_contract_rate",
                "derived__a2_closed_contract_rate",
                "derived__a2_active_dpd_gt30_payment_rate",
                "derived__a2_closed_dpd_gt30_payment_rate",
                "derived__a2_active_overdue_amt_gt0_payment_rate",
                "derived__a2_closed_overdue_amt_gt0_payment_rate",
                "derived__a2_active_minus_closed_dpd_gt30_rate",
                "derived__a2_active_minus_closed_overdue_amt_rate",
            ]
        )
    if "a2_temporal" in components:
        group = "credit_bureau_a_2"
        outputs.extend(
            [
                f"{group}__active_period_max",
                f"{group}__active_period_min",
                f"{group}__active_period_span",
                f"{group}__active_dpd_gt30_period_max",
                f"{group}__closed_period_max",
                f"{group}__closed_period_min",
                f"{group}__closed_period_span",
                f"{group}__closed_dpd_gt30_period_max",
                "derived__a2_active_period_span_ratio",
                "derived__a2_closed_period_span_ratio",
                "derived__a2_active_closed_latest_period_delta",
                "derived__a2_active_bad_latest_period_gap",
                "derived__a2_closed_bad_latest_period_gap",
            ]
        )
    if "semantic_v2" in components or "semantic_v2_stable" in components:
        outputs.extend(
            [
                "derived__person_age_years",
                "derived__bureau_a1_max_dpd",
                "derived__bureau_a1_overdue_instl_max",
                "derived__bureau_a1_any_overdue_amount",
                "derived__bureau_a1_overdue_amount_max",
                "derived__bureau_a1_overdue_to_outstanding",
                "derived__bureau_a1_credit_age_span",
                "derived__prev_max_dpd",
                "derived__prev_dpd_per_application",
                "derived__a2_max_dpd",
                "derived__a2_active_closed_max_dpd_delta",
                "derived__a2_dpd30_to_dpd0_rate",
                "derived__a2_dpd90_to_dpd30_rate",
                "derived__a2_overdue_amount_per_payment",
                "derived__a2_active_overdue_amount_per_payment",
            ]
        )
    if "semantic_v2" in components:
        outputs.extend(
            [
                "derived__income_to_credamount",
                "derived__annuity_to_income",
                "derived__currdebt_to_income",
                "derived__employment_age_ratio",
                "derived__bureau_a1_outstanding_debt_max",
                "derived__bureau_a1_debt_to_credit_limit",
                "derived__bureau_a1_outstanding_to_total_amount",
                "derived__bureau_a1_recent_update_gap",
                "derived__prev_credit_mean_to_current",
                "derived__prev_credit_max_to_current",
                "derived__prev_annuity_max_to_current",
                "derived__prev_debt_max_to_credit_max",
                "derived__prev_downpmt_max_to_credit_max",
                "derived__a2_closed_overdue_amount_per_payment",
            ]
        )
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

    if (
        "a2_delinquency" in components
        or "a2_delinquency_stable" in components
        or "a2_dpd30_contract" in components
    ):
        group = "credit_bureau_a_2"
        contract_count = f"{group}__contract_count"
        payment_count = f"{group}__payment_count"
        if contract_count in df.columns and payment_count in df.columns:
            if "a2_delinquency" in components:
                exprs.append(
                    _safe_ratio_expr(
                        f"{group}__overdue_contract_count",
                        contract_count,
                        "derived__a2_overdue_contract_rate",
                    )
                )
            if "a2_dpd30_contract" not in components:
                exprs.append(_safe_ratio_expr(payment_count, contract_count, "derived__a2_payment_per_contract"))
            for threshold in ("30", "60", "90"):
                if "a2_dpd30_contract" in components and threshold != "30":
                    continue
                count_col = f"{group}__has_dpd_gt{threshold}"
                if count_col in df.columns:
                    exprs.append(
                        _safe_ratio_expr(count_col, contract_count, f"derived__a2_dpd_gt{threshold}_contract_rate")
                    )
        if "a2_delinquency" in components and payment_count in df.columns:
            count_pairs = [
                (f"{group}__dpd_gt0_payment_count", "derived__a2_dpd_gt0_payment_rate"),
                (f"{group}__dpd_gt30_payment_count", "derived__a2_dpd_gt30_payment_rate"),
                (f"{group}__dpd_gt60_payment_count", "derived__a2_dpd_gt60_payment_rate"),
                (f"{group}__dpd_gt90_payment_count", "derived__a2_dpd_gt90_payment_rate"),
                (f"{group}__overdue_amt_gt0_payment_count", "derived__a2_overdue_amt_gt0_payment_rate"),
            ]
            for count_col, alias in count_pairs:
                if count_col in df.columns:
                    exprs.append(_safe_ratio_expr(count_col, payment_count, alias))

    if "a2_payment_mix" in components:
        group = "credit_bureau_a_2"
        contract_count = f"{group}__contract_count"
        payment_count = f"{group}__payment_count"
        active_payment_count = f"{group}__active_payment_count"
        closed_payment_count = f"{group}__closed_payment_count"
        if payment_count in df.columns:
            for count_col, alias in [
                (active_payment_count, "derived__a2_active_payment_share"),
                (closed_payment_count, "derived__a2_closed_payment_share"),
            ]:
                if count_col in df.columns:
                    exprs.append(_safe_ratio_expr(count_col, payment_count, alias))
        if contract_count in df.columns:
            for count_col, alias in [
                (f"{group}__has_active_payment", "derived__a2_active_contract_rate"),
                (f"{group}__has_closed_payment", "derived__a2_closed_contract_rate"),
            ]:
                if count_col in df.columns:
                    exprs.append(_safe_ratio_expr(count_col, contract_count, alias))
        rate_specs = [
            (
                f"{group}__active_dpd_gt30_payment_count",
                active_payment_count,
                "derived__a2_active_dpd_gt30_payment_rate",
            ),
            (
                f"{group}__closed_dpd_gt30_payment_count",
                closed_payment_count,
                "derived__a2_closed_dpd_gt30_payment_rate",
            ),
            (
                f"{group}__active_overdue_amt_gt0_payment_count",
                active_payment_count,
                "derived__a2_active_overdue_amt_gt0_payment_rate",
            ),
            (
                f"{group}__closed_overdue_amt_gt0_payment_count",
                closed_payment_count,
                "derived__a2_closed_overdue_amt_gt0_payment_rate",
            ),
        ]
        for count_col, denom_col, alias in rate_specs:
            if count_col in df.columns and denom_col in df.columns:
                exprs.append(_safe_ratio_expr(count_col, denom_col, alias))
        if (
            f"{group}__active_dpd_gt30_payment_count" in df.columns
            and active_payment_count in df.columns
            and f"{group}__closed_dpd_gt30_payment_count" in df.columns
            and closed_payment_count in df.columns
        ):
            active_rate = _safe_div(
                pl.col(f"{group}__active_dpd_gt30_payment_count").cast(pl.Float64, strict=False),
                pl.col(active_payment_count).cast(pl.Float64, strict=False),
            )
            closed_rate = _safe_div(
                pl.col(f"{group}__closed_dpd_gt30_payment_count").cast(pl.Float64, strict=False),
                pl.col(closed_payment_count).cast(pl.Float64, strict=False),
            )
            exprs.append(
                (active_rate.fill_null(0) - closed_rate.fill_null(0)).alias(
                    "derived__a2_active_minus_closed_dpd_gt30_rate"
                )
            )
        if (
            f"{group}__active_overdue_amt_gt0_payment_count" in df.columns
            and active_payment_count in df.columns
            and f"{group}__closed_overdue_amt_gt0_payment_count" in df.columns
            and closed_payment_count in df.columns
        ):
            active_rate = _safe_div(
                pl.col(f"{group}__active_overdue_amt_gt0_payment_count").cast(pl.Float64, strict=False),
                pl.col(active_payment_count).cast(pl.Float64, strict=False),
            )
            closed_rate = _safe_div(
                pl.col(f"{group}__closed_overdue_amt_gt0_payment_count").cast(pl.Float64, strict=False),
                pl.col(closed_payment_count).cast(pl.Float64, strict=False),
            )
            exprs.append(
                (active_rate.fill_null(0) - closed_rate.fill_null(0)).alias(
                    "derived__a2_active_minus_closed_overdue_amt_rate"
                )
            )

    if "a2_temporal" in components:
        group = "credit_bureau_a_2"
        active_span = f"{group}__active_period_span"
        active_max = f"{group}__active_period_max"
        active_min = f"{group}__active_period_min"
        active_bad_max = f"{group}__active_dpd_gt30_period_max"
        closed_span = f"{group}__closed_period_span"
        closed_max = f"{group}__closed_period_max"
        closed_min = f"{group}__closed_period_min"
        closed_bad_max = f"{group}__closed_dpd_gt30_period_max"
        if active_span in df.columns and active_max in df.columns and active_min in df.columns:
            exprs.append(_safe_ratio_expr(active_span, active_max, "derived__a2_active_period_span_ratio"))
        if closed_span in df.columns and closed_max in df.columns and closed_min in df.columns:
            exprs.append(_safe_ratio_expr(closed_span, closed_max, "derived__a2_closed_period_span_ratio"))
        if active_max in df.columns and closed_max in df.columns:
            exprs.append(
                (
                    pl.col(active_max).cast(pl.Float64, strict=False)
                    - pl.col(closed_max).cast(pl.Float64, strict=False)
                ).alias("derived__a2_active_closed_latest_period_delta")
            )
        if active_max in df.columns and active_bad_max in df.columns:
            exprs.append(
                (
                    pl.col(active_max).cast(pl.Float64, strict=False)
                    - pl.col(active_bad_max).cast(pl.Float64, strict=False)
                ).alias("derived__a2_active_bad_latest_period_gap")
            )
        if closed_max in df.columns and closed_bad_max in df.columns:
            exprs.append(
                (
                    pl.col(closed_max).cast(pl.Float64, strict=False)
                    - pl.col(closed_bad_max).cast(pl.Float64, strict=False)
                ).alias("derived__a2_closed_bad_latest_period_gap")
            )

    if "semantic_v2" in components or "semantic_v2_stable" in components:
        semantic_full = "semantic_v2" in components

        def numeric_col(name: str) -> pl.Expr:
            return pl.col(name).cast(pl.Float64, strict=False)

        def add_ratio(numerator: str, denominator: str, alias: str) -> None:
            if numerator in df.columns and denominator in df.columns:
                exprs.append(_safe_ratio_expr(numerator, denominator, alias))

        def add_horizontal_max(cols: list[str], alias: str) -> None:
            available = [col for col in cols if col in df.columns]
            if available:
                exprs.append(pl.max_horizontal([numeric_col(col) for col in available]).alias(alias))

        def add_horizontal_min(cols: list[str], alias: str) -> None:
            available = [col for col in cols if col in df.columns]
            if available:
                exprs.append(pl.min_horizontal([numeric_col(col) for col in available]).alias(alias))

        if semantic_full:
            add_ratio("person_1__mainoccupationinc_384A__max", "credamount_770A", "derived__income_to_credamount")
            add_ratio("annuity_780A", "person_1__mainoccupationinc_384A__max", "derived__annuity_to_income")
            add_ratio("currdebt_22A", "person_1__mainoccupationinc_384A__max", "derived__currdebt_to_income")
        if "person_1__birth_259D__first" in df.columns:
            age_days = numeric_col("person_1__birth_259D__first")
            exprs.append((age_days / 365.25).alias("derived__person_age_years"))
            if semantic_full and "person_1__empl_employedtotal_800L__first" in df.columns:
                exprs.append(
                    _safe_div(numeric_col("person_1__empl_employedtotal_800L__first"), age_days).alias(
                        "derived__employment_age_ratio"
                    )
                )

        bureau_dpd_cols = [
            "credit_bureau_a_1__dpdmax_139P__max",
            "credit_bureau_a_1__dpdmax_757P__max",
        ]
        bureau_overdue_instl_cols = [
            "credit_bureau_a_1__numberofoverdueinstlmax_1039L__max",
            "credit_bureau_a_1__numberofoverdueinstlmax_1151L__max",
            "credit_bureau_a_1__numberofoverdueinstls_725L__max",
            "credit_bureau_a_1__numberofoverdueinstls_834L__max",
        ]
        bureau_overdue_amount_cols = [
            "credit_bureau_a_1__overdueamountmax_155A__max",
            "credit_bureau_a_1__overdueamountmax_35A__max",
            "credit_bureau_a_1__overdueamountmax2_14A__max",
            "credit_bureau_a_1__overdueamountmax2_398A__max",
            "credit_bureau_a_1__overdueamount_31A__max",
            "credit_bureau_a_1__overdueamount_659A__max",
            "credit_bureau_a_1__debtoverdue_47A__max",
        ]
        bureau_outstanding_cols = [
            "credit_bureau_a_1__outstandingamount_354A__max",
            "credit_bureau_a_1__outstandingamount_362A__max",
            "credit_bureau_a_1__totaloutstanddebtvalue_39A__max",
            "credit_bureau_a_1__totaloutstanddebtvalue_668A__max",
            "credit_bureau_a_1__debtoutstand_525A__max",
            "credit_bureau_a_1__residualamount_488A__max",
            "credit_bureau_a_1__residualamount_856A__max",
        ]
        bureau_credit_limit_cols = [
            "credit_bureau_a_1__credlmt_230A__max",
            "credit_bureau_a_1__credlmt_935A__max",
        ]
        bureau_total_amount_cols = [
            "credit_bureau_a_1__totalamount_6A__max",
            "credit_bureau_a_1__totalamount_996A__max",
            "credit_bureau_a_1__contractsum_5085717L__max",
        ]
        add_horizontal_max(bureau_dpd_cols, "derived__bureau_a1_max_dpd")
        add_horizontal_max(bureau_overdue_instl_cols, "derived__bureau_a1_overdue_instl_max")
        add_horizontal_max(bureau_overdue_amount_cols, "derived__bureau_a1_overdue_amount_max")
        if semantic_full:
            add_horizontal_max(bureau_outstanding_cols, "derived__bureau_a1_outstanding_debt_max")
        if any(col in df.columns for col in bureau_overdue_amount_cols):
            overdue_max = pl.max_horizontal([numeric_col(col) for col in bureau_overdue_amount_cols if col in df.columns])
            exprs.append((overdue_max.fill_null(0) > 0).cast(pl.Int8).alias("derived__bureau_a1_any_overdue_amount"))
            if any(col in df.columns for col in bureau_outstanding_cols):
                outstanding_max = pl.max_horizontal([numeric_col(col) for col in bureau_outstanding_cols if col in df.columns])
                exprs.append(
                    _safe_div(overdue_max, outstanding_max).alias("derived__bureau_a1_overdue_to_outstanding")
                )
        if any(col in df.columns for col in bureau_outstanding_cols) and any(
            col in df.columns for col in bureau_credit_limit_cols
        ):
            outstanding_max = pl.max_horizontal([numeric_col(col) for col in bureau_outstanding_cols if col in df.columns])
            limit_max = pl.max_horizontal([numeric_col(col) for col in bureau_credit_limit_cols if col in df.columns])
            if semantic_full:
                exprs.append(_safe_div(outstanding_max, limit_max).alias("derived__bureau_a1_debt_to_credit_limit"))
        if semantic_full and any(col in df.columns for col in bureau_outstanding_cols) and any(
            col in df.columns for col in bureau_total_amount_cols
        ):
            outstanding_max = pl.max_horizontal([numeric_col(col) for col in bureau_outstanding_cols if col in df.columns])
            total_max = pl.max_horizontal([numeric_col(col) for col in bureau_total_amount_cols if col in df.columns])
            exprs.append(
                _safe_div(outstanding_max, total_max).alias("derived__bureau_a1_outstanding_to_total_amount")
            )
        if semantic_full:
            add_horizontal_min(
                [
                    "credit_bureau_a_1__lastupdate_388D__min",
                    "credit_bureau_a_1__lastupdate_1112D__min",
                ],
                "derived__bureau_a1_recent_update_gap",
            )
        for max_col, min_col in [
            ("credit_bureau_a_1__dateofcredstart_181D__max", "credit_bureau_a_1__dateofcredstart_181D__min"),
            ("credit_bureau_a_1__dateofcredstart_739D__max", "credit_bureau_a_1__dateofcredstart_739D__min"),
        ]:
            if max_col in df.columns and min_col in df.columns:
                exprs.append((numeric_col(max_col) - numeric_col(min_col)).alias("derived__bureau_a1_credit_age_span"))
                break

        if semantic_full:
            add_ratio("applprev_1__credamount_590A__mean", "credamount_770A", "derived__prev_credit_mean_to_current")
            add_ratio("applprev_1__credamount_590A__max", "credamount_770A", "derived__prev_credit_max_to_current")
            add_ratio("applprev_1__annuity_853A__max", "annuity_780A", "derived__prev_annuity_max_to_current")
            add_ratio(
                "applprev_1__currdebt_94A__max",
                "applprev_1__credamount_590A__max",
                "derived__prev_debt_max_to_credit_max",
            )
            add_ratio(
                "applprev_1__downpmt_134A__max",
                "applprev_1__credamount_590A__max",
                "derived__prev_downpmt_max_to_credit_max",
            )
        add_horizontal_max(
            [
                "applprev_1__actualdpd_943P__max",
                "applprev_1__maxdpdtolerance_577P__max",
            ],
            "derived__prev_max_dpd",
        )
        if "applprev_1__maxdpdtolerance_577P__max" in df.columns and "applprev_1__row_count" in df.columns:
            exprs.append(
                _safe_div(numeric_col("applprev_1__maxdpdtolerance_577P__max"), numeric_col("applprev_1__row_count")).alias(
                    "derived__prev_dpd_per_application"
                )
            )

        add_horizontal_max(
            [
                "credit_bureau_a_2__pmts_dpd_1073P__max",
                "credit_bureau_a_2__pmts_dpd_303P__max",
            ],
            "derived__a2_max_dpd",
        )
        if "credit_bureau_a_2__pmts_dpd_1073P__max" in df.columns and "credit_bureau_a_2__pmts_dpd_303P__max" in df.columns:
            exprs.append(
                (
                    numeric_col("credit_bureau_a_2__pmts_dpd_1073P__max")
                    - numeric_col("credit_bureau_a_2__pmts_dpd_303P__max")
                ).alias("derived__a2_active_closed_max_dpd_delta")
            )
        add_ratio("credit_bureau_a_2__dpd_gt30_payment_count", "credit_bureau_a_2__dpd_gt0_payment_count", "derived__a2_dpd30_to_dpd0_rate")
        add_ratio("credit_bureau_a_2__dpd_gt90_payment_count", "credit_bureau_a_2__dpd_gt30_payment_count", "derived__a2_dpd90_to_dpd30_rate")
        add_ratio("credit_bureau_a_2__pmts_overdue_1140A__sum", "credit_bureau_a_2__payment_count", "derived__a2_active_overdue_amount_per_payment")
        if semantic_full:
            add_ratio(
                "credit_bureau_a_2__pmts_overdue_1152A__sum",
                "credit_bureau_a_2__payment_count",
                "derived__a2_closed_overdue_amount_per_payment",
            )
        if (
            "credit_bureau_a_2__pmts_overdue_1140A__sum" in df.columns
            and "credit_bureau_a_2__pmts_overdue_1152A__sum" in df.columns
            and "credit_bureau_a_2__payment_count" in df.columns
        ):
            exprs.append(
                _safe_div(
                    numeric_col("credit_bureau_a_2__pmts_overdue_1140A__sum")
                    + numeric_col("credit_bureau_a_2__pmts_overdue_1152A__sum"),
                    numeric_col("credit_bureau_a_2__payment_count"),
                ).alias("derived__a2_overdue_amount_per_payment")
            )

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
    decision_dates = lf.select(
        [
            "case_id",
            pl.col("date_decision").cast(pl.Float64, strict=False).alias("__decision_date"),
        ]
    )
    for group in preset.depth0_groups:
        lf = lf.join(load_depth0(data_dir, split, group, case_ids), on="case_id", how="left")
    for group in preset.aggregate_groups:
        lf = lf.join(
            aggregate_group(data_dir, split, group, case_ids, feature_set, decision_dates),
            on="case_id",
            how="left",
        )
    if output_columns is not None:
        wanted = set(required_raw_columns_for_outputs(output_columns, feature_set))
        wanted.update(["case_id", "target", "WEEK_NUM"])
        schema = lf.collect_schema()
        lf = lf.select([col for col in schema.names() if col in wanted])
    df = lf.collect(engine="streaming")
    if preset.lite_large_groups:
        case_ids_df = df.select("case_id")
        temp_root = (cache_path.parent if cache_path is not None else Path("outputs/features")) / "_tmp_lite"
        components = feature_set_components(feature_set)
        for group in preset.lite_large_groups:
            if "a2_twostage" in components and group in TWOSTAGE_LARGE_GROUPS:
                lite = aggregate_large_group_twostage_eager(
                    data_dir,
                    split,
                    group,
                    temp_root / f"{split}_{group}_twostage",
                    case_ids=case_ids_df,
                )
            else:
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
