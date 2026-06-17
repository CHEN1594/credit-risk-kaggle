from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROTECTED = {"case_id", "target", "WEEK_NUM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile filtered features for cleanup and ablation planning.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_lab/feature_health"))
    parser.add_argument("--sample-rows", type=int, default=300_000)
    parser.add_argument("--duplicate-sample-rows", type=int, default=300_000)
    return parser.parse_args()


def table_prefix(col: str) -> str:
    if "__" not in col:
        return "base_or_derived"
    return col.split("__", 1)[0]


def agg_suffix(col: str) -> str:
    if "__" not in col:
        return "raw_or_derived"
    return col.rsplit("__", 1)[-1]


def duplicate_report(df: pd.DataFrame, cols: list[str], sample_rows: int) -> pd.DataFrame:
    if sample_rows and len(df) > sample_rows:
        sample = df.sample(sample_rows, random_state=42)
    else:
        sample = df

    seen: dict[int, str] = {}
    rows = []
    for col in cols:
        values = sample[col]
        fingerprint = int(pd.util.hash_pandas_object(values, index=False).sum())
        duplicate_of = seen.get(fingerprint)
        if duplicate_of is None:
            seen[fingerprint] = col
        else:
            rows.append({"feature": col, "duplicate_of": duplicate_of, "hash": fingerprint})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    if args.sample_rows and len(df) > args.sample_rows:
        profile_df = df.sample(args.sample_rows, random_state=42)
    else:
        profile_df = df

    feature_cols = [col for col in profile_df.columns if col not in PROTECTED]
    rows = []
    for col in feature_cols:
        s = profile_df[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "table_prefix": table_prefix(col),
                "agg_suffix": agg_suffix(col),
                "missing_rate": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
                "is_constant": bool(s.nunique(dropna=True) <= 1),
            }
        )

    profile = pd.DataFrame(rows)
    duplicate = duplicate_report(df, feature_cols, args.duplicate_sample_rows)
    profile = profile.merge(duplicate[["feature", "duplicate_of"]] if len(duplicate) else pd.DataFrame(columns=["feature", "duplicate_of"]), on="feature", how="left")

    by_table = (
        profile.groupby("table_prefix", dropna=False)
        .agg(
            features=("feature", "count"),
            mean_missing=("missing_rate", "mean"),
            max_missing=("missing_rate", "max"),
            constant_features=("is_constant", "sum"),
            duplicate_features=("duplicate_of", lambda x: int(x.notna().sum())),
        )
        .reset_index()
        .sort_values(["features", "mean_missing"], ascending=[False, False])
    )
    by_agg = (
        profile.groupby("agg_suffix", dropna=False)
        .agg(
            features=("feature", "count"),
            mean_missing=("missing_rate", "mean"),
            constant_features=("is_constant", "sum"),
            duplicate_features=("duplicate_of", lambda x: int(x.notna().sum())),
        )
        .reset_index()
        .sort_values(["features", "mean_missing"], ascending=[False, False])
    )

    profile.to_csv(args.output_dir / "feature_profile.csv", index=False)
    by_table.to_csv(args.output_dir / "by_table.csv", index=False)
    by_agg.to_csv(args.output_dir / "by_agg.csv", index=False)
    duplicate.to_csv(args.output_dir / "duplicates.csv", index=False)
    summary = {
        "input": str(args.input),
        "rows_profiled": int(len(profile_df)),
        "features": int(len(feature_cols)),
        "constant_features": int(profile["is_constant"].sum()),
        "duplicate_features": int(profile["duplicate_of"].notna().sum()),
        "outputs": {
            "feature_profile": str(args.output_dir / "feature_profile.csv"),
            "by_table": str(args.output_dir / "by_table.csv"),
            "by_agg": str(args.output_dir / "by_agg.csv"),
            "duplicates": str(args.output_dir / "duplicates.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
