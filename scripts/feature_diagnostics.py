from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cached feature table quality.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pl.read_parquet(args.features)
    rows = []
    for col, dtype in zip(df.columns, df.dtypes):
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(dtype),
                "missing_rate": float(series.null_count() / len(df)),
                "n_unique": int(series.n_unique()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["missing_rate", "n_unique"], ascending=[False, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.head(30).to_string(index=False))
    print(f"columns={len(out)} rows={len(df)} saved={args.output}")


if __name__ == "__main__":
    main()
