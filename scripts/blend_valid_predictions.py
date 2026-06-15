from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.metric import gini_score, stability_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a two-model validation blend.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output", type=Path, default=Path("outputs/blend_search.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)
    df = left[["case_id", "WEEK_NUM", "target", "prediction"]].merge(
        right[["case_id", "prediction"]],
        on="case_id",
        suffixes=(f"_{args.left_name}", f"_{args.right_name}"),
    )
    left_col = f"prediction_{args.left_name}"
    right_col = f"prediction_{args.right_name}"
    rows = []
    for weight in np.linspace(0, 1, 21):
        pred = weight * df[left_col] + (1.0 - weight) * df[right_col]
        metric = stability_metric(df["target"].to_numpy(), pred.to_numpy(), df["WEEK_NUM"].to_numpy())
        rows.append(
            {
                f"w_{args.left_name}": weight,
                f"w_{args.right_name}": 1.0 - weight,
                "auc": roc_auc_score(df["target"], pred),
                "gini": gini_score(df["target"], pred),
                **metric,
            }
        )
    result = pd.DataFrame(rows).sort_values("stability", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(10).to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
