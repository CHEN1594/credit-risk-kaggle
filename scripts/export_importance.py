from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LightGBM feature importance from a saved run.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/lgbm_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.run_dir / "model.joblib")
    model = bundle["model"]
    features = bundle["features"]
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    out_path = args.run_dir / "feature_importance.csv"
    importance.to_csv(out_path, index=False)
    print(importance.head(30).to_string(index=False))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
