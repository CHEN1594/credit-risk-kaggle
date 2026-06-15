from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return normalized gini, equivalent to 2 * AUC - 1."""
    return 2.0 * roc_auc_score(y_true, y_pred) - 1.0


def stability_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    week_num: np.ndarray,
    slope_penalty: float = 88.0,
    residual_penalty: float = 0.5,
) -> dict[str, float]:
    """Compute Kaggle's weekly gini stability metric on labeled validation data."""
    frame = pd.DataFrame({"target": y_true, "pred": y_pred, "week": week_num})
    weekly = []
    for week, grp in frame.groupby("week", sort=True):
        if grp["target"].nunique() < 2:
            continue
        weekly.append((float(week), gini_score(grp["target"].to_numpy(), grp["pred"].to_numpy())))

    if len(weekly) < 2:
        return {
            "stability": float("nan"),
            "mean_gini": float("nan"),
            "slope": float("nan"),
            "falling_rate": float("nan"),
            "residual_std": float("nan"),
            "n_weeks": float(len(weekly)),
        }

    weeks = np.array([x[0] for x in weekly], dtype=float)
    ginis = np.array([x[1] for x in weekly], dtype=float)
    slope, intercept = np.polyfit(weeks, ginis, deg=1)
    residuals = ginis - (slope * weeks + intercept)
    residual_std = float(np.std(residuals))
    falling_rate = float(min(0.0, slope))
    stability = float(np.mean(ginis) + slope_penalty * falling_rate - residual_penalty * residual_std)
    return {
        "stability": stability,
        "mean_gini": float(np.mean(ginis)),
        "slope": float(slope),
        "falling_rate": falling_rate,
        "residual_std": residual_std,
        "n_weeks": float(len(weekly)),
    }
