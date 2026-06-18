from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.memory import MemoryLimitExceeded, check_memory
from src.preprocess import PreprocessState, apply_preprocessor


class TabularDNN(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = n_features
        for width in hidden:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch tabular DNN on an existing filtered feature parquet.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-rss-gb", type=float, default=30.0)
    parser.add_argument("--min-available-gb", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_memory("start dnn train", args.max_rss_gb, args.min_available_gb)

    metadata = json.loads((args.run_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    train_path = Path(metadata["train_filtered_path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    manifest_path = args.artifact_dir / "v5_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocess_payload = json.loads((args.artifact_dir / manifest["files"]["preprocess"]).read_text(encoding="utf-8"))
    state = PreprocessState(
        feature_cols=preprocess_payload["feature_cols"],
        category_maps=preprocess_payload["category_maps"],
        fill_values=preprocess_payload["fill_values"],
        dropped_columns=preprocess_payload["dropped_columns"],
        missing_indicator_cols=preprocess_payload.get("missing_indicator_cols", {}),
    )

    train_pdf = pd.read_parquet(train_path)
    check_memory("after read train parquet", args.max_rss_gb, args.min_available_gb)
    y = train_pdf["target"].astype("float32").to_numpy()
    X = apply_preprocessor(train_pdf, state, use_float16=False)
    del train_pdf
    gc.collect()
    check_memory("after preprocess", args.max_rss_gb, args.min_available_gb)

    X_np = X.to_numpy(dtype=np.float32, copy=True)
    del X
    gc.collect()
    check_memory("after numpy matrix", args.max_rss_gb, args.min_available_gb)

    mean = np.nanmean(X_np, axis=0).astype(np.float32)
    std = np.nanstd(X_np, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    X_np = (X_np - mean) / std
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    check_memory("after normalization", args.max_rss_gb, args.min_available_gb)

    dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    hidden = tuple(int(part) for part in args.hidden.split(",") if part.strip())
    model = TabularDNN(X_np.shape[1], hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch = int(len(yb))
            total_loss += float(loss.detach().cpu()) * batch
            seen += batch
        print(json.dumps({"epoch": epoch + 1, "loss": total_loss / max(1, seen)}))
        check_memory(f"after dnn epoch {epoch + 1}", args.max_rss_gb, args.min_available_gb)

    payload = {
        "state_dict": model.cpu().state_dict(),
        "n_features": int(X_np.shape[1]),
        "hidden": hidden,
        "dropout": float(args.dropout),
        "mean": mean,
        "std": std,
        "feature_cols": state.feature_cols,
    }
    torch.save(payload, args.artifact_dir / "dnn_model.pt")

    manifest["version"] = "v15"
    manifest["model_kind"] = "lgb_cat_dnn_ensemble_metric_hack"
    manifest["files"] = dict(manifest["files"])
    manifest["files"]["catboost_model"] = "catboost_model.joblib"
    manifest["files"]["dnn_model"] = "dnn_model.pt"
    manifest["dnn"] = {
        "model": "dnn_model.pt",
        "framework": "pytorch",
        "hidden": list(hidden),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "note": "Tabular DNN trained on the 661 Cat/DNN feature branch.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.artifact_dir / "v15_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"artifact_dir": str(args.artifact_dir), "dnn_model": "dnn_model.pt"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MemoryLimitExceeded as exc:
        print(f"[memory-guard] {exc}")
        raise SystemExit(2)
