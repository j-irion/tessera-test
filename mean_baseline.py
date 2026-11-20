"""Evaluate the training-label mean on the TESSERA run's held-out test labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(paths, prediction):
    total = squares = sse = patch_rmse = 0.0
    count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            target = data["target"]
            valid = data["valid"] & np.isfinite(target)
            values = target[valid].astype(np.float64)
        if not len(values):
            raise ValueError(f"No usable labels in {path}")
        error = values - prediction
        count += len(values)
        total += values.sum()
        squares += np.dot(values, values)
        loss = np.dot(error, error)
        sse += loss
        patch_rmse += np.sqrt(loss / len(values))
    if not count:
        raise ValueError("No test labels")
    variance = squares - total ** 2 / count
    return {"rmse_mg_ha": (sse / count) ** 0.5, "mean_patch_rmse_mg_ha": patch_rmse / len(paths),
            "r2": 1 - sse / variance if variance > 0 else None, "labels": count, "patches": len(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, default=Path("data/runs/tessera_v11_unet_02"))
    args = parser.parse_args()
    manifest = json.loads((args.reference_run / "manifest.json").read_text())
    prediction = max(0, manifest["target_mean"])
    result = {"prediction_mg_ha": prediction, **evaluate(manifest["splits"]["test"], prediction)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
