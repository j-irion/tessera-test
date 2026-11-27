"""Evaluate saved matched models by GEDI reference-biomass bin."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_hybiomass import PreparedDataset, UNet


class BinDataset(PreparedDataset):
    def __getitem__(self, index):
        inputs, _ = super().__getitem__(index)
        with np.load(self.paths[index], allow_pickle=False) as data:
            target = data["target"].astype(np.float32)
            valid = data["valid"].astype(bool)
        target[~valid] = np.nan
        return inputs, torch.from_numpy(target).unsqueeze(0), self.paths[index].parts[-3]


def bin_sums(prediction, target):
    mask = torch.isfinite(target)
    actual = target[mask].double()
    error = prediction[mask].double() - actual
    boundaries = torch.tensor([100., 200., 300., 400.], device=actual.device, dtype=torch.float64)
    bins = torch.bucketize(actual, boundaries, right=True)
    return (torch.bincount(bins, minlength=5),
            torch.bincount(bins, weights=error.square(), minlength=5))


def regional_sums(prediction, target, region_names):
    if len(region_names) != len(target):
        raise ValueError("Region count does not match batch size")
    results = {}
    for region in set(region_names):
        selected = torch.tensor([name == region for name in region_names], device=target.device)
        results[region] = bin_sums(prediction[selected], target[selected])
    return results


def evaluate(run, mode, workers=6):
    manifest = json.loads((run / "manifest.json").read_text())
    paths = [Path(path) for path in manifest["splits"]["test"]]
    if mode == "enmap":
        input_dir = Path(manifest["config"]["enmap_data"])
        input_mean = np.asarray(manifest["input_mean"], dtype=np.float32)
        input_std = np.asarray(manifest["input_std"], dtype=np.float32)
    else:
        input_dir = input_mean = input_std = None
    dataset = BinDataset(paths, manifest["target_mean"], manifest["target_std"], input_dir, input_mean, input_std)
    loader = DataLoader(dataset, batch_size=4, num_workers=workers, pin_memory=True)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model = UNet(in_channels=202 if mode == "enmap" else 128).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    counts = torch.zeros(5, device="cuda", dtype=torch.int64)
    errors = torch.zeros(5, device="cuda", dtype=torch.float64)
    regions = {}
    with torch.inference_mode():
        for i, (inputs, target, region_names) in enumerate(loader, 1):
            inputs, target = inputs.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(inputs).float()
            prediction = (prediction.double() * manifest["target_std"] + manifest["target_mean"]).clamp_min(0)
            for region, (n, sse) in regional_sums(prediction, target, region_names).items():
                counts += n
                errors += sse
                if region not in regions:
                    regions[region] = [torch.zeros_like(counts), torch.zeros_like(errors)]
                regions[region][0] += n
                regions[region][1] += sse
            if i % 100 == 0 or i == len(loader):
                print(f"{mode}: evaluated {i}/{len(loader)} batches", flush=True)
    counts, errors = counts.cpu().numpy(), errors.cpu().numpy()
    labels = ("below_100", "100_to_below_200", "200_to_below_300", "300_to_below_400", "400_plus")
    result = {label: {"count": int(n), "rmse_mg_ha": float(np.sqrt(s / n)) if n else None}
              for label, n, s in zip(labels, counts, errors)}
    overall_rmse = float(np.sqrt(errors.sum() / counts.sum()))
    reference = json.loads((run / "test_metrics.json").read_text())
    if counts.sum() != reference["labels"] or abs(overall_rmse - reference["rmse_mg_ha"]) > 0.001:
        raise ValueError(f"{mode} results do not match saved overall test metrics")
    regional = {}
    for region, (n, sse) in sorted(regions.items()):
        regional[region] = {label: {"count": int(count), "rmse_mg_ha": float(np.sqrt(error / count)) if count else None}
                            for label, count, error in zip(labels, n.cpu().numpy(), sse.cpu().numpy())}
    return {"checkpoint_epoch": checkpoint["epoch"], "patches": len(paths),
            "overall_rmse_mg_ha": overall_rmse, "bins": result, "regions": regional}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/runs/per_region_bin_rmse.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    base = Path("data/runs")
    runs = {"tessera": base / "tessera_v11_unet_02", "enmap": base / "enmap_unet_01"}
    manifests = [json.loads((run / "manifest.json").read_text()) for run in runs.values()]
    if manifests[0]["splits"]["test"] != manifests[1]["splits"]["test"]:
        raise ValueError("Test splits do not match")
    results = {name: evaluate(run, name) for name, run in runs.items()}
    if [b["count"] for b in results["tessera"]["bins"].values()] != [b["count"] for b in results["enmap"]["bins"].values()]:
        raise ValueError("Test label counts differ across models")
    for region, bins in results["tessera"]["regions"].items():
        if region not in results["enmap"]["regions"] or any(
            result["count"] != results["enmap"]["regions"][region][label]["count"]
            for label, result in bins.items()
        ):
            raise ValueError(f"Test label counts differ in region {region}")
    if results["tessera"]["regions"].keys() != results["enmap"]["regions"].keys():
        raise ValueError("Test regions differ across models")
    with args.output.open("x") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
