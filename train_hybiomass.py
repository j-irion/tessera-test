"""Train the TESSERA BioMassters-style U-Net on prepared HyBiomass patches."""

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """Two-stage U-Net described in TESSERA supplement A.11."""
    def __init__(self, in_channels=128):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 256)
        self.down2 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder2 = DoubleConv(1024, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder1 = DoubleConv(512, 256)
        self.output = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        first = self.down1(x)
        second = self.down2(self.pool(first))
        middle = self.bottleneck(self.pool(second))
        x = self.decoder2(torch.cat((self.up2(middle), second), dim=1))
        x = self.decoder1(torch.cat((self.up1(x), first), dim=1))
        return self.output(x)


def split_for_tile(tile, seed=42):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{tile}".encode()).digest()[:8], "big") / 2**64
    return "train" if value < 0.7 else "val" if value < 0.9 else "test"


def split_paths(paths, seed):
    groups = {key: [] for key in ("train", "val", "test")}
    for path in sorted(paths):
        groups[split_for_tile(path.parent.name, seed)].append(path)
    return groups


def target_stats(paths):
    count, total, squares = 0, 0.0, 0.0
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as data:
            target = data["target"]
            values = target[np.isfinite(target)].astype(np.float64)
            count += len(values)
            total += values.sum()
            squares += np.dot(values, values)
        if index % 2000 == 0 or index == len(paths):
            print(f"Training label statistics: {index}/{len(paths)} patches", flush=True)
    if not count:
        raise ValueError("No valid training labels")
    mean = total / count
    std = float(np.sqrt(max(0.0, squares / count - mean ** 2)))
    if std == 0:
        raise ValueError("Training labels have zero variance")
    return float(mean), std


def enmap_path(prepared_path, input_dir):
    return input_dir / prepared_path.parts[-3] / prepared_path.parent.name / prepared_path.name


def usable_alphaearth_paths(paths, input_dir):
    usable = []
    for path in paths:
        alternate = enmap_path(path, input_dir)
        if not alternate.is_file():
            raise FileNotFoundError(f"Missing AlphaEarth patch: {alternate}")
        with np.load(alternate, allow_pickle=False) as data:
            if np.isfinite(data["target"]).any():
                usable.append(path)
    return usable


def enmap_stats(paths, input_dir):
    sums = np.zeros(202, dtype=np.float64)
    squares = np.zeros(202, dtype=np.float64)
    counts = np.zeros(202, dtype=np.int64)
    for index, path in enumerate(paths, 1):
        with np.load(enmap_path(path, input_dir), allow_pickle=False) as data:
            sums += data["sums"]
            squares += data["squares"]
            counts += data["counts"]
        if index % 2000 == 0 or index == len(paths):
            print(f"EnMAP training-only statistics: {index}/{len(paths)} patches", flush=True)
    if np.any(counts == 0):
        raise ValueError("EnMAP training data has a band without finite values")
    mean = sums / counts
    std = np.sqrt(np.maximum(0, squares / counts - mean ** 2))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std == 0):
        raise ValueError("Invalid EnMAP training statistics")
    return mean.astype(np.float32), std.astype(np.float32)


class PreparedDataset(Dataset):
    def __init__(self, paths, mean, std, input_dir=None, input_mean=None, input_std=None, input_channels=128, embedding_dir=None):
        self.paths, self.mean, self.std = paths, mean, std
        self.input_dir, self.input_mean, self.input_std = input_dir, input_mean, input_std
        self.input_channels, self.embedding_dir = input_channels, embedding_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32) if self.input_dir is None else None
            target = data["target"].astype(np.float32)
            valid = data["valid"].astype(bool)
        if self.embedding_dir is not None:
            with np.load(enmap_path(self.paths[index], self.embedding_dir), allow_pickle=False) as data:
                embeddings = data["embeddings"].astype(np.float32)
                valid &= data["valid"]
        if self.input_dir is not None:
            with np.load(enmap_path(self.paths[index], self.input_dir), allow_pickle=False) as data:
                embeddings = data["image"].astype(np.float32)
            if embeddings.shape != (202, 128, 128):
                raise ValueError(f"Unexpected EnMAP input shape: {self.paths[index]}")
            embeddings = (embeddings - self.input_mean[:, None, None]) / self.input_std[:, None, None]
            np.nan_to_num(embeddings, copy=False, nan=0, posinf=0, neginf=0)
        elif embeddings.shape != (self.input_channels, 128, 128):
            raise ValueError(f"Unexpected embedding input shape: {self.paths[index]}")
        if target.shape != (128, 128) or valid.shape != target.shape or not np.isfinite(embeddings).all():
            raise ValueError(f"Unexpected or non-finite patch: {self.paths[index]}")
        target[~valid] = np.nan
        if not np.isfinite(target).any():
            raise ValueError(f"Patch has no usable labels: {self.paths[index]}")
        return torch.from_numpy(embeddings), torch.from_numpy((target - self.mean) / self.std).unsqueeze(0)


def patch_rmse(pred, target):
    losses = []
    for p, t in zip(pred.float(), target):
        mask = torch.isfinite(t)
        if not mask.any():
            raise ValueError("A patch has no valid labels")
        losses.append(torch.sqrt(torch.mean((p[mask] - t[mask]) ** 2) + 1e-12))
    return torch.stack(losses).mean()


def evaluate(model, loader, device, mean, std, clip_negative=False):
    model.eval()
    count = 0
    sse = target_sum = target_squares = patch_sum = 0.0
    with torch.inference_mode():
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x).float()
            for p, t in zip(pred, target):
                mask = torch.isfinite(t)
                actual = t[mask].double() * std + mean
                predicted = p[mask].double() * std + mean
                if clip_negative:
                    predicted = predicted.clamp_min(0)
                error = predicted - actual
                n = actual.numel()
                count += n
                sse += error.square().sum().item()
                target_sum += actual.sum().item()
                target_squares += actual.square().sum().item()
                patch_sum += torch.sqrt(error.square().mean()).item()
    if not count:
        raise ValueError("No valid evaluation labels")
    denominator = target_squares - target_sum ** 2 / count
    return {"rmse_mg_ha": (sse / count) ** 0.5,
            "mean_patch_rmse_mg_ha": patch_sum / len(loader.dataset),
            "r2": 1 - sse / denominator if denominator > 0 else None, "labels": count}


def save_checkpoint(path, checkpoint):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="checkpoint-", suffix=".pt", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--input", choices=("tessera", "enmap", "alphaearth"), default="tessera")
    parser.add_argument("--enmap-data", type=Path, default=Path("data/enmap"))
    parser.add_argument("--embedding-data", type=Path, help="Separate directory for AlphaEarth prepared inputs")
    parser.add_argument("--reference-run", type=Path, help="TESSERA run whose exact split and target stats are matched")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume the last completed epoch in --run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers cannot be negative")
    if args.input != "tessera" and args.reference_run is None:
        parser.error("--reference-run is required for matched baseline experiments")
    if args.input == "alphaearth" and args.embedding_data is None:
        parser.error("--embedding-data is required for AlphaEarth")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no training was started")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = split_paths(args.data.glob("*/*/*.npz"), args.seed)
    if args.input == "alphaearth":
        splits = {name: usable_alphaearth_paths(paths, args.embedding_data) for name, paths in splits.items()}
    for name, paths in splits.items():
        print(f"{name}: {len(paths)} patches, {len({p.parent.name for p in paths})} source tiles", flush=True)
        if not paths:
            raise ValueError(f"{name} split is empty; prepare more source tiles")
    split_manifest = {name: [str(path) for path in paths] for name, paths in splits.items()}
    reference = None
    if args.input != "tessera":
        reference = json.loads((args.reference_run / "manifest.json").read_text())
        if reference["splits"] != split_manifest and args.input != "alphaearth":
            raise ValueError("Prepared patches or splits do not match the reference run")
        if args.input == "alphaearth":
            for name, paths in splits.items():
                reference_paths = {Path(p) for p in reference["splits"][name]}
                if any(path not in reference_paths for path in paths):
                    raise ValueError(f"AlphaEarth {name} patches differ from the reference run")
        for key in ("seed", "batch_size"):
            if reference["config"][key] != getattr(args, key):
                raise ValueError(f"{key} differs from the reference run")
        if args.input == "enmap":
            missing = [str(enmap_path(path, args.enmap_data)) for paths in splits.values() for path in paths
                       if not enmap_path(path, args.enmap_data).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} EnMAP patches, e.g. {missing[0]}")
    if args.resume:
        manifest = json.loads((args.run / "manifest.json").read_text())
        if manifest["splits"] != split_manifest:
            raise ValueError("Prepared dataset or split differs from the original run")
        for key in ("data", "seed", "batch_size", "device", "input", "enmap_data", "embedding_data", "reference_run"):
            original = manifest["config"].get(key, "tessera" if key == "input" else None)
            current = str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)
            if original != current and not (key == "enmap_data" and args.input == "tessera"):
                raise ValueError(f"Cannot change {key} when resuming")
        mean, std = manifest["target_mean"], manifest["target_std"]
    else:
        if args.run.exists():
            raise FileExistsError(f"Refusing to overwrite {args.run}; use --resume if appropriate")
        if reference is not None:
            mean, std = reference["target_mean"], reference["target_std"]
        else:
            print("Calculating training-only label statistics...", flush=True)
            mean, std = target_stats(splits["train"])
    input_mean = input_std = None
    if args.input == "enmap":
        if args.resume:
            input_mean, input_std = np.array(manifest["input_mean"], np.float32), np.array(manifest["input_std"], np.float32)
        else:
            input_mean, input_std = enmap_stats(splits["train"], args.enmap_data)
    device = torch.device(args.device)
    input_dir = args.enmap_data if args.input == "enmap" else None
    channels = 202 if args.input == "enmap" else 64 if args.input == "alphaearth" else 128
    loaders = {name: DataLoader(PreparedDataset(paths, mean, std, input_dir, input_mean, input_std, channels, args.embedding_data), batch_size=args.batch_size,
                                shuffle=name == "train", num_workers=args.workers,
                                pin_memory=device.type == "cuda") for name, paths in splits.items()}
    if not args.resume:
        args.run.mkdir(parents=True, exist_ok=False)
        manifest = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "target_mean": mean, "target_std": std, "splits": split_manifest}
        if input_mean is not None:
            manifest["input_mean"], manifest["input_std"] = input_mean.tolist(), input_std.tolist()
        (args.run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = UNet(in_channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best, first_epoch = float("inf"), 1
    if args.resume:
        checkpoint = torch.load(args.run / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best, first_epoch = checkpoint["best"], checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        print(f"Resuming at epoch {first_epoch}", flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        losses = []
        for batch, (x, target) in enumerate(loaders["train"], 1):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x)
                loss = patch_rmse(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if batch % 20 == 0 or batch == len(loaders["train"]):
                print(f"epoch {epoch}/{args.epochs}: train {batch}/{len(loaders['train'])}", flush=True)
        val = evaluate(model, loaders["val"], device, mean, std)
        summary = {"epoch": epoch, "seconds": round(time.monotonic() - epoch_started, 1),
                   "train_normalized_patch_rmse": float(np.mean(losses)), "validation": val}
        print(json.dumps(summary), flush=True)
        with (args.run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        if val["mean_patch_rmse_mg_ha"] < best:
            best = val["mean_patch_rmse_mg_ha"]
            save_checkpoint(args.run / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                                     "target_mean": mean, "target_std": std})
        save_checkpoint(args.run / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                                 "epoch": epoch, "best": best, "torch_rng": torch.get_rng_state(),
                                                 "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None})
    model.load_state_dict(torch.load(args.run / "best.pt", map_location=device, weights_only=True)["model"])
    result = evaluate(model, loaders["test"], device, mean, std, clip_negative=True)
    (args.run / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Test:", json.dumps(result), flush=True)
    if device.type == "cuda":
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
