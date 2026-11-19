"""Train the TESSERA BioMassters-style U-Net on prepared HyBiomass patches."""

import argparse
import hashlib
import json
import os
import random
import tempfile
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
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(128, 256)
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
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            target = data["target"]
            values = target[np.isfinite(target)].astype(np.float64)
            count += len(values)
            total += values.sum()
            squares += np.dot(values, values)
    if not count:
        raise ValueError("No valid training labels")
    mean = total / count
    std = float(np.sqrt(max(0.0, squares / count - mean ** 2)))
    if std == 0:
        raise ValueError("Training labels have zero variance")
    return float(mean), std


class PreparedDataset(Dataset):
    def __init__(self, paths, mean, std):
        self.paths, self.mean, self.std = paths, mean, std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            target = data["target"].astype(np.float32)
            valid = data["valid"].astype(bool)
        if embeddings.shape != (128, 128, 128) or target.shape != (128, 128):
            raise ValueError(f"Unexpected patch shape: {self.paths[index]}")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"Non-finite embedding: {self.paths[index]}")
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


def evaluate(model, loader, device, mean, std):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no training was started")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = split_paths(args.data.glob("*/*/*.npz"), args.seed)
    for name, paths in splits.items():
        print(f"{name}: {len(paths)} patches, {len({p.parent.name for p in paths})} source tiles", flush=True)
        if not paths:
            raise ValueError(f"{name} split is empty; prepare more source tiles")
    mean, std = target_stats(splits["train"])
    device = torch.device(args.device)
    loaders = {name: DataLoader(PreparedDataset(paths, mean, std), batch_size=args.batch_size,
                                shuffle=name == "train", num_workers=args.workers,
                                pin_memory=device.type == "cuda") for name, paths in splits.items()}
    args.run.mkdir(parents=True, exist_ok=False)
    manifest = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "target_mean": mean, "target_std": std,
                "splits": {name: [str(path) for path in paths] for name, paths in splits.items()}}
    (args.run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch, (x, target) in enumerate(loaders["train"], 1):
            x, target = x.to(device), target.to(device)
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
        summary = {"epoch": epoch, "train_normalized_patch_rmse": float(np.mean(losses)), "validation": val}
        print(json.dumps(summary), flush=True)
        with (args.run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        if val["mean_patch_rmse_mg_ha"] < best:
            best = val["mean_patch_rmse_mg_ha"]
            with tempfile.NamedTemporaryFile(dir=args.run, prefix=".checkpoint-", delete=False) as f:
                temporary = Path(f.name)
            try:
                torch.save({"model": model.state_dict(), "epoch": epoch, "target_mean": mean,
                            "target_std": std}, temporary)
                os.replace(temporary, args.run / "best.pt")
            finally:
                temporary.unlink(missing_ok=True)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location=device, weights_only=True)["model"])
    result = evaluate(model, loaders["test"], device, mean, std)
    (args.run / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Test:", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
