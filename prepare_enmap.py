"""Extract EnMAP inputs for exactly the TESSERA experiment's prepared patches."""

import argparse
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from check_coverage import DATA, REGIONS


def input_stats(values):
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0).astype(np.float64)
    return (clean.sum(axis=(1, 2)), np.einsum("chw,chw->c", clean, clean),
            finite.sum(axis=(1, 2)))


def save_input(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    sums, squares, counts = input_stats(values)
    # EnMAP imagery is usually integer-valued; convert only when exactly lossless.
    if (np.isfinite(values).all() and values.min() >= 0 and values.max() <= 65535
            and np.equal(values, np.floor(values)).all()):
        values = values.astype(np.uint16)
    fd, temporary = tempfile.mkstemp(prefix=".enmap-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(out, image=values, sums=sums, squares=squares, counts=counts)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def prepare(region, reference, output, limit=None, max_per_tile=None):
    manifest = json.loads(reference.read_text())
    selected = {Path(p) for paths in manifest["splits"].values() for p in paths if Path(p).parts[-3] == region}
    if not selected:
        raise ValueError(f"No reference patches for {region}")
    found = set()
    per_tile = {}
    start = time.monotonic()
    with tarfile.open(DATA / f"hybiomass_{region}.tar.gz", "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".tif"):
                continue
            source = Path(member.name)
            path = DATA / "prepared" / region / source.parent.name / f"{source.stem}.npz"
            if path not in selected:
                continue
            per_tile[source.parent.name] = per_tile.get(source.parent.name, 0) + 1
            if max_per_tile is not None and per_tile[source.parent.name] > max_per_tile:
                continue
            if limit is not None and len(found) >= limit:
                break
            found.add(path)
            dest = output / region / source.parent.name / f"{source.stem}.npz"
            if dest.exists():
                print(f"[{len(found)}] {dest}: already exists, skipping", flush=True)
                continue
            with MemoryFile(archive.extractfile(member).read()) as memory, memory.open() as image, \
                    np.load(path, allow_pickle=False) as matched:
                if image.count != 204 or image.height != 128 or image.width != 128 or image.crs is None:
                    raise ValueError(f"Unexpected source layout: {member.name}")
                if str(image.crs) != str(matched["crs"]) or not np.allclose(tuple(image.transform)[:6], matched["transform"]):
                    raise ValueError(f"Raster grid differs from reference: {member.name}")
                target = matched["target"]
                mask = np.isfinite(target)
                original = image.read(203)
                if not np.array_equal(original[mask], target[mask]):
                    raise ValueError(f"GEDI labels differ from reference: {member.name}")
                x = image.read(list(range(1, 203)))
                save_input(dest, x)
            print(f"[{len(found)}] {member.name}: saved EnMAP inputs, {time.monotonic()-start:.0f}s elapsed", flush=True)
    tile_counts = {}
    for path in selected:
        tile_counts[path.parent.name] = tile_counts.get(path.parent.name, 0) + 1
    expected = sum(min(n, max_per_tile) if max_per_tile is not None else n for n in tile_counts.values())
    if limit is not None:
        expected = min(expected, limit)
    if len(found) != expected:
        raise ValueError(f"Found {len(found)} of {expected} reference patches for {region}")
    print(f"Finished {region}: {len(found)} matched patches", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("region", choices=REGIONS)
    parser.add_argument("--reference", type=Path, default=DATA / "runs/tessera_v11_unet_02/manifest.json")
    parser.add_argument("--output", type=Path, default=DATA / "enmap")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-per-tile", type=int)
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or (args.max_per_tile is not None and args.max_per_tile <= 0):
        parser.error("limits must be positive")
    prepare(args.region, args.reference, args.output, args.limit, args.max_per_tile)


if __name__ == "__main__":
    main()
