"""Align published TESSERA v1.1 embeddings to HyBiomass's 30 m grid."""

import argparse
import os
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTesseraZarr
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject

from check_coverage import DATA, REGIONS


def aggregate_embeddings(embeddings, source_transform, source_crs, target_shape, target_transform, target_crs):
    """Area-average valid embeddings and require full coverage of each target cell."""
    if embeddings.ndim != 3 or embeddings.shape[-1] != 128:
        raise ValueError("Expected an H x W x 128 TESSERA embedding array")
    src = np.moveaxis(embeddings, -1, 0).astype(np.float32, copy=False)
    good = np.isfinite(src).all(axis=0)
    src[:, ~good] = np.nan
    result = np.full((128, *target_shape), np.nan, dtype=np.float32)
    common = dict(src_transform=source_transform, src_crs=source_crs,
                  dst_transform=target_transform, dst_crs=target_crs,
                  resampling=Resampling.average, num_threads=2)
    reproject(src, result, src_nodata=np.nan, dst_nodata=np.nan, **common)
    fraction = np.zeros(target_shape, dtype=np.float32)
    reproject(good.astype(np.float32), fraction, src_nodata=None, dst_nodata=0, **common)
    valid = (fraction >= 0.9999) & np.isfinite(result).all(axis=0)
    result[:, ~valid] = 0
    return result, valid


def save_patch(path, embeddings, target, valid, year, crs, transform):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fd, temporary = tempfile.mkstemp(prefix=".patch-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(out, embeddings=embeddings, target=target, valid=valid,
                                year=year, crs=str(crs), transform=tuple(transform)[:6],
                                dataset="1.1/dclimate")
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def prepare(region, output, limit=None, max_per_tile=None):
    dates = pd.read_csv(DATA / "hybiomass_tiles_datetime_ids.csv", dtype=str)
    dates = dates[dates.region == region.title()].set_index("tile_index").datetime_id
    client = GeoTesseraZarr(cache_dir=DATA / "coverage/geotessera-zarr-cache")
    start = time.monotonic()
    visited = 0
    per_tile = {}
    with tarfile.open(DATA / f"hybiomass_{region}.tar.gz", "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".tif"):
                continue
            source_path = Path(member.name)
            tile = source_path.parent.name
            per_tile[tile] = per_tile.get(tile, 0) + 1
            if max_per_tile is not None and per_tile[tile] > max_per_tile:
                continue
            if limit is not None and visited >= limit:
                break
            visited += 1
            dest = output / region / tile / (source_path.stem + ".npz")
            if dest.exists():
                print(f"[{visited}] {dest}: already exists, skipping", flush=True)
                continue
            year = int(dates.loc[tile][:4])
            with MemoryFile(archive.extractfile(member).read()) as memory, memory.open() as image:
                if image.count != 204 or image.crs is None:
                    raise ValueError(f"Unexpected raster layout: {member.name}")
                transform, crs = image.transform, image.crs
                shape = (image.height, image.width)
                target = image.read(image.count - 1).astype(np.float32)
                center_x, center_y = transform * (image.width / 2, image.height / 2)
                lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(center_x, center_y)
                # read_patch pads the footprint so each 30 m pixel has a complete 10 m source area.
                width = max(image.width * abs(transform.a), image.height * abs(transform.e))
                size = int(np.ceil(width / 10)) + 8
                if size % 2:
                    size += 1
                vectors, source_transform, source_crs = client.read_patch(
                    lon, lat, year=year, size_px=size, dst_crs=str(crs), resampling="nearest")
                embeddings, valid = aggregate_embeddings(vectors, source_transform, source_crs,
                                                          shape, transform, crs)
                target[~np.isfinite(target) | ~valid] = np.nan
                if not np.isfinite(target).any():
                    print(f"[{visited}] {member.name}: no usable labels, skipping", flush=True)
                    continue
                save_patch(dest, embeddings, target, valid, year, crs, transform)
                print(f"[{visited}] {member.name}: {np.isfinite(target).sum()} labels, "
                      f"{valid.mean():.1%} embedding coverage, {time.monotonic()-start:.0f}s elapsed", flush=True)
    print(f"Finished {region}: {visited} patches visited", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("region", choices=REGIONS)
    parser.add_argument("--output", type=Path, default=DATA / "prepared")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-per-tile", type=int)
    args = parser.parse_args()
    if (args.limit is not None and args.limit <= 0) or (args.max_per_tile is not None and args.max_per_tile <= 0):
        parser.error("limits must be positive")
    prepare(args.region, args.output, args.limit, args.max_per_tile)


if __name__ == "__main__":
    main()
