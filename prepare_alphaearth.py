"""Align public AlphaEarth annual embeddings to the prepared HyBiomass patches."""

import argparse
import math
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import transform as transform_geometry

from check_coverage import DATA, REGIONS
from prepare_hybiomass import aggregate_embeddings, save_patch

INDEX_PATH = DATA / "alphaearth/aef_index.parquet"
MIRROR = "https://data.source.coop/tge-labs/aef/v1/annual/"
ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
       "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": str(512 * 1024**2), "GDAL_HTTP_MAX_RETRY": "3"}


def patch_polygon(image):
    target = box(*array_bounds(image.height, image.width, image.transform))
    crs = Transformer.from_crs(image.crs, "EPSG:4326", always_xy=True)
    return transform_geometry(crs.transform, target)


def matching_files(index, footprint, year):
    candidates = index.iloc[index.sindex.query(footprint, predicate="intersects")]
    selected = candidates[(candidates.year == year) & candidates.geometry.intersects(footprint)]
    if selected.empty:
        raise FileNotFoundError(f"No AlphaEarth source intersects {footprint} in {year}")
    return selected.drop_duplicates("path")


def read_band(url, band, window):
    with rasterio.Env(**ENV), rasterio.open(url) as image:
        return image.read(band, window=window)


def read_source(row, footprint, crs, height, width, workers):
    url = MIRROR + row.path.split("/annual/", 1)[1]
    with rasterio.Env(**ENV), rasterio.open(url) as source:
        if source.count != 64 or source.nodata != -128 or source.dtypes[0] != "int8":
            raise ValueError(f"Unexpected AlphaEarth layout: {url}")
        bounds = transform_bounds("EPSG:4326", source.crs, *footprint.bounds, densify_pts=21)
        pad = 30
        bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
        inverse = ~source.transform
        points = [inverse * (x, y) for x in (bounds[0], bounds[2]) for y in (bounds[1], bounds[3])]
        xs, ys = zip(*points)
        x0, x1 = max(0, math.floor(min(xs))), min(source.width, math.ceil(max(xs)))
        y0, y1 = max(0, math.floor(min(ys))), min(source.height, math.ceil(max(ys)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"AlphaEarth source has no pixels for patch: {url}")
        window = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
        transform = source.transform * Affine.translation(x0, y0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        bands = list(pool.map(lambda band: read_band(url, band, window), range(1, 65)))
    raw = np.stack(bands)
    if (raw == -128).any(axis=0).any():
        values = raw.astype(np.float32)
        values[(raw == -128).any(axis=0)] = np.nan
    else:
        values = raw.astype(np.float32)
    values = (values / 127.5) ** 2 * np.sign(values)
    return np.moveaxis(values, 0, -1), transform, source.crs


def normalize_vectors(values, valid):
    norms = np.linalg.norm(values, axis=0)
    valid &= norms > 0
    values[:, valid] /= norms[valid]
    values[:, ~valid] = 0
    return values, valid


def prepare(region, output, limit=None, max_per_tile=None, workers=16):
    index = gpd.read_parquet(INDEX_PATH)
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
            dest = output / region / tile / f"{source_path.stem}.npz"
            if dest.exists():
                print(f"[{visited}] {dest}: exists, skipping", flush=True)
                continue
            reference = DATA / "prepared" / region / tile / f"{source_path.stem}.npz"
            with np.load(reference, allow_pickle=False) as prepared:
                year = int(prepared["year"])
                target = prepared["target"].copy()
            with MemoryFile(archive.extractfile(member).read()) as memory, memory.open() as image:
                footprint = patch_polygon(image)
                rows = matching_files(index, footprint, year)
                vectors = np.zeros((64, image.height, image.width), np.float32)
                valid = np.zeros((image.height, image.width), bool)
                for _, row in rows.iterrows():
                    embeddings, transform, crs = read_source(row, footprint, image.crs, image.height, image.width, workers)
                    values, available = aggregate_embeddings(embeddings, transform, crs,
                                                             (image.height, image.width), image.transform, image.crs)
                    use = available & ~valid
                    vectors[:, use] = values[:, use]
                    valid |= available
                    if valid.all():
                        break
                vectors, valid = normalize_vectors(vectors, valid)
                target[~valid] = np.nan
                if not np.isfinite(target).any():
                    print(f"[{visited}] {member.name}: no usable labels, skipping", flush=True)
                    continue
                save_patch(dest, vectors, target, valid, year, image.crs, image.transform,
                           dataset="alphaearth/v1/annual")
                print(f"[{visited}] {member.name}: {np.isfinite(target).sum()} labels, "
                      f"{valid.mean():.1%} coverage, {time.monotonic()-start:.0f}s elapsed", flush=True)
    print(f"Finished {region}: {visited} patches visited", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("region", choices=REGIONS)
    parser.add_argument("--output", type=Path, default=DATA / "alphaearth_prepared")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-per-tile", type=int)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if any(n is not None and n <= 0 for n in (args.limit, args.max_per_tile, args.workers)):
        parser.error("limits and workers must be positive")
    prepare(args.region, args.output, args.limit, args.max_per_tile, args.workers)


if __name__ == "__main__":
    main()
