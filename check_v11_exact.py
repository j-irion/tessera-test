"""Check actual v1.1 scale values at every GEDI pixel in one region."""

import argparse
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from geotessera import GeoTesseraZarr
from geotessera.store import _nearest_indices, _zone_for_lon
from pyproj import Transformer
from rasterio.io import MemoryFile

from check_coverage import DATA, REGIONS

BATCH_SIZE = 2000
ROOT = DATA / "coverage/v11_exact"


def save_new(frame, path):
    partial = path.with_suffix(path.suffix + ".part")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {path} or {partial}")
    frame.to_parquet(partial, index=False)
    path.hardlink_to(partial)
    partial.unlink()


def extract_labels(region, output):
    dates = pd.read_csv(DATA / "hybiomass_tiles_datetime_ids.csv", dtype=str)
    dates = dates[dates.region == region.title()].set_index("tile_index").datetime_id
    frames = []
    with tarfile.open(DATA / f"hybiomass_{region}.tar.gz", "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".tif"):
                continue
            tile = Path(member.name).parent.name
            year = int(dates.loc[tile][:4])
            with MemoryFile(archive.extractfile(member).read()) as memory, memory.open() as image:
                biomass = image.read(image.count - 1, masked=True)
                rows, cols = np.where(np.isfinite(biomass.filled(np.nan)))
                x, y = image.transform * (cols + 0.5, rows + 0.5)
                lon, lat = Transformer.from_crs(image.crs, "EPSG:4326", always_xy=True).transform(x, y)
                frames.append(pd.DataFrame({"patch": member.name, "year": year, "lon": lon, "lat": lat}))
            if len(frames) % 50 == 0:
                print(f"{region}: extracted labels from {len(frames)} patches", flush=True)
    labels = pd.concat(frames, ignore_index=True)
    labels["zone"] = [_zone_for_lon(lon) for lon in labels.lon]
    save_new(labels, output)
    print(f"{region}: saved {len(labels):,} label locations", flush=True)


def pixel_indices(coords, values):
    indices = _nearest_indices(coords, np.asarray(values))
    inside = np.isfinite(values) & (np.abs(coords[indices] - values) <= 10)
    return indices, inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("region", choices=REGIONS)
    args = parser.parse_args()
    started = time.monotonic()
    directory = ROOT / args.region
    directory.mkdir(parents=True, exist_ok=True)
    labels_path = directory / "labels.parquet"
    if not labels_path.exists():
        extract_labels(args.region, labels_path)
    labels = pd.read_parquet(labels_path)
    gt = GeoTesseraZarr(cache_dir=DATA / "coverage/geotessera-zarr-cache")
    for (zone, year), group in labels.groupby(["zone", "year"], sort=True):
        zone, year = int(zone), int(year)
        ds = gt.open_zone(zone=zone)
        lon, lat = group.lon.to_numpy(), group.lat.to_numpy()
        east, north = Transformer.from_crs("EPSG:4326", ds.tessera.crs, always_xy=True).transform(lon, lat)
        xi, x_ok = pixel_indices(ds.x.values, east)
        yi, y_ok = pixel_indices(ds.y.values, north)
        indices = group.index.to_numpy()
        inside = x_ok & y_ok
        scales = gt._zone_arrays(zone)["scales"]
        order = np.lexsort((xi // scales.chunks[2], yi // scales.chunks[1]))
        time_index = list(gt.years).index(year)
        batch_count = (len(group) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch in range(batch_count):
            result_path = directory / f"zone{zone:02d}_{year}_{batch:05d}.parquet"
            if result_path.exists():
                continue
            take = order[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            valid = np.zeros(len(take), dtype=bool)
            eligible = np.flatnonzero(inside[take])
            if len(eligible):
                selected = take[eligible]
                values = scales.vindex[np.full(len(selected), time_index), yi[selected], xi[selected]]
                valid[eligible] = np.isfinite(values)
            save_new(pd.DataFrame({"row": indices[take], "valid": valid}), result_path)
            print(f"{args.region}: zone {zone:02d}, {year}, batch {batch + 1}/{batch_count}; "
                  f"{len(take)} pixels; {time.monotonic() - started:.0f}s elapsed", flush=True)
    parts = [pd.read_parquet(path) for path in sorted(directory.glob("zone*.parquet"))]
    results = pd.concat(parts, ignore_index=True).sort_values("row")
    if len(results) != len(labels) or not np.array_equal(results.row.to_numpy(), np.arange(len(labels))):
        raise ValueError("Checkpoint rows do not match the label locations")
    valid = int(results.valid.sum())
    print(f"{args.region}: {valid:,}/{len(labels):,} valid label pixels "
          f"({valid / len(labels):.2%}); {time.monotonic() - started:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
