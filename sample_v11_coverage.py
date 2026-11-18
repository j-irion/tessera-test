"""Estimate v1.1 coverage from a random sample of HyBiomass patches."""

import argparse
import csv
import hashlib
import random
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from geotessera import GeoTesseraZarr
from pyproj import Transformer
from rasterio.io import MemoryFile

from check_coverage import DATA, REGIONS

SAMPLE_SIZE = 100


def selected_patches(frame, seed):
    patches = sorted(frame[(frame.variant == "beta1") & (frame.year == frame.enmap_year)].patch.unique())
    return set(random.Random(seed).sample(patches, min(SAMPLE_SIZE, len(patches))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("region", choices=REGIONS)
    region = parser.parse_args().region
    source = DATA / f"coverage/{region}_gedi_coverage.csv"
    output = DATA / f"coverage/{region}_v11_sample.csv"
    partial = output.with_suffix(".csv.part")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}")
    rows = pd.read_csv(source)
    samples = selected_patches(rows, seed=42)
    metadata = rows[(rows.variant == "beta1") & (rows.year == rows.enmap_year)].set_index("patch")
    records = []
    with tarfile.open(DATA / f"hybiomass_{region}.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.name not in samples:
                continue
            with MemoryFile(archive.extractfile(member).read()) as memory, memory.open() as image:
                biomass = image.read(image.count - 1, masked=True)
                rr, cc = np.where(np.isfinite(biomass.filled(np.nan)))
                if not len(rr):
                    raise ValueError(f"No GEDI pixels in {member.name}")
                seed = int.from_bytes(hashlib.sha256(member.name.encode()).digest()[:8], "big")
                index = random.Random(seed).randrange(len(rr))
                x, y = image.transform * (cc[index] + 0.5, rr[index] + 0.5)
                lon, lat = Transformer.from_crs(image.crs, "EPSG:4326", always_xy=True).transform(x, y)
                records.append({"patch": member.name, "year": int(metadata.loc[member.name].enmap_year),
                                "gedi_pixels": len(rr), "lon": lon, "lat": lat})
            if len(records) % 10 == 0:
                print(f"{region}: selected {len(records)}/{len(samples)} patches", flush=True)
            if len(records) == len(samples):
                break
    if len(records) != len(samples):
        raise ValueError(f"Only found {len(records)} of {len(samples)} sampled patches")
    gt = GeoTesseraZarr(cache_dir=DATA / "coverage/geotessera-zarr-cache")
    for year in sorted({r["year"] for r in records}):
        group = [r for r in records if r["year"] == year]
        print(f"{region}: checking {len(group)} pixels for {year}", flush=True)
        embeddings = gt.sample_points([(r["lon"], r["lat"]) for r in group], year=year, search_px=0)
        for record, embedding in zip(group, embeddings):
            record["valid"] = bool(np.isfinite(embedding).all())
    with partial.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch", "year", "gedi_pixels", "lon", "lat", "valid"])
        writer.writeheader()
        writer.writerows(records)
    output.hardlink_to(partial)
    partial.unlink()
    valid = sum(r["valid"] for r in records)
    print(f"{region}: {valid}/{len(records)} sampled pixels valid; saved {output}", flush=True)


if __name__ == "__main__":
    main()
