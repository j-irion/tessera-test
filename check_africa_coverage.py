"""Compare HyBiomass Africa GEDI pixel centers to Tessera tile inventories."""

import csv
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.io import MemoryFile

DATA = Path("data")
YEARS = range(2019, 2025)


def tile_keys(lon, lat):
    return list(zip((np.floor(lon * 10).astype(int) * 10 + 5).tolist(),
                    (np.floor(lat * 10).astype(int) * 10 + 5).tolist()))


def main():
    dates = pd.read_csv(DATA / "hybiomass_tiles_datetime_ids.csv", dtype=str)
    dates = dates[dates.region == "Africa"].set_index("tile_index").datetime_id
    inventories = {}
    for variant in ("beta1", "beta2"):
        frame = pd.read_parquet(DATA / f"coverage/v2-{variant}-manifest.parquet")
        inventories[variant] = {
            year: set(zip(group.lon_i, group.lat_i))
            for year, group in frame.groupby("year")
        }

    output = DATA / "coverage/africa_gedi_coverage.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with tarfile.open(DATA / "hybiomass_africa.tar.gz", "r|gz") as archive, output.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patch", "enmap_year", "gedi_pixels", "variant", "year", "covered_pixels"])
        count = 0
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".tif"):
                continue
            tile = Path(member.name).parent.name
            enmap_year = int(dates.loc[tile][:4])
            with MemoryFile(archive.extractfile(member).read()) as memory, memory.open() as image:
                biomass = image.read(image.count - 1, masked=True)
                rows, cols = np.where(np.isfinite(biomass.filled(np.nan)))
                x, y = image.transform * (cols + 0.5, rows + 0.5)
                lon, lat = Transformer.from_crs(image.crs, "EPSG:4326", always_xy=True).transform(x, y)
                keys = tile_keys(lon, lat)
            for variant, inventory in inventories.items():
                for year in YEARS:
                    available = inventory.get(year, set())
                    writer.writerow([member.name, enmap_year, len(keys), variant, year,
                                     sum(key in available for key in keys)])
            count += 1
            if count % 50 == 0:
                print(f"Checked {count} patches", flush=True)
        print(f"Checked {count} patches; saved {output}", flush=True)


if __name__ == "__main__":
    main()
