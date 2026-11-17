"""Download HyBiomass Africa after prompting for DLR credentials."""

from getpass import getpass
from pathlib import Path

import requests
from hyperproc.archive.dlr import sign_in

URL = "https://download.geoservice.dlr.de/ENMAP/files/HYBIOMASS/hybiomass_africa.tar.gz"
DEST = Path("data/hybiomass_africa.tar.gz")


def download(session, dest):
    partial = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {dest} or {partial}")
    with session.get(URL, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        chunks = response.iter_content(chunk_size=1024 * 1024)
        first = next(chunks, b"")
        if not first.startswith(b"\x1f\x8b"):
            raise ValueError("Server did not return a gzip archive; check DLR access")
        with partial.open("xb") as output:
            output.write(first)
            for chunk in chunks:
                output.write(chunk)
    expected = response.headers.get("Content-Length")
    if expected is not None and partial.stat().st_size != int(expected):
        raise IOError("Incomplete download; partial file retained")
    # Hard-link creation refuses to overwrite an existing destination.
    dest.hardlink_to(partial)
    partial.unlink()
    print(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)")


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() or DEST.with_suffix(DEST.suffix + ".part").exists():
        raise FileExistsError("Archive or partial download already exists; inspect it first")
    username = input("DLR username: ")
    password = getpass("DLR password (not displayed): ")
    with requests.Session() as session:
        session.trust_env = False
        sign_in(session, URL.rsplit("/", 1)[0] + "/", (username, password), sensor="ENMAP")
        del password
        download(session, DEST)


if __name__ == "__main__":
    main()
