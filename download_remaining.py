"""Download the six HyBiomass regions other than Africa."""

from getpass import getpass
from pathlib import Path

import requests
from hyperproc.archive.dlr import sign_in

from download_africa import URL, download

BASE_URL = URL.rsplit("/", 1)[0] + "/"
REGIONS = ("australasia", "europe", "north_america", "north_asia", "south_america", "south_asia")


def main():
    data = Path("data")
    data.mkdir(parents=True, exist_ok=True)
    pending = []
    for region in REGIONS:
        dest = data / f"hybiomass_{region}.tar.gz"
        partial = dest.with_suffix(dest.suffix + ".part")
        if partial.exists():
            raise FileExistsError(f"Partial download exists: {partial}; inspect it before retrying")
        if dest.exists():
            print(f"Skipping existing {dest}", flush=True)
        else:
            pending.append(dest)
    if not pending:
        print("All archives already exist; nothing to download.")
        return

    username = input("DLR username: ")
    password = getpass("DLR password (not displayed): ")
    with requests.Session() as session:
        session.trust_env = False
        print("Signing in to DLR...", flush=True)
        sign_in(session, BASE_URL, (username, password), sensor="ENMAP")
        del password
        for index, dest in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {dest.name}", flush=True)
            download(session, dest, BASE_URL + dest.name)


if __name__ == "__main__":
    main()
