"""Download the six HyBiomass regions other than Africa."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from hyperproc.archive.dlr import sign_in

from download_africa import URL, credentials, download

BASE_URL = URL.rsplit("/", 1)[0] + "/"
REGIONS = ("australasia", "europe", "north_america", "north_asia", "south_america", "south_asia")
WORKERS = 3


def download_one(dest, cookies):
    with requests.Session() as session:
        session.trust_env = False
        session.cookies.update(cookies)
        download(session, dest, BASE_URL + dest.name)


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

    username, password = credentials()
    with requests.Session() as session:
        session.trust_env = False
        print("Signing in to DLR...", flush=True)
        sign_in(session, BASE_URL, (username, password), sensor="ENMAP")
        del password
        cookies = session.cookies.copy()
    print(f"Downloading {len(pending)} archives with up to {WORKERS} connections.", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(lambda dest: download_one(dest, cookies), pending))


if __name__ == "__main__":
    main()
