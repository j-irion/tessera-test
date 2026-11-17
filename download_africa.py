"""Download HyBiomass Africa using DLR credentials and aria2."""

from copy import copy
from getpass import getpass
from http.cookiejar import MozillaCookieJar
from pathlib import Path
import os
import subprocess
import tempfile
import time

import requests
from dotenv import dotenv_values
from hyperproc.archive.dlr import sign_in

URL = "https://download.geoservice.dlr.de/ENMAP/files/HYBIOMASS/hybiomass_africa.tar.gz"
DEST = Path("data/hybiomass_africa.tar.gz")


def credentials():
    values = dotenv_values(".env")
    username = values.get("DLR_USERNAME") or input("DLR username: ")
    password = values.get("DLR_PASSWORD") or getpass("DLR password (not displayed): ")
    return username, password


def download(session, dest, url=URL):
    partial = dest.with_suffix(dest.suffix + ".part")
    control = Path(str(partial) + ".aria2")
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    if partial.exists() != control.exists():
        raise FileExistsError(f"Incomplete download without matching aria2 control file: {partial}")

    # Cookies contain access credentials, so the temporary file must be private.
    fd, cookie_path = tempfile.mkstemp(prefix="dlr-cookies-", suffix=".txt")
    os.close(fd)
    try:
        jar = MozillaCookieJar(cookie_path)
        for cookie in session.cookies:
            cookie = copy(cookie)
            if cookie.expires is None:
                cookie.expires = int(time.time()) + 3600
                cookie.discard = False
            jar.set_cookie(cookie)
        jar.save(ignore_discard=True)
        print(f"Downloading {dest.name} (up to 4 connections)...", flush=True)
        subprocess.run([
            "aria2c", "--no-conf=true", "--continue=true", "--max-connection-per-server=4",
            "--split=4", "--min-split-size=1M", "--max-concurrent-downloads=1",
            "--auto-file-renaming=false", "--allow-overwrite=false", "--summary-interval=5",
            f"--load-cookies={cookie_path}", f"--user-agent={session.headers['User-Agent']}",
            f"--dir={dest.parent.resolve()}",
            f"--out={partial.name}", url,
        ], check=True)
    finally:
        os.unlink(cookie_path)
    with partial.open("rb") as f:
        if f.read(2) != b"\x1f\x8b":
            raise ValueError(f"Not a gzip archive; retained {partial}")
    dest.hardlink_to(partial)
    partial.unlink()
    print(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)", flush=True)


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    username, password = credentials()
    with requests.Session() as session:
        session.trust_env = False
        print("Signing in to DLR...", flush=True)
        sign_in(session, URL.rsplit("/", 1)[0] + "/", (username, password), sensor="ENMAP")
        del password
        download(session, DEST)


if __name__ == "__main__":
    main()
