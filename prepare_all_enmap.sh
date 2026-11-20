#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
exec 9>data/.enmap-preparation.lock
if ! flock -n 9; then
    echo "Another EnMAP preparation is already running." >&2
    exit 1
fi
printf '%s\n' north_america australasia europe south_america south_asia north_asia africa |
    xargs -n 1 -P 3 bash -c 'echo "Preparing $1"; .venv-v11/bin/python -u prepare_enmap.py "$1"' _
echo "All matched EnMAP inputs prepared."
