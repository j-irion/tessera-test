#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for region in africa australasia europe north_america north_asia south_america south_asia; do
    echo "$region: starting or resuming"
    .venv-v11/bin/python -u check_v11_exact.py "$region"
done
echo "All v1.1 exact coverage checks completed."
