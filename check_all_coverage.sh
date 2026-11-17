#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for region in africa australasia europe north_america north_asia south_america south_asia; do
    output="data/coverage/${region}_gedi_coverage.csv"
    if [[ -f "$output" ]]; then
        echo "$region: existing result found; skipping"
        continue
    fi
    echo "$region: starting"
    .venv/bin/python -u check_coverage.py "$region"
done

echo "All region checks completed."
