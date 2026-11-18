#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for region in africa australasia europe north_america north_asia south_america south_asia; do
    if [[ -f "data/coverage/${region}_v11_sample.csv" ]]; then
        echo "$region: existing sample found; skipping"
        continue
    fi
    .venv-v11/bin/python -u sample_v11_coverage.py "$region"
done
echo "All v1.1 regional samples completed."
