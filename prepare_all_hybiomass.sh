#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for region in africa australasia europe north_america north_asia south_america south_asia; do
    echo "Preparing $region"
    .venv-v11/bin/python -u prepare_hybiomass.py "$region"
done
echo "All HyBiomass regions prepared."
