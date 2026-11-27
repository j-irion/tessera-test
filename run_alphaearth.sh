#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -e data/runs/alphaearth_unet_01 ]]; then
    echo "Run directory already exists; refusing to overwrite. See ALPHAEARTH.md for resume instructions." >&2
    exit 1
fi
bash prepare_all_alphaearth.sh
.venv-v11/bin/python -u train_hybiomass.py \
    --input alphaearth --embedding-data data/alphaearth_prepared --data data/prepared \
    --reference-run data/runs/tessera_v11_unet_02 \
    --run data/runs/alphaearth_unet_01 --batch-size 4 --workers 6 --epochs 80
