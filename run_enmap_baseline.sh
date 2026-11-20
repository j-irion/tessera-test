#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -e data/runs/enmap_unet_01 ]]; then
    echo "Run directory already exists; refusing to overwrite. See ENMAP_BASELINE.md for resume instructions." >&2
    exit 1
fi
bash prepare_all_enmap.sh
.venv-v11/bin/python -u train_hybiomass.py \
    --input enmap --enmap-data data/enmap --data data/prepared \
    --reference-run data/runs/tessera_v11_unet_02 \
    --run data/runs/enmap_unet_01 --batch-size 4 --workers 6 --epochs 80
