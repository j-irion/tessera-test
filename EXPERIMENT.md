# HyBiomass with TESSERA v1.1

This experiment uses the v1.1/dclimate embeddings and the U-Net architecture and patch-wise RMSE described in the TESSERA BioMassters experiment. The pretrained TESSERA encoder is not trained. The dataset and model code are separate; only local training runs are produced. See `THIRD_PARTY_NOTICES.md` for the U-Net source.

## Data preparation

The scripts expect the HyBiomass region archives and `hybiomass_tiles_datetime_ids.csv` in `data/`, and require Python 3.12, GDAL/rasterio and the pinned dependencies in `requirements-experiment.txt`. On siena09 they are installed in `.venv-v11`. `data/` is ignored by Git.

For each HyBiomass patch, `prepare_hybiomass.py` reads year-matched TESSERA data on its native grid and area-averages its 10 m embeddings directly onto the original HyBiomass 30 m UTM grid. It keeps all 128 dimensions and spatial context. A 30 m embedding is valid only if the whole contributing area is valid. Invalid inputs are set to zero; their target labels are masked, as are missing GEDI labels. Each prepared patch preserves its target CRS, transform and year. The GEDI estimates are targets, not field-measured truth.

Prepare one region, or all seven:

```bash
.venv-v11/bin/python -u prepare_hybiomass.py africa
bash prepare_all_hybiomass.sh
```

Run long jobs in tmux. The output `data/prepared/<region>/<tile>/<patch>.npz` is written atomically; existing patches are skipped. The command is restartable, but re-reads the compressed archives. It never overwrites an existing patch. Use a separate `--output` directory for any changes to preprocessing settings. A small pilot is possible with `--output data/prepared_pilot_v2 --limit 12 --max-per-tile 1`. Do not mix with `data/prepared_pilot`: that early pilot used an intermediate 10 m grid; the current method reads the native grid directly.

Preparation downloads embeddings, reports progress per patch, and requires significant network traffic and storage. The initial 12-patch pilot used about 85 MiB; the full amount depends on the data and compression. Do not assume labeled-pixel availability implies entire input-patch availability.

## Training

```bash
.venv-v11/bin/python -u train_hybiomass.py --data data/prepared --run data/runs/experiment_01
```

The run directory must not exist. Defaults: one CUDA GPU, batch size 1, BF16 autocast, Adam at 1e-4, and 80 epochs. The model uses the paper's 256/512-channel encoder and 1024-channel bottleneck; it is not a reduced U-Net. `--epochs`, `--batch-size` and `--workers` can be overridden. `--device cpu` is for testing only. Validation selects the lowest mean patch RMSE checkpoint; final test output includes pixel-weighted RMSE, mean patch RMSE and pixel-weighted R² in original Mg/ha units. Target normalization uses training labels only. Empty splits fail before training.

Splits are deterministic 70/20/10 by source tile ID, shared across regions so the same tile cannot occur in different splits. They are not the HyBiomass paper's five random patch splits and do not guarantee geographic separation between different tiles. For strict comparison, reproduce the published splits and report any spatial leakage risk; for geographic generalization, define spatial holdouts separately.

`manifest.json` records all patch paths and split assignments; `metrics.jsonl`, `best.pt` and `test_metrics.json` are saved within the new run directory. The best checkpoint can be used for inference, but the current script does not resume optimizer state after an interrupted training run.

Temporal matching uses the EnMAP scene year, not necessarily the GEDI acquisition year. Annual embeddings may include observations after the GEDI measurement. This and mask/aggregation rules must be stated when reporting results. The 12-patch one-epoch pilot is only a pipeline check, not a performance benchmark. Earlier pilot runs are kept separate from `data/prepared`.
