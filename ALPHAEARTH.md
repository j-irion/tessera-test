# Matched AlphaEarth experiment

This run compares Google/Google DeepMind's annual **AlphaEarth Satellite Embedding V1** with the existing TESSERA and EnMAP U-Nets. It uses the same source-tile split, target normalization, patch-wise masked RMSE, U-Net architecture, batch size 4, Adam 1e-4, BF16, 80 epochs and best-validation checkpoint selection on one GPU. The only model architecture difference is the required first-layer input width: **64 channels** instead of TESSERA's 128 or EnMAP's 202.

The source is the public AlphaEarth annual COG collection: https://source.coop/tge-labs/aef/v1/annual. The official description and dequantization instructions are at https://developers.google.com/earth-engine/guides/aef_on_gcs_readme. Attribution: **The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.** Dataset license: CC BY 4.0. No Earth Engine credentials are needed for the public COG files. The index is cached in `data/alphaearth/aef_index.parquet`.

For each patch, the script selects files intersecting its WGS84 footprint in its EnMAP acquisition year. Raw signed 8-bit channels are mapped with `sign(x) * (x / 127.5)^2`; `-128` is missing in all channels. Valid 10 m vectors are area-averaged onto the exact original 30 m HyBiomass UTM grid. Following Google's downsample procedure, each resulting vector is normalized to unit Euclidean length. Incomplete or zero-norm outputs are invalid and are zero-filled for the network. GEDI labels at invalid embedding locations are masked. This is a direct area aggregation, not a prediction-grid shift or arbitrary 3x3 index grouping.

`data/alphaearth_prepared/<region>/<tile>/<patch>.npz` is written atomically and never overwritten. Empty patches are retained for split bookkeeping but excluded from training and evaluation if all labels become masked. Other patches keep the exact original split and GEDI target normalization. The program checks the source rasters and input files; missing files fail rather than silently changing the dataset.

## Full run

```bash
bash run_alphaearth.sh
```

This prepares the seven regions with up to three independent region workers, then starts training. Each patch's 64 remote band reads are parallelized (16 connections per region worker by default). Expect substantial network traffic and a long preparation; progress is printed per patch. Start this in tmux. To resume training after a completed epoch, repeat the training command in `run_alphaearth.sh` with `--resume`; preparation itself skips existing outputs. Do not change the preprocessing settings into the same output directory.

The 12-patch realThe pilot worked on the GPU reported as NVIDIA L40 with batch size 4, using approximately 1.05 GiB peak allocated memory. It is not a performance result. Full-run coverage, training scores and any missing patches must be checked afterward. Existing TESSERA/EnMAP scores use slightly different label masks when their embedding inputs are incomplete; when comparing modalities, report both the evaluated patch and label counts. The current test set has already been inspected, so it is not an untouched holdout after further comparisons.
