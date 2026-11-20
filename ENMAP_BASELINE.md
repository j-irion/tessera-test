# Matched EnMAP baseline

This experiment compares the existing TESSERA v1.1 U-Net run with the **same architecture, patch splits, GEDI targets, target masks, loss and test metrics**, replacing only the 128 aggregated TESSERA input channels with 202 original EnMAP channels. The first convolution necessarily has more input channels. Both use batch size 4, Adam 1e-4, BF16 autocast, 80 epochs and best-validation checkpoint selection. One GPU is used.

The reference run is `data/runs/tessera_v11_unet_02`. Its `manifest.json` fixes all split paths and target mean/std. The training program checks that the prepared dataset and splits exactly match the reference before training; it also verifies every EnMAP input exists. The target arrays and masks are read directly from `data/prepared`, so both methods are evaluated at exactly the same labeled pixels. As the existing test set has already been inspected, do not treat it as an untouched final holdout after further tuning.

## Prepare inputs

```bash
bash prepare_all_enmap.sh
```

This reads the existing HyBiomass archives and saves matched inputs to `data/enmap/<region>/<tile>/<patch>.npz`. It checks the source TIFF's CRS, transform and valid labels against the TESSERA-prepared patch, and never overwrites files. Original integer-valued reflectances are converted to uint16 only when exact; otherwise float32 is retained. Restarting skips existing files and re-reads the compressed archives. Three separate region workers run concurrently, showing progress per patch.

Each file includes per-band sums, squared sums and counts. The training program combines **only training patches'** statistics to calculate one mean and standard deviation per EnMAP band; validation/test inputs are normalized with those same values. Non-finite input values are imputed to zero *after normalization* (the band mean); targets are not discarded for this reason, preserving the exact matched label mask. Target normalization is taken from the reference run.

## Train

```bash
.venv-v11/bin/python -u train_hybiomass.py \
  --input enmap --enmap-data data/enmap --data data/prepared \
  --reference-run data/runs/tessera_v11_unet_02 \
  --run data/runs/enmap_unet_01 --batch-size 4 --workers 6 --epochs 80
```

The run directory must not exist. To resume after a completed epoch, repeat the same command with `--resume`; it loads `last.pt` including optimizer and random state. Results and per-band normalization values are recorded in the new run directory. For the matched constant predictor, run `.venv-v11/bin/python mean_baseline.py`.

The 12-patch pilot ran on the server's CUDA device reported as NVIDIA L40; batch size 4 used approximately 1.1 GiB peak allocated GPU memory. The pilot is a pipeline check only. The full input extraction requires substantial disk; the 12-patch pilot occupied about 57 MiB, but actual compression varies. No pretrained HyBiomass encoder weights are used. This is a controlled comparison to our TESSERA setup, **not** an exact reproduction of the HyBiomass paper's U-Net architecture, normalization or splits.
