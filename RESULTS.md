# HyBiomass biomass prediction: current results

## Summary

On the same held-out HyBiomass test patches and GEDI labels, a U-Net trained on area-aggregated TESSERA v1.1 embeddings achieved a pixel-weighted RMSE of **76.34 Mg/ha**, compared with **89.18 Mg/ha** for the matched U-Net trained from scratch on EnMAP imagery. This is a **14.4% lower RMSE** for TESSERA on this split. These are results from one run per input type, not an uncertainty estimate.

## Dataset and preparation

- HyBiomass contains EnMAP imagery and sparse GEDI-derived above-ground biomass density (AGBD) estimates. The labels are modeled GEDI estimates, not direct field measurements.
- Seven regional archives were downloaded and verified against DLR's published MD5 checksums.
- TESSERA v1.1/dclimate embedding availability was checked at all 14,096,147 labeled-pixel centers for each patch's EnMAP year: **14,050,658 (99.68%)** had a finite scale in the nominal UTM zone. A small number of missing locations may be available in an overlapping neighboring zone.
- TESSERA embeddings were read from their native 10 m grid and area-averaged onto the exact HyBiomass 30 m patch grid, retaining all 128 channels. A 30 m input is valid only when its contributing source area is fully valid. Missing embeddings are filled with zero for the CNN; labels without valid embeddings are excluded from loss and metrics.
- After alignment and masking, **34,223 patches** and **14,030,381 GEDI-labeled pixels** remained usable. Nine of 34,232 original patches had no usable labels and were skipped.
- EnMAP inputs preserve all 202 original bands. The 202 band means and standard deviations were computed from the training patches only. Non-finite EnMAP values are imputed to the band mean after normalization; the comparison uses exactly the TESSERA-prepared target masks for both inputs.
- Embeddings were matched to the EnMAP acquisition year, which is not necessarily the GEDI observation year. The annual embedding may include observations after the GEDI measurement.

For context, our earlier **TESSERA v2 beta1** check found that 621,066 of 14,096,147 labeled-pixel centers (4.41%) fell inside a listed tile for their EnMAP year. This was an inventory check, not an actual-pixel validity check, so it is not directly equivalent to the v1.1 figure.

## Matched experiment

Both models use the U-Net design described for BioMassters in the [TESSERA supplement, section A.11](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Feng_TESSERA_Temporal_Embeddings_CVPR_2026_supplemental.pdf): two encoder and decoder stages, 256/512 encoder channels, a 1024-channel bottleneck and one regression output. The first convolution has 128 input channels for TESSERA and 202 for EnMAP. The EnMAP model is trained from scratch; it is not a pretrained hyperspectral foundation model.

- Both runs used the exact same split manifest and target normalization, seed 42, batch size 4, Adam at learning rate 0.0001, BF16 autocast and 80 epochs on one NVIDIA GPU reported by PyTorch as an L40.
- The loss is masked RMSE averaged per patch. The checkpoint with the lowest validation mean patch RMSE is selected.
- The split groups patches by source tile ID and uses approximate 70/20/10 train/validation/test proportions. These are not the HyBiomass paper's published splits, and grouping by tile does not guarantee spatially independent geographic holdouts.
- There were 24,318 training, 6,723 validation and 3,182 test patches. Both models were tested against the same **1,340,952 valid labeled test pixels**. The constant baseline always predicts the training-label mean, 145.64 Mg/ha.
- Test predictions were clipped to zero before evaluation; validation predictions used for checkpoint selection were not clipped.

## Held-out test results

| Input to predictor | Pixel-weighted RMSE ↓ (Mg/ha) | Mean patch RMSE ↓ (Mg/ha) | R² ↑ |
|---|---:|---:|---:|
| **TESSERA v1.1 + U-Net** | **76.34** | **70.43** | **0.541** |
| EnMAP + U-Net, trained from scratch | 89.18 | 81.53 | 0.373 |
| Constant training-set mean | 112.65 | 107.40 | approximately 0 |

Pixel-weighted RMSE pools errors across all valid labeled pixels. Mean patch RMSE first calculates RMSE within each patch, then averages patches equally. Both use Mg/ha, equivalent to tonnes per hectare.

### Test pixel-weighted RMSE by region

| Region | Test patches | TESSERA RMSE ↓ (Mg/ha) | EnMAP RMSE ↓ (Mg/ha) |
|---|---:|---:|---:|
| Africa | 65 | 59.75 | **59.44** |
| Australasia | 409 | **84.40** | 92.26 |
| Europe | 510 | **55.89** | 67.92 |
| North America | 971 | **77.44** | 95.27 |
| North Asia | 452 | **76.43** | 87.13 |
| South America | 437 | **78.07** | 84.77 |
| South Asia | 338 | **89.27** | 100.21 |

Regional RMSE is affected by biomass distributions, test-set size and GEDI label uncertainty; it is not a complete ranking of geographic difficulty.

### Test pixel-weighted RMSE by GEDI reference biomass

| GEDI reference bin (Mg/ha) | Test labels | TESSERA RMSE ↓ (Mg/ha) | EnMAP RMSE ↓ (Mg/ha) |
|---|---:|---:|---:|
| 100 to <200 | 382,177 | 61.81 | **56.25** |
| 200 to <300 | 228,146 | **71.46** | 81.32 |
| 300 to <400 | 102,160 | **118.42** | 150.21 |
| 400 and above | 47,693 | **186.47** | 233.55 |

The bins use the original GEDI reference value, not the prediction. Intervals include their lower bound and exclude their upper bound; the last bin includes 400. The 580,776 test labels below 100 Mg/ha are excluded from the requested table. Both models were evaluated on the same pixels using their best-validation checkpoints and nonnegative-clipped predictions. RMSE is pooled over pixels in each bin; it is not mean patch RMSE. Higher absolute error in the high-biomass bins does not by itself establish whether predictions are systematically too high or too low.

### Per-region and biomass-bin test RMSE

Each cell is **TESSERA / EnMAP** pixel-weighted RMSE in Mg/ha; lower is better. Bins refer to the GEDI reference and use the same boundaries as above.

| Region | 100 to <200 | 200 to <300 | 300 to <400 | 400+ |
|---|---:|---:|---:|---:|
| Africa | 86.89 / 71.56 | 108.09 / 103.21 | 182.09 / 184.90 | 283.61 / 277.31 |
| Australasia | 65.23 / 62.78 | 73.34 / 75.84 | 131.48 / 145.10 | 205.30 / 227.71 |
| Europe | 51.72 / 50.02 | 62.12 / 75.77 | 111.94 / 143.86 | 199.02 / 240.32 |
| North America | 66.31 / 56.08 | 73.76 / 89.89 | 108.08 / 155.98 | 167.33 / 234.97 |
| North Asia | 54.92 / 53.47 | 76.36 / 77.99 | 119.04 / 141.12 | 182.79 / 219.44 |
| South America | 62.30 / 63.20 | 69.75 / 67.28 | 141.47 / 143.64 | 243.96 / 249.14 |
| South Asia | 61.62 / 54.91 | 67.60 / 80.21 | 121.60 / 150.54 | 199.76 / 235.36 |

Test GEDI pixel counts, identical for both models:

| Region | 100 to <200 | 200 to <300 | 300 to <400 | 400+ |
|---|---:|---:|---:|---:|
| Africa | 2,799 | 1,327 | 298 | 59 |
| Australasia | 42,706 | 28,205 | 14,209 | 7,739 |
| Europe | 61,705 | 36,794 | 8,120 | 1,065 |
| North America | 148,054 | 83,426 | 45,897 | 24,868 |
| North Asia | 55,537 | 28,873 | 11,666 | 4,605 |
| South America | 33,012 | 24,688 | 9,396 | 3,149 |
| South Asia | 38,364 | 24,833 | 12,574 | 6,208 |

The highest Africa bin has only 59 test labels; results in small or spatially correlated samples may be unstable. No uncertainty intervals were calculated.

## Validation and overfitting

| Run | Best epoch | Best validation pixel RMSE (Mg/ha) | Best validation mean patch RMSE (Mg/ha) | Best validation R² | Epoch 80 validation mean patch RMSE (Mg/ha) |
|---|---:|---:|---:|---:|---:|
| TESSERA + U-Net | 26 | 74.50 | 68.69 | 0.555 | 74.66 |
| EnMAP + U-Net | 35 | 86.27 | 79.72 | 0.404 | 85.51 |

Training loss continued decreasing after each run's best validation checkpoint while validation error worsened. This pattern is consistent with overfitting, although validation fluctuates. Test metrics above use the saved best-validation checkpoints, not epoch 80.

## Interpretation and limitations

- Under this specific matched setup, aggregated TESSERA v1.1 embeddings were useful for predicting held-out GEDI estimates and outperformed a scratch-trained EnMAP U-Net overall and in six of seven regions.
- The inputs differ in more than sensor: TESSERA uses an annual Sentinel-1/2 time series and pretraining, whereas EnMAP is a single hyperspectral acquisition. These results do not establish which difference caused the gap.
- The TESSERA paper reports 27.43 Mg/ha mean patch RMSE on BioMassters, but that Finnish 10 m dataset has different labels, geography and splits. Its number is **not directly comparable** with our HyBiomass scores.
- Results are from one split and seed. The current test set has already been inspected; avoid using it repeatedly for model selection. Repeat runs and a separately defined geographic or external holdout are needed for stronger claims.
- Our EnMAP comparison is not an exact reproduction of the published HyBiomass benchmark. Its original pretrained hyperspectral models and published patch-level splits have not been evaluated here.

## Artifacts on siena09

- TESSERA run: `~/tessera-test/data/runs/tessera_v11_unet_02/`
- EnMAP run: `~/tessera-test/data/runs/enmap_unet_01/`
- Constant baseline metrics: `~/tessera-test/data/runs/mean_baseline.json`
- Per-biomass-bin test results: `~/tessera-test/data/runs/per_bin_rmse.json`
- Per-region, per-bin test results: `~/tessera-test/data/runs/per_region_bin_rmse.json`
- Each model run contains `manifest.json`, `metrics.jsonl`, `best.pt`, `last.pt` and `test_metrics.json`.
- Preparation/training details: [`EXPERIMENT.md`](EXPERIMENT.md) and [`ENMAP_BASELINE.md`](ENMAP_BASELINE.md).
