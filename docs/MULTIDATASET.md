# LiTS and DRIVE reproduction protocols

## Fixed cohorts

LiTS uses `Dataset822_LiverCT_Seed2026_Corrected`: 92 training, 13 validation and 26 held-out volumes. This is the verified MSD Task03_Liver source, with `liver_n` mapped to `Liver_{n+1:04d}`. The supplied cohort and fold-0 split are authoritative. The historical dataset identifier is retained for compatibility.

DRIVE uses `Dataset804_DRIVE`: the official 20-image training pool is split into 16 training and 4 validation images; all 20 official test images are held out. RGB channels are stored in one `_0000.tif` file and expanded by NaturalImage2DIO. Labels are the first observer's manual vessel annotations, mapped to 0/1.

Images are not redistributed. Plans and normalization statistics are included. LiTS uses 128³ 3D patches, CT normalization and batch size 2. DRIVE uses 512×512 2D patches, RGB z-score normalization and batch size 2. Do not regenerate different plans or folds for an exact-protocol comparison.

## Original paper trainers

- LiTS: `nnUNetTrainer_LiTS822_KcovAB`, defined in `nnUNetTrainer_LiTS822_KcovA.py`.
- DRIVE: `nnUNetTrainer_DRIVE804_KcovAB`, defined in `nnUNetTrainer_2D_KcovAB.py`.

Both use the historical decoder with an E100 schedule: A=50, B1=20, C1=10, B2=10, C2=10 epochs. Stage A includes the original K=0 auxiliary objective (weight 0.8). Their B phases preserve the proposal through a separate backbone objective; unlike the BraTS recipe, the backbone is not entirely frozen in B. FiLM remains frozen. The original implementations and inheritance chains are retained without changing computation.

DRIVE uses the established singleton-depth adapter for the shared 3D implementation with native 2D I/O. This is the architecture used for the reported results, not a newly substituted 2D network.

## Prediction and evaluation

LiTS uses `checkpoint_final.pth`, no TTA; DRIVE uses validation-selected `checkpoint_best.pth`, TTA. Each setting uses the same checkpoint. Sliding-window step size is 0.5 with the existing Gaussian blending implementation. Evaluate K=0,1,2,4,8,16,32 and `inf`.

LiTS whole-liver masks are the union of labels 1 and 2; tumor is label 2. DRIVE uses vessel label 1 over the complete image, without an extra FOV mask. Empty/empty gives NaN. Cohort means are means of per-case region means. All 26 or 20 cases must be present for every requested setting.

`results/lits_depth_results.csv` and `results/drive_depth_results.csv` match the original full-cohort evaluation tables. The accompanying case CSVs were checked against the fixed test IDs and reproduce all eight reported means. Reference measurements use historical checkpoints; this release does not include binary weights or claim a new training/evaluation run.

## Release verification

The additional trainers and their imported library computation were checked against the research source. Synthetic tests cover data conversion, disjoint fixed cohorts, region semantics, perfect masks and missing-case rejection. CPU inference smoke checks exercise the original architectures at finite depth and analytic mode; their exact shapes and scope are recorded in `verification/multidataset_smoke.json`. No training jobs or full-cohort inference were launched for packaging.
