# VD-Refine: BraTS reproduction package

Anonymous review release **v1.0.0** for *Virtual-Depth Supervision for Stable Recurrent Medical Image Segmentation*.

This package contains the BraTS2020 VD-Refine recipe (VirtualDeep + FrozenFiLM + RandDepth), its required nnU-Net runtime, the fixed **236 train / 59 validation / 74 test** split, and metadata for the anonymized **historical checkpoint_best**. The binary weights are distributed separately in the full release archive. Other datasets and experiment variants are outside the release scope.

## Code-only anonymous mirror

This GitHub snapshot omits the 17 MB `.pth` file to meet the anonymous mirror's per-file limit. Obtain the companion `vd-refine-brats-v1.0.0.zip` full release archive from the submission supplementary material when available, and copy `weights/historical/fold_0/checkpoint_best.pth` into the same path here. Verify its SHA256 against `weights/historical/provenance.json`. Until that file is supplied, historical inference and the model verification probe cannot run. Fresh training does not require it. This repository alone does not contain pretrained weights.

## What can be reproduced

- Historical model inference: K = 0, 1, 2, 4, 8, 16, 32 and analytic `inf`, with TTA and all 74 test cases.
- Historical K32 mean Dice: **85.258416%**. The finite-depth reference table is in `results/historical_scaling_dice.csv`.
- Fresh training: seed 2027 or 2028, fold 0, 100 total epochs. These are **new replicates, not the historical model's original seeds**. Their E100 results were not complete at packaging time; no unfinished replicate results are presented as completed experiments.
- Historical weights are inference-only. Optimizer, training logger, paths, and resume metadata have been excluded; all exported network tensors are bit-identical to the source checkpoint. See `weights/historical/provenance.json`.

Read `docs/PROTOCOL.md` for the historical Stage-A reuse and scratch-training distinction, and `docs/VERIFICATION.md` for the exact scope of checks. No claim of bitwise reproducibility across hardware or a full retraining reproduction is made.

## Install

Use a separate Linux environment with Python 3.10. Do not install alongside another nnunetv2 source tree.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==24.2" "setuptools==83.0.0" "wheel==0.44.0"
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python tests/test_evaluate.py
```

The lock targets the captured Linux/CUDA runtime. See `docs/ENVIRONMENT.md`; CUDA training requires sufficient GPU memory. Commands below run from the release root.

## Prepare BraTS2020

Obtain the original BraTS2020 training images under the dataset provider's access terms. Images are not included. `--source` must contain directories such as `BraTS20_Training_001`, each with `_t1`, `_t1ce`, `_t2`, `_flair`, and `_seg` NIfTI files.

```bash
python scripts/convert_brats.py --source /path/to/BraTS2020_TrainingData --output data/nnUNet_raw/Dataset802_BraTS
python scripts/validate_dataset.py --raw data/nnUNet_raw/Dataset802_BraTS
python scripts/run.py preprocess --data data --workers 4
```

The converter uses the explicit case list, not a newly sampled split. It maps original labels `0→0, 1→2, 2→1, 4→3`. Channel order is T1, T1ce, T2, FLAIR. nnU-Net `imagesTr/labelsTr` contain all **295** development cases; `splits_final.json` selects **236/59** for fold 0. `imagesTs/labelsTs` contain **74** held-out cases. The supplied plans preserve the experiment's patch, spacing, batch size and normalization. Do not regenerate a different fold or plan for a claimed exact-protocol reproduction.

## Historical checkpoint inference and evaluation

```bash
python scripts/run.py predict --historical --data data --output runs --gpu 0
python scripts/run.py evaluate --historical --data data --output runs
```

Predictions are saved under `runs/historical/testK{K}_best`. TTA is enabled for every mode. The `inf` mode executes eight real updates, extrapolates from d4/d6/d8 and decodes the normalized estimate; it does not run an extremely long trajectory and is not a proven exact fixed point.

To run a subset, pass for example `--steps 8 32 inf` to **both** commands. The evaluator always requires all 74 cases for each requested mode. Full-volume prediction needs only the converted raw test images and supplied weights/plans; preprocessing is needed for training.

## Fresh E100 training

```bash
python scripts/run.py train --seed 2027 --data data --gpu 0
python scripts/run.py predict --seed 2027 --data data --output runs --gpu 0
python scripts/run.py evaluate --seed 2027 --data data --output runs
# Repeat with --seed 2028 in a separate run namespace.
```

Only use `python scripts/run.py resume --seed 2027 --data data --gpu 0` for an intentional continuation. The runner refuses to overwrite existing training, prediction or evaluation output. Seed prediction requires a final checkpoint with exactly 100 logged epochs, and uses `checkpoint_best.pth`. No destructive `pipeline` action is provided. For a new prediction attempt, choose a new `--output` directory.

The runner clears inherited inference overrides and Stage-A warm-start variables. Python/NumPy/Torch and augmentation-worker seeds follow the original seed mixin. Asynchronous augmentation and backend behavior are inherited; exact repeated-run determinism is not claimed.

## Evaluation rules

WT={1,2,3}, TC={2,3}, ET={3}. Empty GT **and** empty prediction give NaN; empty GT with a nonempty prediction gives 0. Each case's average is `nanmean(WT,TC,ET)` and the cohort average is the mean over valid case averages. Report NaN counts. Missing/extra cases, unexpected labels, mismatched shapes or physical geometry fail evaluation instead of reducing the cohort silently.

## Source organization

- `nnunetv2/nets/LiteRBUNeXt3D.py`: architecture, finite-K fast path and analytic extrapolation.
- `nnunetv2/training/nnUNetTrainer/`: two seed entries and the necessary inheritance chain. Long legacy names are retained for checkpoint compatibility; they are not additional advertised experiments.
- `configs/`, `dataset/`: fixed recipe and exact cohort/plans.
- `scripts/`: portable data conversion, training, prediction and strict evaluation.
- `tests/`, `verification/`: checks and recorded results.
- `weights/historical/`, `results/`: historical inference artifact and reference metrics, with provenance.

This snapshot contains no Git history or project-author identity metadata. Upstream attribution and licensing are intentionally retained in `LICENSE` and `THIRD_PARTY_NOTICES.md`. Dataset access terms are separate from the code license. The supplied checkpoint is derived model data, not a redistribution of images.
