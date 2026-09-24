# VD-Refine

Anonymous code for **Virtual-Depth Supervision for Stable Recurrent Medical Image Segmentation**.

[Project page](https://anonymous-vd-refine.github.io/VD-Refine/) · [BraTS protocol](docs/PROTOCOL.md) · [LiTS / DRIVE protocol](docs/MULTIDATASET.md) · [Verification](docs/VERIFICATION.md)

VD-Refine combines initial-state-conditioned FiLM, multi-start correction training, and geometric virtual-depth supervision. An eight-step real training prefix supplies virtual supervised endpoints through depth 32. At inference, one checkpoint supports a finite rollout or an analytic state estimate.

## Released datasets and reported results

| Dataset | Train / validation / test | Checkpoint | TTA | K=32 Dice (%) | Analytic Dice (%) |
|---|---|---|---|---:|---:|
| BraTS2020 | 236 / 59 / 74 volumes | validation best | on | 85.26 | 85.24 |
| LiTS | 92 / 13 / 26 volumes | final | off | 79.73 | 79.95 |
| DRIVE | 16 / 4 / 20 images | validation best | on | 82.32 | 82.35 |

These are historical, single-checkpoint results from complete held-out cohorts. The public package contains the required model/trainer implementation, fixed cohorts and plans, portable conversion/training/prediction/evaluation tools, and reference measurements. Dataset images and pretrained binary weights are not included. Fresh training does not require pretrained weights. New training is not guaranteed to reproduce historical scores exactly; no full E100 retraining was performed to validate this release.

## Install

Use a separate Linux Python 3.10 environment, without another nnunetv2 checkout on `PYTHONPATH`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install "pip==24.2" "setuptools==83.0.0" "wheel==0.44.0"
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python tests/test_evaluate.py
python tests/test_multidataset.py
```

The lock captures the Linux/CUDA runtime. See [environment notes](docs/ENVIRONMENT.md). Commands below run from this repository root.

## BraTS

Obtain the provider's BraTS2020 training data. The converter preserves the explicit published cohort and maps the original segmentation labels to WT/TC/ET regions.

```bash
python scripts/convert_brats.py --source /path/to/BraTS2020_TrainingData --output data/nnUNet_raw/Dataset802_BraTS
python scripts/run.py preprocess --data data
python scripts/run.py train --data data --gpu 0
python scripts/run.py predict --data data --output runs --gpu 0
python scripts/run.py evaluate --data data --output runs
```

For separately supplied historical BraTS weights, use `--historical` for prediction and evaluation after installing the model at `weights/historical`. Check its checksum against [provenance](weights/historical/provenance.json). See [BraTS protocol](docs/PROTOCOL.md) for historical Stage-A reuse versus scratch training.

## LiTS and DRIVE

The LiTS cohort was rebuilt from the verified **MSD Task03_Liver** archive, mapping `liver_n` to `Liver_{n+1:04d}`. The converter expects its `imagesTr` and `labelsTr` directories. DRIVE expects the official `training` and `test` directories, RGB images, and first-observer manual annotations. Obtain data under the original provider's access terms.

```bash
python scripts/prepare_dataset.py convert --dataset lits --source /path/to/Task03_Liver --raw data/nnUNet_raw/Dataset822_LiverCT_Seed2026_Corrected
python scripts/prepare_dataset.py convert --dataset drive --source /path/to/DRIVE --raw data/nnUNet_raw/Dataset804_DRIVE

python scripts/run_dataset.py preprocess --dataset lits --data data
python scripts/run_dataset.py train --dataset lits --data data --gpu 0
python scripts/run_dataset.py predict --dataset lits --data data --output runs --gpu 0
python scripts/run_dataset.py evaluate --dataset lits --data data --output runs
# Repeat the four commands with --dataset drive.
```

The runners select the dataset's original paper trainer and checkpoint/TTA protocol automatically. Prediction requires a completed E100 run. An external historical model can instead be passed with `--model /path/to/model` to prediction/evaluation; it must contain `fold_0/checkpoint_{best|final}.pth`, `plans.json`, and `dataset.json`. Its trainer class must be included in this repository. There are no LiTS or DRIVE pretrained weights in this snapshot.

`--steps 8 32 inf` selects a subset for both prediction and evaluation. `inf` executes eight real updates and decodes a normalized geometric estimate; it is not a guaranteed feature fixed point. Existing output directories are protected against overwrite. Use `resume` only for an intentional continuation, or choose new data/output roots for independent runs.

## Evaluation and provenance

- BraTS: WT={1,2,3}, TC={2,3}, ET={3}; per-case region mean, then cohort mean.
- LiTS: whole liver={1,2}, tumor={2}; per-case region mean, then cohort mean.
- DRIVE: vessel={1}; full-image Dice using the first observer, without an added FOV mask.
- Empty GT and empty prediction give NaN; empty GT with nonempty prediction gives zero. NaNs and complete case counts are reported.
- Missing/extra cases, invalid labels or incompatible geometry fail evaluation. No test-set threshold or postprocessing search is performed.

`results/` contains reference cohort and case-level measurements. [LiTS / DRIVE notes](docs/MULTIDATASET.md) record the additional dataset-specific training behavior, source equivalence and validation scope.

## Layout

`nnunetv2/` contains the required runtime, network and original trainer chains. Long class names are retained for checkpoint compatibility. `configs/` and `dataset/` hold the three protocols. `scripts/`, `tests/`, `verification/` provide runners and checks. `docs/` contains the project website and research documentation.

Code licensing and retained upstream attribution are in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset access terms are separate.
