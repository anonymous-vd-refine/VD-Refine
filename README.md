# VD-Refine

Anonymous code for **Virtual-Depth Supervision for Stable Recurrent Medical Image Segmentation**.

[Project page](https://anonymous-vd-refine.github.io/VD-Refine/) · [Method](docs/METHOD_MAP.md) · [Verification](docs/VERIFICATION.md)

VD-Refine combines initial-state-conditioned FiLM, multi-start correction training, and geometric virtual-depth supervision. An eight-step real training prefix supplies virtual supervised endpoints through depth 32. At inference, one checkpoint supports a finite rollout or an analytic state estimate.

## Released datasets and reported results

| Dataset | Train / validation / test | K=32 Dice (%) | Analytic Dice (%) |
|---|---|---:|---:|
| BraTS2020 | 236 / 59 / 74 volumes | 85.26 | 85.24 |
| LiTS | 92 / 13 / 26 volumes | 79.73 | 79.95 |
| DRIVE | 16 / 4 / 20 images | 82.32 | 82.35 |

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

## Data preparation, training and evaluation

The release supports BraTS2020, LiTS and DRIVE through the same workflow: prepare the data, preprocess, train, predict and evaluate. Obtain the source data under the original providers' access terms. The converters preserve the published cohorts and prepare the images and labels for training and evaluation.

| Dataset | Source data |
|---|---|
| BraTS2020 | The provider's BraTS2020 training data. |
| LiTS | **MSD Task03_Liver**, with its `imagesTr` and `labelsTr` directories. |
| DRIVE | The official `training` and `test` directories, with RGB images and first-observer manual annotations. |

Run the conversion command for the dataset you want to use:

```bash
python scripts/convert_brats.py --source /path/to/BraTS2020_TrainingData --output data/nnUNet_raw/Dataset802_BraTS
python scripts/prepare_dataset.py convert --dataset lits --source /path/to/Task03_Liver --raw data/nnUNet_raw/Dataset822_LiverCT_Seed2026_Corrected
python scripts/prepare_dataset.py convert --dataset drive --source /path/to/DRIVE --raw data/nnUNet_raw/Dataset804_DRIVE
```

Then select `brats`, `lits` or `drive` and run the following in Bash. The runner loads the corresponding configuration automatically.

```bash
DATASET=brats  # brats, lits or drive
case "$DATASET" in
  brats) runner=(python scripts/run.py) ;;
  lits|drive) runner=(python scripts/run_dataset.py --dataset "$DATASET") ;;
  *) echo "Choose brats, lits or drive"; exit 1 ;;
esac

"${runner[@]}" preprocess --data data
"${runner[@]}" train --data data --gpu 0
"${runner[@]}" predict --data data --output runs --gpu 0
"${runner[@]}" evaluate --data data --output runs
```

Prediction follows a completed E100 training run. Add `--steps 8 32 inf` to prediction and evaluation to select a subset of inference depths. `inf` executes eight real updates and decodes a normalized geometric estimate; it is not a guaranteed feature fixed point. Existing output directories are protected against overwrite. Use `resume` to continue an existing training run, or choose new data/output roots for an independent run.

## Evaluation and reference results

Evaluation reports Dice over each complete held-out cohort, with case-level measurements and cohort summaries. It checks case counts, labels and image geometry before scoring, and reports empty-region cases explicitly. No test-set threshold or postprocessing search is performed.

`results/` contains the reference measurements. Fixed splits and configurations are provided in `dataset/` and `configs/`; implementation and validation details are documented in `docs/`.

## Layout

`nnunetv2/` contains the required runtime, network and original trainer chains. Long class names are retained for checkpoint compatibility. `configs/` and `dataset/` hold dataset configurations, plans and fixed splits. `scripts/`, `tests/`, `verification/` provide runners and checks. `docs/` contains the project website and research documentation.

Code licensing and retained upstream attribution are in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Dataset access terms are separate.
