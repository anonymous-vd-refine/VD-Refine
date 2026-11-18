# BraTS protocol and provenance

## Cohort and selection

BraTS2020 has 369 cases in this experiment. The published cohort is 236 training, 59 validation and 74 held-out test volumes. Fold 0 of the included five-fold development split was used; this is **one fold**, not an ensemble of five trained folds. `dataset.json:numTraining=295` denotes the entire development pool, not the gradient-training subset.

The original dataset-description string said 258/37/74. Historical VD-Refine and its FiLM Stage-A source logs instead record 236/59, and all finite-depth result rows record 74 test cases. This release corrects the description without changing the cohort. The historical K32 CSV average is 0.852584161836534, matching 85.26% in the manuscript.

Patch: 128³. Batch size: 2. Spacing: 1mm³. Z-score normalization uses nonzero masks for all four channels. The supplied plan uses `batch_dice=false`; do not substitute a separate BatchDiceLoss trainer. The network has 4,399,737 parameters and decoder block counts [1,1,1,1,1].

## Training

100 total epoch indices 0–99; no early stopping. Stage A: 0–49; B1: 50–69; C1: 70–79; B2: 80–89; C2: 90–99. AdamW, initial LR 1e-4, weight decay 1e-5; per-group LRs change with stage. Deep supervision is disabled at initialization. The B phases expose only `single_refiner.shared_block` to optimization; FiLM is frozen. The main recipe has no K0 auxiliary objective from KcovAB.

Stage A samples K=1 (probability 0.6) or K=2. The B-phase real starts come from bands [0,1], [2,4], [4,6], with two differentiable corrections. Virtual starts: B1 uniform integers [8,14]; B2 0.6×uniform[8,14]+0.4×uniform[15,30]. The endpoint is the start plus two real corrections. Virtual loss weights ramp 0.05→0.20 across B1 and 0.20→0.25 across B2. Odd virtual starts use fractional powers of the two-step decay ratio.

### Historical checkpoint and scratch training

The historical run loaded the plain-FiLM Stage-A checkpoint and logger state, then continued at epoch 50 with a fresh optimizer. Its Stage-A source used the same recorded 236/59 split. The final recipe ran through epoch 99. The released runner trains the original recipe from scratch and disables inherited warm-start paths. Historical numbers are reference results; a new run is not guaranteed to reproduce them exactly.

## Inference and metrics

Use checkpoint_best selected by validation, fold 0, full-volume sliding-window prediction with step size 0.5, Gaussian blending, mirroring on axes 0/1/2. Finite K=0/1/2/4/8/16/32; analytic `inf` uses real d4/d6/d8 and q clipped to [0,0.95], then restores conditioned moments and decodes once. Default finite inference calls the decoder tail/head once. It is a geometric-series estimate, not a convergence guarantee.

All 74 full test volumes are required per mode. The reported mean is a mean of case-level means; because regions can be NaN, it need not equal the unweighted mean of the three region-column means. The release evaluator preserves the historical arithmetic and adds cohort, label and physical-geometry validation.

Do not choose K, thresholds or checkpoints by optimizing the held-out test labels. No postprocessing search is provided in the release workflow.

The inference export's `best_checkpoint_logged_epochs` records the logger length at validation-best selection, not the final training budget. A best checkpoint can precede epoch 99 even though training subsequently completes E100.
