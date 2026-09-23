# Release verification

Checks were run on the source snapshot and the extracted release, without modifying the source experiment or using its GPUs.

## Passed checks

1. **Cohort**: 236/59/74 unique cases, disjoint sets, exact label and four-modality filename sets in the actual dataset. The supplied fold-0 list is the recorded split.
2. **Method-source equivalence**: 111 required source modules were checked by AST. Method computations are unchanged. Documentation and machine-specific standalone examples are excluded from this comparison; `paths.py` intentionally changes only the default data root. See `verification/source_equivalence.json` and `source_manifest.json`.
3. **Network and trainer**: 4,399,737 parameters, decoder [1,1,1,1,1], E100, AdamW, no deep supervision after initialization, two seed entrypoints import successfully.
4. **Inference**: CPU FP32, deterministic synthetic input [1,4,64,64,64], supplied historical weights. K=0/1/2/4/8/16/32 and analytic inf output hashes are identical between source and release. Finite fast-path logits are bit-identical to explicit all-step decoding. Hooks verify one final decoder-tail/head call and K refiner calls (eight for inf).
5. **Training updates**: one synthetic training step in A, B1, C1, B2 and C2. Losses and updated state-dictionary hashes match source and release exactly. B1/B2 have 11 trainable shared-block tensors and preserve all other network tensors.
6. **Preprocessing**: a real held-out case was processed using both implementations; preprocessed image and segmentation array hashes match exactly. See `verification/preprocessing.json`.
7. **Historical evaluation**: strict evaluator re-read all existing historical full-volume predictions: 74 cases × 8 modes = 592 case-mode rows, 2368 region/average Dice values. Every value matches the original per-case CSV, including NaNs; maximum finite absolute difference is **0**. No full-cohort inference was rerun for this check. Results: `results/strict_historical_scaling_dice*.csv`.
8. **Evaluator boundaries**: tests for empty masks, correct region conversion, missing/extra cases, physical-geometry mismatches and per-case aggregation pass. E100-gate tests reject 99/101 epochs and missing final checkpoints, and accept exactly 100 logged epochs with a best checkpoint present.
9. **Inference artifact**: 260 network tensors in the anonymized historical checkpoint are exactly equal to the original checkpoint. Predictor initialization resolves the retained trainer and loads the tensor dictionary strictly.
10. **Packaging**: a wheel builds and imports from a separate virtual environment rather than the original editable source. A 78-package runtime lock resolves the original numexpr/blosc2 conflict; its active requirement graph is validated. The installed package with corrected dependencies is numerically checked against the original report.
11. **Anonymity**: personal filesystem roots/private addresses and inference-checkpoint metadata are scanned. Project-author identifiers and Git history are excluded from the archive. Third-party names, copyright and license text are deliberately preserved.

## Limits

The numerical probes use a 64³ CPU patch, while training uses 128³ patches and CUDA. This release verification does not rerun E100, all 74 full-volume predictions, or cross-device reproducibility. Two new seeds remain incomplete in the dated status manifest. Installing the wheel in a virtual environment reused the existing third-party runtime and overlaid corrected packages; this is not a fresh download/install of every CUDA dependency. The historical weights are inference-only and cannot resume training.

The package records historical results and the exact reproduction protocol without claiming that a fresh seeded run must reproduce the historical numbers bit-for-bit.
