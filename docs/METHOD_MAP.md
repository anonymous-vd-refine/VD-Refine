# Method-to-code map

The trainer chain is retained to preserve checkpoint compatibility and original computation. Only the required chain is packaged; users select the two short seed configs or the historical inference artifact.

| Component | Implementation |
|---|---|
| Architecture and final-step decode | `nnunetv2/nets/LiteRBUNeXt3D.py`: `LiteRBUNeXt3DFeatureRefiner.forward` |
| Analytic estimate | Same class: `_forward_inf_extrapolate` |
| Portable workflow | `scripts/run.py` |
| Exact cohort and geometry-aware metrics | `scripts/evaluate.py`, `dataset/cohort.json` |

## Trainer inheritance

- `nnUNetTrainer` → `object`
- `nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2` → `nnUNetTrainerLiteStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderUNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMUNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMUNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanUNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMultiStartK8TBPTT2ABCUNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMultiStartK8TBPTT2ABCUNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8UNeXt3DStableV2`
- `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8UNeXt3DStableV2` → `nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2`
- `nnUNetTrainerLiteSDRRAlternatingABCDec16Dec32UNeXt3DStableV2` → `nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2`
- `nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2` → `nnUNetTrainerLiteStableV2`
- `nnUNetTrainerLiteStableV2` → `nnUNetTrainer`
- `nnUNetTrainer_brats802_PaperSeed2027` → `PaperReplicateSeedMixin`, `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2`
- `nnUNetTrainer_brats802_PaperSeed2028` → `PaperReplicateSeedMixin`, `nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2`
- `PaperReplicateSeedMixin` → 

The release does not collapse these classes into a new monolithic trainer. Executable method ASTs are compared to the source snapshot, and numerical probes compare every inference mode plus updates in each of the five training phases.
