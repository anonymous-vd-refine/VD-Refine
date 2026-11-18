# Third-party attribution

The bundled nnU-Net runtime derives from nnU-Net v2 (the source distribution identifies itself as 2.1.1), by the Division of Medical Image Computing / German Cancer Research Center and contributors. The Apache-2.0 license supplied with that source is retained verbatim in `LICENSE`; existing copyright notices remain in source files.

Please cite Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods (2021), when using the framework. The upstream project is https://github.com/MIC-DKFZ/nnUNet . The package is a modified research snapshot, not an unmodified upstream release.

The research source was maintained in a LightM-UNet-derived nnU-Net tree. The BraTS method uses the included LiteRBUNeXt3D architecture and VD-Refine training implementation; the release does not redistribute other baseline architectures or their results as this method. Relevant methodological references include LightM-UNet (Liao et al., 2024) and UNeXt (Valanarasu and Patel, 2022).

The release adds portable entrypoints, a strict BraTS evaluator, fixed-cohort metadata, documentation, and tests. It removes unrelated experiment modules and machine-specific runnable examples. Core method computations and checkpoint tensor values are preserved; `verification/source_manifest.json` records source hashes and the verification report describes the equivalence checks.

Dependencies (including PyTorch, NumPy, SciPy, SimpleITK, batchgenerators, acvl-utils and dynamic-network-architectures) are installed separately and retain their respective licenses. They are not vendored into this archive. BraTS data must be obtained separately under its own terms. Upstream author and institutional names retained for attribution do not identify this submission's authors.
