# Environment

Target: Linux x86_64, Python 3.10, PyTorch 2.4.0 with its CUDA 12.1 runtime. The original experiment environment contained unrelated software; the release lock includes only the dependency closure needed for the packaged runtime.

Install in an isolated environment, then install this source with `pip install --no-deps -e .`. Do not install the separate nnunetv2 package over this package: both provide the same import namespace. The portable runner sets PYTHONPATH to this snapshot and disables user-site imports.

A compatible NVIDIA driver and adequate GPU memory are needed for training. The native training plan is batch 2 × 4 modalities × 128³. CPU verification uses a smaller 64³ synthetic input to avoid occupying experiment GPUs. Such a smoke test is not a memory benchmark or full E100 reproduction.


The original environment's `blosc2==4.3.3` metadata required `numexpr>=2.14.1`, but `numexpr==2.10.0` was installed. The release pins `numexpr==2.14.1` and supplies the missing declared dependencies `future` and `unittest2` (and their small dependency closure). This is a packaging correction, not a model-code change. The updated runtime is checked against the numerical reference.

Installation validation builds and installs a wheel into a separate virtual environment and verifies imports resolve to the installed wheel, not the original editable source tree. The virtual environment reuses the machine's installed third-party runtime (`--system-site-packages`) and overlays the dependency corrections. Thus the test does **not** claim a completely fresh network installation of all CUDA wheels. The 78-package pinned dependency graph is checked separately for missing or conflicting active requirements.
The released runners use the original trainer recipes. Asynchronous data loading, hardware and dependency changes may introduce numerical differences; bitwise deterministic retraining is not claimed.
