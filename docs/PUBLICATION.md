# Publishing the anonymous code release

This v1.1.0 snapshot contains code, cohort metadata, reference results and verification records for BraTS, LiTS and DRIVE. It contains no dataset images or binary pretrained weights. Preserve LICENSE, THIRD_PARTY_NOTICES.md and the historical checkpoint provenance when redistributing it.

The public repository is https://github.com/anonymous-vd-refine/VD-Refine and the project page is https://anonymous-vd-refine.github.io/VD-Refine/ . Do not add private logs, shell histories, data symlinks, original checkpoint payloads or experiment-workspace Git history.

Use a fixed commit or versioned archive when citing the code in a submission. Verify the final uploaded files against SHA256SUMS; the checksum list describes this complete release snapshot, including its website. The wheel alone does not include the dataset metadata, scripts and reference tables stored alongside the source.

Run `sha256sum -c SHA256SUMS` from the extracted repository root on Linux to verify integrity. Dataset access terms and pretrained-weight availability are separate from the code release.
