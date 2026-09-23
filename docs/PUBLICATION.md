# Publishing this snapshot

Upload the versioned release archive or its extracted contents to the chosen anonymous hosting service. Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, the historical checkpoint provenance and verification reports with the code. Do not add local logs, shell histories, original checkpoint payloads, data symlinks, private manifests or Git history from the experiment workspace.

The full release archive is self-contained for code and historical weights. This GitHub mirror is code-only and omits the large checkpoint; see the README for the companion-archive procedure. Dataset images must be obtained separately. The wheel alone is not the complete release: it does not carry the dataset metadata, scripts, results or weights stored alongside the source.

Verify the archive SHA256 after upload and check the link while logged out before adding it to the manuscript. Use a fixed version or snapshot link so subsequent edits do not silently change the reviewed artifact. Any later code or result change should receive a new version and updated checksum; do not replace the v1.0.0 archive in place.

Local file integrity can be checked from the extracted root with `sha256sum -c SHA256SUMS` on Linux. No hosting account or public link is configured in this package.
