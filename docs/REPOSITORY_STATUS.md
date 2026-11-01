# Repository status

## Import provenance

Imported on 2 November 2026 from the supplied Desktop project and HANDOFF.md. Active source is the fifth-version source archive. No application code was changed. The original guide is retained as PROJECT_GUIDE.md; the handoff is a historical record, and its old Linux paths do not refer to this checkout.

## Evidence and limitations

- The handoff reports 46 passing tests in its previous environment; this is distinct from verification in this checkout.
- Supplied two-camera performance figures are simulations. The second-version simulated validation report records 90% reliability against a 97% target.
- One real broadcast example is provided. The automatic fifth-version result reports NOT OUT, with 94% of sampled paths missing the stumps.
- Version four changed the decision policy: an uncertain finite projection can receive a verdict and a low-confidence flag. This policy change must not be mistaken for an accuracy improvement.
- Single-camera depth, inferred foot location, and physics priors affect the projection. Speed may be undetermined.
- Physical field validation, broader real-video testing, and competition-specific rule approval remain pending.
- The fifth-version archive does not include raw data for every performance claim in its guide. Earlier raw study files are retained in the original project archive.

## Verification of this import

Source files were parsed for Python syntax and preserved archive files were checked against their SHA-256 manifest. Runtime test results will be recorded below if dependencies permit execution.

Runtime verification on this Windows checkout: **44 passed, 2 skipped** in 302 seconds. The two browser-dependent tests skipped because Playwright is not installed. Command: python -m pytest -q -p no:cacheprovider tests. All 41 imported source/template/vendor files match the fifth-version ZIP; all 218 preserved original files match the archive manifest.
