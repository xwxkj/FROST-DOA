# Contributing

1. Create a feature branch and keep the public method name `FROST-DOA` unchanged.
2. Do not modify files in `src/frost_doa/_frozen_core/models/` without retraining and documenting the new calibration.
3. Run `PYTHONPATH=src pytest -q` before opening a pull request.
4. Run the smoke benchmark before a full rerun.
5. Any result-table change must be generated from terminal CSV files and must pass `scripts/07_audit_manuscript_results.py`.
6. Third-party implementations of DA-MUSIC and SubspaceNet are not redistributed; use their upstream repositories and licenses.
