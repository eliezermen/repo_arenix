# Arenix

Persistent sports prediction research platform.

## Architecture
- Supabase: source of truth, historical data, features, model registry, metrics, checkpoints.
- GitHub Actions: Python training runtime.
- 2026 NFL data remains live/out-of-sample holdout.

## NFL moneyline protocol
- Train: 2021-2023
- Validation/tuning: 2024
- Refit: 2021-2024
- Untouched test: 2025
- Live holdout: 2026

Models are challengers until probabilistic metrics, calibration and later EV/CLV/ROI justify promotion. Closing market probability is benchmark-only and is not a training feature.
