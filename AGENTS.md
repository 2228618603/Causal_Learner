# Repository Guidelines

## Project Structure & Module Organization

- `Qwen-PC/`: Qwen3-VL utilities, finetuning, and point-cloud work (see `Qwen-PC/AGENTS.md`).
  - `qwen-vl-utils/`: installable Python package (`src/qwen_vl_utils/`).
  - `qwen-vl-finetune/`: training framework and scripts (`qwenvl/train/`).
  - `PointLLM/`: point-cloud LLM subproject.
  - `Qwen3-VL/`: upstream/reference snapshot for baseline alignment.
- `ECCV/`: script-driven dataset generation + long-video pipeline (see `ECCV/AGENTS.md`).
  - `three_stage/`: 3-stage generation pipeline + validators and docs.
  - `two_stage/`: legacy 2-stage generators + post-processing helpers.
  - `three_stage/causal_spafa_plan_dataset_long/`, `causal_spafa_plan_dataset*/`, `generated_plans_output_*`: generated artifacts; treat as outputs, not source.
- `e2e-data/`: end-to-end fixtures and generated artifacts; treat as outputs, not source.

## Coding Style & Naming Conventions

- Python: 4-space indentation; keep functions small; add type hints for public APIs when practical.
- Naming: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Formatting: `Qwen-PC/qwen-vl-utils` uses Ruff (`line-length = 119`); keep changes consistent with its config.
- Scripts: avoid hard-coded machine-specific absolute paths; prefer CLI flags and config files.

## Testing Guidelines

- No repo-wide unit test suite today. Validate changes with small, targeted runs of the relevant scripts/validators in the area you changed.
- If you add new functionality, include a minimal runnable check close to the feature (small script or validator).

## Agent-Specific Notes

- Follow the most specific `AGENTS.md` in the directory you are editing (nested guides override this file).
