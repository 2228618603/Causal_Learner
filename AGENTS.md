# Repository Guidelines

## Project Structure & Module Organization

- `Qwen-PC/`: Qwen3-VL demos, finetuning, and point-cloud work (see `Qwen-PC/AGENTS.md`).
  - `qwen-vl-utils/`: installable Python package (`src/qwen_vl_utils/`).
  - `qwen-vl-finetune/`: training framework and scripts (`qwenvl/train/`).
  - `PointLLM/`: point-cloud LLM subproject.
- `ECCV/`: script-driven dataset generation + long-video pipeline (see `ECCV/AGENTS.md`).
  - `three_stage/`: 3-stage generation pipeline + validators and docs.
  - `causal_spafa_plan_dataset*/`, `generated_plans_output_*`: generated artifacts; treat as outputs, not source.

## Build, Test, and Development Commands

- Qwen3-VL web demo: `python Qwen-PC/web_demo_mm.py --backend hf -c <checkpoint-or-hf-id>`
- Qwen utils (editable install + format/lint): `cd Qwen-PC/qwen-vl-utils && pip install -e . && ruff format . && ruff check .`
- ECCV long-video generator: `python ECCV/mani_longvideo.py`
- ECCV three-stage pipeline: `python ECCV/three_stage/pipeline.py --input-video /abs/path/video.mp4`

## Coding Style & Naming Conventions

- Python: 4-space indentation; keep functions small; add type hints for public APIs when practical.
- Naming: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Formatting: `Qwen-PC/qwen-vl-utils` uses Ruff (`line-length = 119`); keep changes consistent with its config.
- Scripts: avoid hard-coded machine-specific absolute paths; prefer CLI flags and config files.

## Testing Guidelines

- No repo-wide unit test suite today. Validate changes with targeted “smoke” runs of the relevant entrypoints (demo, training, eval, or `ECCV/three_stage/`).
- If you add new functionality, include a minimal runnable check close to the feature (small script or validator).

## Commit & Pull Request Guidelines

- This workspace may be a source snapshot without git history; when working in a git clone, use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`).
- PRs should include: what changed, how to run/verify, and any hardware assumptions (GPU/VRAM).
- Do not commit model weights, API keys, or large generated outputs under dataset/output folders.

## Agent-Specific Notes

- Follow the most specific `AGENTS.md` in the directory you are editing (nested guides override this file).
