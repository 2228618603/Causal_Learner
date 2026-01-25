# Repository Guidelines

## Project Structure & Module Organization

- `web_demo_mm.py`: Gradio web demo entrypoint (Hugging Face or vLLM backend).
- `qwen-vl-utils/`: Python package (`src/qwen_vl_utils`) for image/video preprocessing helpers.
- `qwen-vl-finetune/`: Training framework (`qwenvl/`) plus `scripts/` (DeepSpeed configs) and `demo/` sample data.
  - 2D train: `qwen-vl-finetune/qwenvl/train/train_qwen.py` (image/video + text)
  - 3D train: `qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py` (point cloud + text, optional image/video)
  - Point model: `qwen-vl-finetune/qwenvl/train/modeling_qwen3_vl_pointcloud.py`
- `PointLLM/`: Point cloud LLM subproject (installable package + training scripts).
- `evaluation/`: Evaluation tooling (e.g., `evaluation/mmmu/`).
- `cookbooks/`, `docs/`, `docker/`: Notebooks, documentation, and container helpers.
- `checkpoints/`, `outputs/`: Local artifacts; keep these out of commits/PRs.

## Build, Test, and Development Commands

- Install core runtime deps: `pip install "transformers>=4.57.0" torch torchvision gradio`
- Develop on utils package: `pip install -e qwen-vl-utils`
- 2D finetune: `python3 qwen-vl-finetune/qwenvl/train/train_qwen.py --model_name_or_path <ckpt> --dataset_use <name> --output_dir outputs/run`
- 3D finetune: `python3 qwen-vl-finetune/qwenvl/train/train_qwen_pointcloud.py --model <ckpt> --train-file <abs>.jsonl --output-dir outputs/pc`
- Run web demo (HF): `python3 web_demo_mm.py --backend hf -c <checkpoint-or-hf-id>`
- Run web demo (vLLM): `python3 web_demo_mm.py --backend vllm -c <checkpoint-or-hf-id>` (requires `vllm` + `qwen-vl-utils`)
- Docker web demo: `bash docker/docker_web_demo.sh -c /path/to/checkpoint --port 8901`
- MMMU evaluation: `cd evaluation/mmmu && pip install -r requirements.txt` then `python3 run_mmmu.py infer ...`

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep functions small, and add type hints for public APIs where practical.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- `qwen-vl-utils` uses Ruff (`line-length = 119`): `cd qwen-vl-utils && ruff format . && ruff check .`

## Testing Guidelines

- No repo-wide test runner today; validate changes with targeted smoke runs (demo scripts, training/eval entrypoints).
- When adding new code, prefer a small runnable check close to the feature (e.g., under the relevant module).

## Commit & Pull Request Guidelines

- Follow Conventional Commits as used in history: `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`.
- PRs should include: what changed, how to run/verify, and any hardware assumptions (GPU/VRAM).
- Do not commit model weights, API keys, or large generated files.
