# Repository Guidelines

## Scope
- This guide applies to `ECCV/` only. For the monorepo overview, see `../AGENTS.md`.

## Project Structure & Script Organization
- Script-driven generators (two-stage legacy): `two_stage/mani_longvideo.py`, `two_stage/mani_longvideo_fast.py`.
- Post-processing helpers (two-stage legacy): `two_stage/extract_last_frame_segments.py`, `two_stage/extract_cumulative_last_frame_segments.py`.
- Task specs/notes (two-stage legacy): `tasklist/mani_longvideo_taskslist_final.md` (canonical), `tasklist/mani_longvideo_taskslist.md` (draft).
- Three-stage long-video pipeline: `three_stage/` (separate implementation; see `three_stage/README.md` and `python3 three_stage/pipeline.py -h`).
- Outputs (artifacts): `three_stage/causal_spafa_plan_dataset_long/`, `causal_spafa_plan_dataset*`, `generated_plans_output_*` (JSONL, frames, clips). Treat as outputs, not source.

## Three-Stage Long-Video Pipeline (`three_stage/`)
- Stage 1 (Draft): sample the full video (typically ≤50 frames) → `stage1/draft_plan.json` (must NOT include any keyframe fields).
- Stage 2 (Localize/Cut): predict `{start_frame_index, end_frame_index}` on the same full-video frame pool (1-based; `end_frame_index` is exclusive) → `stage2/step_clips/`.
- Stage 3 (Refine+Keyframes): resample each step clip (≤50 frames/clip) and fill `critical_frames` → `causal_plan_with_keyframes.json` (keyframe `frame_index` is 1-based on the per-clip frame pool).
- Treat `stage1/frame_manifest.json` (full video) and `<step_folder>/frame_manifest.json` (clip) as different index spaces.

## `tasklist/mani_longvideo_taskslist_final.md`
- Canonical spec for converting a single-video item folder into multimodal supervision (Task_01–Task_30).
- Defines item layout, evidence types, and `meta` fields + fallback rules (schema alignment notes included).

## Build, Test, and Development Commands
- Long-video generator (two-stage legacy): `python3 two_stage/mani_longvideo.py`
- Three-stage pipeline: `python3 three_stage/pipeline.py --input-video /abs/path/video.mp4`
- Segment export utilities (two-stage legacy): `python3 two_stage/extract_last_frame_segments.py`, `python3 two_stage/extract_cumulative_last_frame_segments.py`

## Contribution Notes
- Keep scripts configurable (avoid hard-coded machine-specific absolute paths).
- Do not commit large outputs under `causal_spafa_plan_dataset*` / `generated_plans_output_*` unless publishing a dataset.

## Agent-Specific Instructions
- CLI conversations: use Chinese (Simplified) when chatting with agents in the terminal.
