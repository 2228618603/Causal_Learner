# Repository Guidelines

## Scope
- This guide applies to `ECCV/` only. For the monorepo overview, see `../AGENTS.md`.

## Project Structure & Script Organization
- Script-driven generators: `mani_*video*.py` (manipulation), `nav_*` (navigation), `generate_phyplan_api*.py` (API-based generators).
- Post-processing helpers: `extract_*_segments.py`, `post_select_keyframes.py`, `video_duration_classifier*.py`.
- Task specs/notes: `*_spec.md`, `mani_*_tasks_plan*.md`, with `mani_longvideo_tasks_plan_final.md` as the canonical long-video task set.
- Three-stage long-video pipeline: `three_stage/` (separate implementation; spec in `three_stage/THREE_STAGE_PIPELINE.md`).
- Outputs (artifacts): `causal_spafa_plan_dataset*`, `generated_plans_output_*` (JSONL, frames, clips). Treat as outputs, not source.

## Three-Stage Long-Video Pipeline (`three_stage/`)
- Stage 1 (Draft): sample the full video (typically ≤50 frames) → `stage1/draft_plan.json` (must NOT include any keyframe fields).
- Stage 2 (Localize/Cut): predict `{start_frame_index, end_frame_index}` on the same full-video frame pool (1-based; `end_frame_index` is exclusive) → `stage2/step_clips/`.
- Stage 3 (Refine+Keyframes): resample each step clip (≤50 frames/clip) and fill `critical_frames` → `causal_plan_with_keyframes.json` (keyframe `frame_index` is 1-based on the per-clip frame pool).
- Treat `stage1/frame_manifest.json` (full video) and `<step_folder>/frame_manifest.json` (clip) as different index spaces.

## `mani_longvideo_tasks_plan_final.md`
- Canonical spec for converting a single-video item folder into multimodal supervision (Task_01–Task_30).
- Defines item layout, `causal_plan_with_keyframes.json` schema, evidence types, and `meta` fields + fallback rules.

## Build, Test, and Development Commands
- Long-video generator: `python mani_longvideo.py`
- Three-stage pipeline: `python three_stage/pipeline.py --input-video /abs/path/video.mp4`
- Segment export utilities: `python extract_last_frame_segments.py`, `python extract_cumulative_last_frame_segments.py`

## Contribution Notes
- Keep scripts configurable (avoid hard-coded machine-specific absolute paths).
- Do not commit large outputs under `causal_spafa_plan_dataset*` / `generated_plans_output_*` unless publishing a dataset.

## Agent-Specific Instructions
- CLI conversations: use Chinese (Simplified) when chatting with agents in the terminal.
