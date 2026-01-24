#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from common import (
    add_api_cli_args,
    add_sampling_cli_args,
    VIDEO_EXTS,
    api_config_from_args,
    build_api_content,
    build_retry_prefix,
    call_chat_completion,
    collect_videos,
    default_output_root,
    ensure_video_out_dir_safe,
    extract_json_from_response,
    format_duration,
    initialize_api_client,
    guard_schema_fingerprint,
    logger,
    normalize_stage3_step_output,
    now_utc_iso,
    read_json,
    sample_video_to_frames,
    sampling_config_from_args,
    sanitize_filename,
    save_keyframe_images_from_manifest,
    save_sampled_frames_jpegs,
    update_run_summary,
    video_id_from_path,
    write_frame_manifest,
    write_json,
    write_text,
)
from prompts import SYSTEM_PROMPT_ANALYST, build_stage3_user_prompt

if TYPE_CHECKING:
    from common import ApiConfig, SamplingConfig


def _step_meta_matches_segment(
    step_meta_path: str,
    expected_clip_abs: str,
    expected_clip_start_sec: float,
    expected_clip_end_sec: float,
    *,
    tol_sec: float = 1e-3,
) -> bool:
    if not os.path.exists(step_meta_path):
        return False
    try:
        meta = read_json(step_meta_path)
    except Exception:
        return False

    rel = meta.get("clip_path")
    if not isinstance(rel, str) or not rel:
        return False
    step_folder = os.path.dirname(step_meta_path)
    meta_clip_abs = os.path.abspath(os.path.join(step_folder, rel))
    if os.path.abspath(expected_clip_abs) != meta_clip_abs:
        return False

    try:
        s = float(meta.get("clip_start_sec"))
        e = float(meta.get("clip_end_sec"))
    except Exception:
        return False
    if abs(s - float(expected_clip_start_sec)) > float(tol_sec):
        return False
    if abs(e - float(expected_clip_end_sec)) > float(tol_sec):
        return False
    return True


def _load_manifest_frame_timestamps(manifest_path: str) -> Optional[List[float]]:
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return None
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return None
    out: List[float] = []
    for fr in frames:
        if not isinstance(fr, dict):
            out.append(0.0)
            continue
        try:
            out.append(float(fr.get("timestamp_sec", 0.0)))
        except Exception:
            out.append(0.0)
    return out


def _expected_keyframe_image_paths(
    manifest_path: str,
    frame_indices_1based: List[int],
    step_folder: str,
) -> Dict[int, str]:
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return {}

    by_idx: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("frames", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx1 = int(entry.get("frame_index_1based"))
        except Exception:
            continue
        by_idx[idx1] = entry

    out: Dict[int, str] = {}
    for idx1 in frame_indices_1based:
        entry = by_idx.get(int(idx1))
        if not entry:
            continue
        try:
            ts = float(entry.get("timestamp_sec", 0.0))
        except Exception:
            ts = 0.0
        name = f"frame_{int(idx1):03d}_ts_{ts:.2f}s.jpg"
        out[int(idx1)] = os.path.abspath(os.path.join(step_folder, name))
    return out


def _can_resume_stage3_final_plan(
    final_path: str,
    draft_path: str,
    segments_path: str,
    video_out: str,
    *,
    max_frames_fallback: int,
) -> bool:
    """Return True if the merged final plan exists, is consistent, and looks complete."""
    if not os.path.exists(final_path):
        return False

    try:
        data = read_json(final_path)
    except Exception:
        return False

    high_level_goal = data.get("high_level_goal")
    steps = data.get("steps", [])
    if not (isinstance(high_level_goal, str) and high_level_goal.strip() and isinstance(steps, list) and steps):
        return False

    # If Stage-1 draft exists, ensure the final plan is aligned with it (step_id/step_goal match).
    expected_by_id: Optional[Dict[int, str]] = None
    if os.path.exists(draft_path):
        try:
            draft = read_json(draft_path)
        except Exception:
            return False
        expected_by_id = {}
        for st in draft.get("steps", []):
            if not isinstance(st, dict) or st.get("step_id") is None:
                continue
            try:
                sid = int(st.get("step_id"))
            except Exception:
                continue
            goal = str(st.get("step_goal", "")).strip()
            if sid > 0 and goal:
                expected_by_id[sid] = goal
        if not expected_by_id:
            return False

    got_by_id: Dict[int, str] = {}
    step_by_id: Dict[int, Dict[str, Any]] = {}
    for st in steps:
        if not isinstance(st, dict) or st.get("step_id") is None:
            return False
        try:
            sid = int(st.get("step_id"))
        except Exception:
            return False
        goal = str(st.get("step_goal", "")).strip()
        if sid <= 0 or not goal:
            return False
        if sid in got_by_id:
            return False
        got_by_id[sid] = goal
        step_by_id[sid] = st

    if expected_by_id is not None and expected_by_id != got_by_id:
        return False

    # Ensure Stage-2 segments still match the cached Stage-3 outputs to avoid
    # producing a final plan that is inconsistent with the current clips.
    if not os.path.exists(segments_path):
        return False
    try:
        seg_data = read_json(segments_path)
    except Exception:
        return False
    segs = seg_data.get("segments", [])
    if not isinstance(segs, list) or not segs:
        return False
    stage2_dir = os.path.join(video_out, "stage2")
    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segs:
        if not isinstance(seg, dict) or seg.get("step_id") is None:
            continue
        try:
            sid = int(seg.get("step_id"))
        except Exception:
            continue
        seg_by_id[sid] = seg
    if not seg_by_id:
        return False

    # Validate per-step structure and ensure keyframe images exist.

    for sid, goal in got_by_id.items():
        if sid not in seg_by_id:
            return False
        seg = seg_by_id[sid]
        try:
            clip_start_sec = float(seg.get("start_sec"))
            clip_end_sec = float(seg.get("end_sec"))
        except Exception:
            return False
        clip_rel = seg.get("clip_relpath")
        if not isinstance(clip_rel, str) or not clip_rel:
            return False
        clip_abs = os.path.join(stage2_dir, clip_rel)
        if not os.path.exists(clip_abs):
            return False

        step_folder = os.path.join(video_out, f"{sid:02d}_{sanitize_filename(goal)}")
        step_meta_path = os.path.join(step_folder, "step_meta.json")
        if not _step_meta_matches_segment(step_meta_path, clip_abs, clip_start_sec, clip_end_sec):
            return False

        manifest_path = os.path.join(step_folder, "frame_manifest.json")
        if not os.path.exists(manifest_path):
            return False
        try:
            manifest = read_json(manifest_path)
            num_frames = int(manifest.get("num_frames", 0) or 0)
            if num_frames <= 0:
                num_frames = len(manifest.get("frames", []) or []) or int(max_frames_fallback)
        except Exception:
            return False
        frame_timestamps = _load_manifest_frame_timestamps(manifest_path)

        step_out_path = os.path.join(step_folder, "step_final.json")
        if not os.path.exists(step_out_path):
            return False
        try:
            step_file = read_json(step_out_path)
        except Exception:
            return False

        normalized_file, errs_file = normalize_stage3_step_output(
            step_file, sid, goal, num_frames, frame_timestamps=frame_timestamps
        )
        if normalized_file is None:
            return False

        step_obj = step_by_id.get(sid)
        if not isinstance(step_obj, dict):
            return False
        normalized_final, errs_final = normalize_stage3_step_output(
            step_obj, sid, goal, num_frames, frame_timestamps=frame_timestamps
        )
        if normalized_final is None:
            return False
        if normalized_final != normalized_file:
            return False

        chosen = [int(cf["frame_index"]) for cf in normalized_file.get("critical_frames", [])]
        expected_paths = _expected_keyframe_image_paths(manifest_path, chosen, step_folder)
        if len(expected_paths) != len(chosen):
            return False
        if any(not os.path.exists(p) for p in expected_paths.values()):
            return False

    return True


def _load_valid_cached_step_final(
    step_out_path: str,
    manifest_path: str,
    step_meta_path: str,
    step_id: int,
    step_goal: str,
    expected_clip_abs: str,
    expected_clip_start_sec: float,
    expected_clip_end_sec: float,
    *,
    max_frames_fallback: int,
) -> Optional[Dict[str, Any]]:
    """Load a cached step_final.json if it is consistent and its keyframe images exist."""
    if not os.path.exists(step_out_path):
        return None
    if not _step_meta_matches_segment(step_meta_path, expected_clip_abs, expected_clip_start_sec, expected_clip_end_sec):
        return None
    try:
        if not os.path.exists(manifest_path):
            return None
        manifest = read_json(manifest_path)
        num_frames = int(manifest.get("num_frames", 0) or 0)
        if num_frames <= 0:
            num_frames = len(manifest.get("frames", []) or []) or int(max_frames_fallback)
        frame_timestamps = _load_manifest_frame_timestamps(manifest_path)
        existing = read_json(step_out_path)
    except Exception:
        return None
    if not isinstance(existing, dict):
        return None

    normalized, errs = normalize_stage3_step_output(
        existing, step_id, step_goal, num_frames, frame_timestamps=frame_timestamps
    )
    if normalized is None:
        return None

    step_folder = os.path.dirname(step_out_path)
    chosen = [int(cf["frame_index"]) for cf in normalized.get("critical_frames", [])]
    expected_paths = _expected_keyframe_image_paths(manifest_path, chosen, step_folder)
    if len(expected_paths) != len(chosen):
        return None
    if any(not os.path.exists(p) for p in expected_paths.values()):
        return None

    if existing != normalized:
        write_json(step_out_path, normalized)
    return normalized


def run_stage3_for_video(
    video_path: str,
    output_root: str,
    api_cfg: ApiConfig,
    sampling_cfg: SamplingConfig,
    overwrite: bool,
    max_retries: int,
    *,
    allow_legacy_resume: bool = False,
) -> str:
    t_start = time.perf_counter()
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    ensure_video_out_dir_safe(video_out, video_path)
    stage1_dir = os.path.join(video_out, "stage1")
    stage2_dir = os.path.join(video_out, "stage2")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    final_path = os.path.join(video_out, "causal_plan_with_keyframes.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    if not os.path.exists(draft_path):
        raise FileNotFoundError(f"Stage 1 draft not found: {draft_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Stage 2 segments not found: {segments_path}")
    will_resume = not overwrite and _can_resume_stage3_final_plan(
        final_path,
        draft_path,
        segments_path,
        video_out,
        max_frames_fallback=sampling_cfg.max_frames,
    )
    schema_fp = guard_schema_fingerprint(
        run_summary_path,
        video_out,
        stage="Stage 3",
        overwrite=overwrite,
        allow_legacy_resume=allow_legacy_resume,
        will_resume=will_resume,
    )
    if will_resume:
        logger.info(f"[stage3] video_id={vid} resume: {os.path.relpath(video_out, output_root)}")
        return video_out

    logger.info(
        f"[stage3] video_id={vid} start: overwrite={bool(overwrite)} max_frames={int(sampling_cfg.max_frames)} "
        f"src={os.path.abspath(video_path)}"
    )

    draft = read_json(draft_path)
    high_level_goal = str(draft.get("high_level_goal", "")).strip()
    draft_steps = draft.get("steps", [])
    if not isinstance(draft_steps, list) or not draft_steps:
        raise RuntimeError("Draft plan has no steps.")
    if not (3 <= len(draft_steps) <= 8):
        raise RuntimeError(
            f"Draft step count must be within [3, 8] for the three-stage pipeline (got {len(draft_steps)}). "
            "Re-run Stage 1 with a better prompt (or use --overwrite)."
        )

    segs = read_json(segments_path).get("segments", [])
    if not isinstance(segs, list) or not segs:
        raise RuntimeError("Stage 2 segments missing/empty.")
    seg_by_id = {int(s.get("step_id")): s for s in segs if isinstance(s, dict) and s.get("step_id") is not None}

    client = initialize_api_client(api_cfg)
    if not client:
        raise SystemExit("Failed to initialize API client.")

    final_steps: List[Dict[str, Any]] = []
    ordered_steps = sorted([s for s in draft_steps if isinstance(s, dict)], key=lambda x: int(x.get("step_id", 0)))
    total_steps = len(ordered_steps)
    logger.info(f"[stage3] video_id={vid} steps={total_steps} (sampling per-clip)")
    outline_lines: List[str] = []
    for st in ordered_steps:
        try:
            sid = int(st.get("step_id", 0))
        except Exception:
            continue
        goal = str(st.get("step_goal", "")).strip()
        if sid > 0 and goal:
            outline_lines.append(f"- Step {sid}: {goal}")
    draft_plan_outline = "\n".join(outline_lines)

    for i_step, step in enumerate(ordered_steps, start=1):
        sid = int(step.get("step_id", 0))
        goal = str(step.get("step_goal", "")).strip()
        goal_short = " ".join(goal.split())
        if len(goal_short) > 80:
            goal_short = goal_short[:77] + "..."
        if sid not in seg_by_id:
            raise RuntimeError(f"Missing Stage 2 segment for step_id={sid}")
        seg = seg_by_id[sid]
        try:
            clip_start_sec = float(seg.get("start_sec", 0.0))
            clip_end_sec = float(seg.get("end_sec", 0.0))
        except Exception:
            clip_start_sec = 0.0
            clip_end_sec = 0.0
        clip_rel = seg.get("clip_relpath")
        if not clip_rel:
            raise RuntimeError(f"Missing clip_relpath for step_id={sid}")
        clip_path = os.path.join(stage2_dir, clip_rel)
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"Clip not found: {clip_path}")

        step_folder = os.path.join(video_out, f"{sid:02d}_{sanitize_filename(goal)}")
        os.makedirs(step_folder, exist_ok=True)
        sampled_frames_dir = os.path.join(step_folder, "sampled_frames")
        manifest_path = os.path.join(step_folder, "frame_manifest.json")
        raw_path = os.path.join(step_folder, "stage3_raw_response.txt")
        step_out_path = os.path.join(step_folder, "step_final.json")
        step_meta_path = os.path.join(step_folder, "step_meta.json")

        # Step-level resume: if a valid step_final.json exists (and its keyframe images exist),
        # reuse it to avoid re-calling the model.
        if not overwrite:
            cached = _load_valid_cached_step_final(
                step_out_path,
                manifest_path,
                step_meta_path,
                sid,
                goal,
                expected_clip_abs=clip_path,
                expected_clip_start_sec=clip_start_sec,
                expected_clip_end_sec=clip_end_sec,
                max_frames_fallback=sampling_cfg.max_frames,
            )
            if cached is not None:
                logger.info(
                    f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
                    f"reuse cached step_final goal='{goal_short}'"
                )
                final_steps.append(cached)
                continue

        step_started = time.perf_counter()
        logger.info(
            f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid}: "
            f"goal='{goal_short}' clip={os.path.relpath(clip_path, video_out)} ({clip_start_sec:.2f}s..{clip_end_sec:.2f}s)"
        )
        sampled_frames, _ = sample_video_to_frames(clip_path, sampling_cfg)
        # Convert clip-local timestamps to original-video timestamps so that downstream tools
        # (e.g., extract_last_frame_segments.py) can cut segments on the source video timeline.
        for fr in sampled_frames:
            try:
                fr["timestamp_sec"] = float(fr.get("timestamp_sec", 0.0)) + clip_start_sec
            except Exception:
                fr["timestamp_sec"] = clip_start_sec
        frame_timestamps = [float(fr.get("timestamp_sec", 0.0)) for fr in sampled_frames]
        save_sampled_frames_jpegs(sampled_frames, sampled_frames_dir)
        write_frame_manifest(sampled_frames, sampled_frames_dir, manifest_path)

        draft_step_json = json.dumps(step, ensure_ascii=False)
        base_prompt = build_stage3_user_prompt(high_level_goal, draft_plan_outline, draft_step_json, len(sampled_frames))
        write_text(os.path.join(step_folder, "stage3_system_prompt.txt"), SYSTEM_PROMPT_ANALYST)
        write_text(os.path.join(step_folder, "stage3_user_prompt.txt"), base_prompt)
        frames_content = build_api_content(
            sampled_frames,
            api_cfg.embed_index_on_api_images,
            include_manifest=False,
            # Always include explicit "Frame N" text labels to reduce index mistakes caused by
            # OCR/counting errors on long frame pools.
            # (Stage 3 forbids mentioning frame numbers in free-form text fields; violations are caught by strict
            # normalization and retried.)
            include_frame_labels=True,
        )
        base_user_content = [{"type": "text", "text": base_prompt}] + frames_content
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_ANALYST}

        last_content = ""
        last_errors: List[str] = []
        normalized_step: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
            logger.info(
                f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid} model_call attempt={attempt}/{max_retries}"
            )
            if attempt == 1:
                user_content = base_user_content
            else:
                prefix = build_retry_prefix(last_errors, last_content)
                user_content = [{"type": "text", "text": prefix + base_prompt}] + frames_content
            messages = [system_msg, {"role": "user", "content": user_content}]
            content = call_chat_completion(client, api_cfg, messages, max_tokens=api_cfg.max_tokens)
            last_content = content
            try:
                clean = extract_json_from_response(content)
                obj = json.loads(clean)
            except Exception as e:
                last_errors = [f"JSON parse error: {e}"]
                continue

            normalized_step, errs = normalize_stage3_step_output(
                obj, sid, goal, len(sampled_frames), frame_timestamps=frame_timestamps
            )
            if normalized_step is not None:
                break
            last_errors = errs

        if normalized_step is None:
            write_text(raw_path, last_content)
            raise RuntimeError(f"Stage 3 failed for step_id={sid}: " + " | ".join(last_errors[:10]))

        write_text(raw_path, last_content)

        # Save keyframe images (do not write any filesystem paths into JSON; see prompts.py).
        chosen = [int(cf["frame_index"]) for cf in normalized_step["critical_frames"]]
        save_keyframe_images_from_manifest(manifest_path, chosen, output_dir=step_folder)

        write_json(step_out_path, normalized_step)
        write_json(
            step_meta_path,
            {
                "step_id": sid,
                "step_goal": goal,
                "clip_path": os.path.relpath(clip_path, step_folder),
                "clip_start_sec": clip_start_sec,
                "clip_end_sec": clip_end_sec,
                "num_frames": len(sampled_frames),
                "generated_at_utc": now_utc_iso(),
                "manifest_path": os.path.relpath(manifest_path, step_folder),
            },
        )

        logger.info(
            f"[stage3] video_id={vid} step {i_step}/{total_steps} step_id={sid} done in "
            f"{format_duration(time.perf_counter() - step_started)}"
        )
        final_steps.append(normalized_step)

    final_plan = {"high_level_goal": high_level_goal, "steps": final_steps}
    write_json(final_path, final_plan)

    update_run_summary(
        run_summary_path,
        {
            "source_video": os.path.abspath(video_path),
            "video_id": vid,
            "output_root": os.path.abspath(output_root),
            "schema_fingerprint": schema_fp,
            "updated_at_utc": now_utc_iso(),
            "stage3": {
                "status": "completed",
                "generated_at_utc": now_utc_iso(),
                "final_plan_path": os.path.relpath(final_path, video_out),
                "frame_index_note": (
                    "In this three-stage pipeline, critical_frames[*].frame_index is 1-based on EACH STEP CLIP's "
                    f"{int(sampling_cfg.max_frames)}-frame pool; see each step folder's frame_manifest.json."
                ),
                "api_config": {
                    "api_base_url": api_cfg.api_base_url,
                    "model_provider_id": api_cfg.model_provider_id,
                    "model_name": api_cfg.model_name,
                    "max_tokens": int(api_cfg.max_tokens),
                    "temperature": float(getattr(api_cfg, "temperature", 0.2)),
                    "api_call_retries": int(getattr(api_cfg, "api_call_retries", 1)),
                    "api_call_retry_backoff_sec": float(getattr(api_cfg, "api_call_retry_backoff_sec", 1.0)),
                },
            },
        },
    )

    logger.info(
        f"[stage3] video_id={vid} completed: steps={len(final_steps)} elapsed={format_duration(time.perf_counter() - t_start)}"
    )
    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: refine per-step using clips and generate keyframes.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root under ECCV/three_stage/...")

    add_api_cli_args(parser, include_no_embed_index=True)

    add_sampling_cli_args(parser, default_max_frames=50, default_jpeg_quality=95)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (legacy outputs).",
    )
    args = parser.parse_args()

    api_cfg = api_config_from_args(args)
    sampling_cfg = sampling_config_from_args(args)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)

    if not videos:
        raise SystemExit("No videos found.")

    for vp in videos:
        run_stage3_for_video(
            vp,
            args.output_root,
            api_cfg,
            sampling_cfg,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            allow_legacy_resume=args.allow_legacy_resume,
        )


if __name__ == "__main__":
    main()
