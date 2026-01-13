#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from common import (
    ApiConfig,
    SamplingConfig,
    VIDEO_EXTS,
    build_api_content,
    build_retry_prefix,
    call_chat_completion,
    collect_videos,
    default_output_root,
    extract_json_from_response,
    initialize_api_client,
    normalize_stage3_step_output,
    now_utc_iso,
    read_json,
    sample_video_to_frames,
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


_CANONICAL_CAUSAL_CHAIN_KEYS = {
    "agent",
    "action",
    "patient",
    "causal_effect_on_patient",
    "causal_effect_on_environment",
}
_CANONICAL_HOTSPOT_KEYS = {"description", "affordance_type", "mechanism"}


def _upgrade_step_schema_inplace(step: Dict[str, Any]) -> bool:
    """Upgrade legacy keys in-place to the canonical ECCV schema.

    - affordance_hotspot: rename legacy `causal_role` -> `mechanism` (and drop extra keys)
    - causal_chain: drop any extra keys beyond the canonical 5 fields
    """
    changed = False
    cfs = step.get("critical_frames")
    if not isinstance(cfs, list):
        return False
    for cf in cfs:
        if not isinstance(cf, dict):
            continue
        cc = cf.get("causal_chain")
        if isinstance(cc, dict):
            extra_cc = sorted(set(cc.keys()) - _CANONICAL_CAUSAL_CHAIN_KEYS)
            for k in extra_cc:
                cc.pop(k, None)
                changed = True
        hs = cf.get("affordance_hotspot")
        if isinstance(hs, dict):
            if "mechanism" not in hs:
                legacy = hs.get("causal_role")
                if isinstance(legacy, str) and legacy.strip():
                    hs["mechanism"] = legacy
                    changed = True
            if "causal_role" in hs:
                hs.pop("causal_role", None)
                changed = True
            extra_hs = sorted(set(hs.keys()) - _CANONICAL_HOTSPOT_KEYS)
            for k in extra_hs:
                hs.pop(k, None)
                changed = True
    return changed


def _is_step_schema_canonical(step: Dict[str, Any]) -> bool:
    """Return True if a step strictly matches the canonical keys expected downstream."""
    cfs = step.get("critical_frames")
    if not isinstance(cfs, list) or not cfs:
        return False
    for cf in cfs:
        if not isinstance(cf, dict):
            return False
        cc = cf.get("causal_chain")
        if not isinstance(cc, dict):
            return False
        if set(cc.keys()) - _CANONICAL_CAUSAL_CHAIN_KEYS:
            return False
        hs = cf.get("affordance_hotspot")
        if not isinstance(hs, dict):
            return False
        if "causal_role" in hs:
            return False
        mech = hs.get("mechanism")
        if not (isinstance(mech, str) and mech.strip()):
            return False
        if set(hs.keys()) - _CANONICAL_HOTSPOT_KEYS:
            return False
    return True


def _validate_step_final_dict(
    step: Any,
    step_id: int,
    step_goal: str,
    *,
    num_frames: int,
    require_keyframe_paths: bool,
) -> bool:
    if not isinstance(step, dict):
        return False
    try:
        if int(step.get("step_id")) != int(step_id):
            return False
    except Exception:
        return False
    if str(step.get("step_goal", "")).strip() != str(step_goal).strip():
        return False

    try:
        ok = (
            isinstance(step.get("critical_frames"), list)
            and 1 <= len(step["critical_frames"]) <= 2
            and isinstance(step.get("spatial_postconditions_detail"), list)
            and len(step.get("spatial_postconditions_detail") or []) > 0
            and isinstance(step.get("affordance_postconditions_detail"), list)
            and len(step.get("affordance_postconditions_detail") or []) > 0
            and isinstance(step.get("predicted_next_actions"), list)
            and 2 <= len(step.get("predicted_next_actions") or []) <= 4
        )
    except Exception:
        return False
    if not ok:
        return False

    for cf in step.get("critical_frames") or []:
        if not isinstance(cf, dict):
            return False
        try:
            fi = int(cf.get("frame_index"))
        except Exception:
            return False
        if fi < 1 or fi > int(num_frames):
            return False

        if require_keyframe_paths:
            kfp = cf.get("keyframe_image_path")
            if not (isinstance(kfp, str) and os.path.exists(kfp)):
                return False

        cc = cf.get("causal_chain")
        if not isinstance(cc, dict):
            return False
        required_cc = [
            "agent",
            "action",
            "patient",
            "causal_effect_on_patient",
            "causal_effect_on_environment",
        ]
        if any(not str(cc.get(k, "")).strip() for k in required_cc):
            return False

        hs = cf.get("affordance_hotspot")
        if not isinstance(hs, dict):
            return False
        mechanism = str(hs.get("mechanism") or hs.get("causal_role") or "").strip()
        if not str(hs.get("description", "")).strip():
            return False
        if not str(hs.get("affordance_type", "")).strip():
            return False
        if not mechanism:
            return False

    return True


def _can_resume_stage3_final_plan(
    final_path: str,
    draft_path: str,
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

    # Validate per-step structure and ensure keyframe images exist.
    for sid, goal in got_by_id.items():
        num_frames = int(max_frames_fallback)
        step_folder = os.path.join(video_out, f"{sid:02d}_{sanitize_filename(goal)}")
        manifest_path = os.path.join(step_folder, "frame_manifest.json")
        if os.path.exists(manifest_path):
            try:
                num_frames = int(read_json(manifest_path).get("num_frames", num_frames))
            except Exception:
                return False
        step_obj = step_by_id.get(sid)
        if not _validate_step_final_dict(
            step_obj,
            sid,
            goal,
            num_frames=num_frames,
            require_keyframe_paths=True,
        ):
            return False
        if not _is_step_schema_canonical(step_obj):
            return False

    return True


def _load_valid_cached_step_final(
    step_out_path: str,
    manifest_path: str,
    step_id: int,
    step_goal: str,
    *,
    max_frames_fallback: int,
) -> Optional[Dict[str, Any]]:
    """Load a cached step_final.json if it is consistent and its keyframe images exist."""
    if not os.path.exists(step_out_path):
        return None
    try:
        num_frames = int(max_frames_fallback)
        if os.path.exists(manifest_path):
            manifest = read_json(manifest_path)
            num_frames = int(manifest.get("num_frames", num_frames))
        existing = read_json(step_out_path)
    except Exception:
        return None
    if not _validate_step_final_dict(
        existing,
        step_id,
        step_goal,
        num_frames=num_frames,
        require_keyframe_paths=True,
    ):
        return None
    changed = _upgrade_step_schema_inplace(existing)
    if changed:
        write_json(step_out_path, existing)
    return existing


def run_stage3_for_video(
    video_path: str,
    output_root: str,
    api_cfg: ApiConfig,
    sampling_cfg: SamplingConfig,
    overwrite: bool,
    max_retries: int,
) -> str:
    vid = video_id_from_path(video_path)
    video_out = os.path.join(output_root, vid)
    stage1_dir = os.path.join(video_out, "stage1")
    stage2_dir = os.path.join(video_out, "stage2")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    final_path = os.path.join(video_out, "causal_plan_with_keyframes.json")
    run_summary_path = os.path.join(video_out, "run_summary.json")

    if not overwrite and _can_resume_stage3_final_plan(
        final_path,
        draft_path,
        video_out,
        max_frames_fallback=sampling_cfg.max_frames,
    ):
        return video_out

    if not os.path.exists(draft_path):
        raise FileNotFoundError(f"Stage 1 draft not found: {draft_path}")
    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Stage 2 segments not found: {segments_path}")

    draft = read_json(draft_path)
    high_level_goal = str(draft.get("high_level_goal", "")).strip()
    draft_steps = draft.get("steps", [])
    if not isinstance(draft_steps, list) or not draft_steps:
        raise RuntimeError("Draft plan has no steps.")
    if not (4 <= len(draft_steps) <= 9):
        raise RuntimeError(
            f"Draft step count must be within [4, 9] for the three-stage pipeline (got {len(draft_steps)}). "
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

    for step in ordered_steps:
        sid = int(step.get("step_id", 0))
        goal = str(step.get("step_goal", "")).strip()
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

        # Step-level resume: if a valid step_final.json exists (and its keyframe images exist),
        # reuse it to avoid re-calling the model.
        if not overwrite:
            cached = _load_valid_cached_step_final(
                step_out_path,
                manifest_path,
                sid,
                goal,
                max_frames_fallback=sampling_cfg.max_frames,
            )
            if cached is not None:
                final_steps.append(cached)
                continue

        sampled_frames, _dims = sample_video_to_frames(clip_path, sampling_cfg)
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
        frames_content = build_api_content(sampled_frames, api_cfg.embed_index_on_api_images, include_manifest=False)
        base_user_content = [{"type": "text", "text": base_prompt}] + frames_content
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_ANALYST}

        last_content = ""
        last_errors: List[str] = []
        normalized_step: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_retries + 1):
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

        # Save keyframe images and fill keyframe_image_path
        chosen = [int(cf["frame_index"]) for cf in normalized_step["critical_frames"]]
        keyframe_paths = save_keyframe_images_from_manifest(manifest_path, chosen, output_dir=step_folder)
        for cf in normalized_step["critical_frames"]:
            idx1 = int(cf["frame_index"])
            cf["keyframe_image_path"] = keyframe_paths.get(idx1)

        write_json(step_out_path, normalized_step)
        write_json(
            os.path.join(step_folder, "step_meta.json"),
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

        final_steps.append(normalized_step)

    final_plan = {"high_level_goal": high_level_goal, "steps": final_steps}
    write_json(final_path, final_plan)

    update_run_summary(
        run_summary_path,
        {
            "updated_at_utc": now_utc_iso(),
            "stage3": {
                "status": "completed",
                "generated_at_utc": now_utc_iso(),
                "final_plan_path": os.path.relpath(final_path, video_out),
                "frame_index_note": "In this three-stage pipeline, critical_frames[*].frame_index is 1-based on EACH STEP CLIP's 50-frame pool; see each step folder's frame_manifest.json.",
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

    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: refine per-step using clips and generate keyframes.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root under ECCV/three_stage/...")

    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", "http://model.mify.ai.srv/v1"))
    parser.add_argument("--provider", default=os.environ.get("MODEL_PROVIDER_ID", "vertex_ai"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemini-3-pro-preview"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "30000")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.2")))
    parser.add_argument("--api-call-retries", type=int, default=int(os.environ.get("API_CALL_RETRIES", "3")))
    parser.add_argument(
        "--api-call-retry-backoff-sec",
        type=float,
        default=float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0")),
    )
    parser.add_argument("--no-embed-index", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_cfg = ApiConfig(
        api_key=args.api_key,
        api_base_url=args.api_base,
        model_provider_id=args.provider,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_call_retries=args.api_call_retries,
        api_call_retry_backoff_sec=args.api_call_retry_backoff_sec,
        embed_index_on_api_images=not args.no_embed_index,
        verbose=args.verbose,
    )
    sampling_cfg = SamplingConfig(max_frames=args.max_frames, jpeg_quality=args.jpeg_quality)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)

    if not videos:
        raise SystemExit("No videos found.")

    for vp in videos:
        run_stage3_for_video(vp, args.output_root, api_cfg, sampling_cfg, overwrite=args.overwrite, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
