#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from common import normalize_stage3_step_output, read_json, sanitize_filename, validate_stage2_localization, write_json
from stage1_generate_draft import _draft_hard_errors, _stage1_raw_schema_errors

def _keyframe_filename(frame_index_1based: int, timestamp_sec: float) -> str:
    return f"frame_{int(frame_index_1based):03d}_ts_{float(timestamp_sec):.2f}s.jpg"


def _looks_like_two_stage_output(video_out: str) -> bool:
    if not os.path.isdir(video_out):
        return False
    if os.path.isdir(os.path.join(video_out, "stage1")) or os.path.isdir(os.path.join(video_out, "stage2")):
        return False
    if not os.path.exists(os.path.join(video_out, "causal_plan_with_keyframes.json")):
        return False
    # Heuristic: has step folders like "01_xxx/"
    for name in os.listdir(video_out):
        if re.match(r"^\\d{2}_.+", name) and os.path.isdir(os.path.join(video_out, name)):
            return True
    return False


def _check_optional_root_compat(video_out: str) -> List[str]:
    warnings: List[str] = []
    if not os.path.exists(os.path.join(video_out, "sampled_frames")):
        warnings.append("Missing <video_out>/sampled_frames (compat artifact); some downstream tools may expect it.")
    if not os.path.exists(os.path.join(video_out, "frame_manifest.json")):
        warnings.append("Missing <video_out>/frame_manifest.json (compat artifact); some downstream tools may expect it.")
    return warnings


def _check_stage1(video_out: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    stage1_dir = os.path.join(video_out, "stage1")
    draft_path = os.path.join(stage1_dir, "draft_plan.json")
    manifest_path = os.path.join(stage1_dir, "frame_manifest.json")
    frames_dir = os.path.join(stage1_dir, "sampled_frames")

    if not os.path.exists(draft_path):
        errors.append(f"Missing: {draft_path}")
        return {}, {}, [], errors, warnings
    if not os.path.exists(manifest_path):
        errors.append(f"Missing: {manifest_path}")
        return {}, {}, [], errors, warnings
    if not os.path.isdir(frames_dir):
        errors.append(f"Missing dir: {frames_dir}")
        return {}, {}, [], errors, warnings

    draft = read_json(draft_path)
    manifest = read_json(manifest_path)

    raw_errs = _stage1_raw_schema_errors(draft)
    hard_errs = _draft_hard_errors(draft)
    if raw_errs:
        errors.extend([f"Stage1 draft schema error: {e}" for e in raw_errs])
    if hard_errs:
        errors.extend([f"Stage1 draft hard error: {e}" for e in hard_errs])

    num_frames = int(manifest.get("num_frames", 0) or 0)
    frames = manifest.get("frames", [])
    if num_frames <= 0 or not isinstance(frames, list) or len(frames) != num_frames:
        errors.append("Stage1 manifest invalid: num_frames/frames mismatch.")
        return draft, manifest, [], errors, warnings

    ts_list: List[float] = []
    for i, fr in enumerate(frames, start=1):
        if not isinstance(fr, dict):
            errors.append(f"Stage1 manifest frames[{i}] is not an object.")
            continue
        try:
            idx1 = int(fr.get("frame_index_1based"))
        except Exception:
            idx1 = None
        if idx1 != i:
            errors.append(f"Stage1 manifest frame_index_1based mismatch at #{i}: got {fr.get('frame_index_1based')}")
        try:
            ts = float(fr.get("timestamp_sec", 0.0))
        except Exception:
            ts = 0.0
        ts_list.append(ts)
        rel = fr.get("image_relpath")
        if not isinstance(rel, str) or not rel:
            errors.append(f"Stage1 manifest frames[{i}] missing image_relpath.")
            continue
        img = os.path.join(stage1_dir, rel)
        if not os.path.exists(img):
            errors.append(f"Stage1 sampled frame missing: {img}")

    return draft, manifest, ts_list, errors, warnings


def _draft_outline_by_id(draft: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for st in draft.get("steps", []) or []:
        if not isinstance(st, dict) or st.get("step_id") is None:
            continue
        try:
            sid = int(st.get("step_id"))
        except Exception:
            continue
        goal = str(st.get("step_goal", "")).strip()
        if sid > 0 and goal:
            out[sid] = goal
    return out


def _check_stage2(
    video_out: str,
    draft: Dict[str, Any],
    stage1_ts: List[float],
    stage1_num_frames: int,
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, Any]], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    stage2_dir = os.path.join(video_out, "stage2")
    loc_path = os.path.join(stage2_dir, "localization_raw.json")
    segments_path = os.path.join(stage2_dir, "step_segments.json")
    if not os.path.exists(loc_path):
        errors.append(f"Missing: {loc_path}")
        return {}, {}, errors, warnings
    if not os.path.exists(segments_path):
        errors.append(f"Missing: {segments_path}")
        return {}, {}, errors, warnings

    localization = read_json(loc_path)
    ok, loc_errors, by_id = validate_stage2_localization(
        draft, localization, stage1_num_frames, frame_timestamps=stage1_ts
    )
    if not ok:
        errors.extend([f"Stage2 localization error: {e}" for e in loc_errors])
        return {}, {}, errors, warnings

    seg_data = read_json(segments_path)
    seg_num_frames = int(seg_data.get("num_frames", -1))
    if seg_num_frames != int(stage1_num_frames):
        errors.append(f"Stage2 step_segments num_frames={seg_num_frames} mismatches Stage1 num_frames={stage1_num_frames}")

    segs = seg_data.get("segments", [])
    if not isinstance(segs, list) or not segs:
        errors.append("Stage2 step_segments.json missing/empty segments list.")
        return by_id, {}, errors, warnings

    expected = _draft_outline_by_id(draft)
    seg_by_id: Dict[int, Dict[str, Any]] = {}
    for i, seg in enumerate(segs):
        if not isinstance(seg, dict) or seg.get("step_id") is None:
            errors.append(f"Stage2 segments[{i}] invalid/missing step_id.")
            continue
        try:
            sid = int(seg.get("step_id"))
        except Exception:
            errors.append(f"Stage2 segments[{i}].step_id is non-int: {seg.get('step_id')}")
            continue
        goal = str(seg.get("step_goal", "")).strip()
        if expected.get(sid) != goal:
            errors.append(f"Stage2 segments step_goal mismatch for step_id={sid}: expected '{expected.get(sid)}', got '{goal}'")

        loc = by_id.get(sid)
        if loc:
            if int(seg.get("start_frame_index", -1)) != int(loc["start_frame_index"]):
                errors.append(f"Stage2 segments start_frame_index mismatch step_id={sid}")
            if int(seg.get("end_frame_index", -1)) != int(loc["end_frame_index"]):
                errors.append(f"Stage2 segments end_frame_index mismatch step_id={sid}")

        try:
            sidx = int(seg.get("start_frame_index"))
            eidx = int(seg.get("end_frame_index"))
            start_sec = float(seg.get("start_sec"))
            end_sec = float(seg.get("end_sec"))
        except Exception:
            errors.append(f"Stage2 segments invalid indices/sec types for step_id={sid}")
            continue

        if not (1 <= sidx <= stage1_num_frames and 1 <= eidx <= stage1_num_frames):
            errors.append(f"Stage2 segments indices out of range for step_id={sid}: {sidx}, {eidx}")
        if not (start_sec < end_sec):
            errors.append(f"Stage2 segments non-positive duration for step_id={sid}: start_sec={start_sec}, end_sec={end_sec}")

        # Start should match the Stage1 manifest exactly; end may be >= the boundary timestamp if adjusted.
        if 1 <= sidx <= len(stage1_ts):
            ref_s = float(stage1_ts[sidx - 1])
            if abs(start_sec - ref_s) > 1e-3:
                warnings.append(
                    f"Stage2 start_sec differs from Stage1 manifest for step_id={sid}: {start_sec:.3f} vs {ref_s:.3f}"
                )
        if 1 <= eidx <= len(stage1_ts):
            ref_e = float(stage1_ts[eidx - 1])
            if end_sec + 1e-3 < ref_e:
                warnings.append(
                    f"Stage2 end_sec earlier than Stage1 manifest boundary for step_id={sid}: {end_sec:.3f} < {ref_e:.3f}"
                )

        clip_rel = seg.get("clip_relpath")
        if not isinstance(clip_rel, str) or not clip_rel:
            errors.append(f"Stage2 segments missing clip_relpath for step_id={sid}")
        else:
            clip_abs = os.path.join(stage2_dir, clip_rel)
            if not os.path.exists(clip_abs):
                errors.append(f"Stage2 clip missing for step_id={sid}: {clip_abs}")
            elif os.path.getsize(clip_abs) <= 0:
                errors.append(f"Stage2 clip is empty for step_id={sid}: {clip_abs}")

        seg_by_id[sid] = seg

    # Ensure coverage of all draft steps.
    for sid in expected:
        if sid not in seg_by_id:
            errors.append(f"Stage2 segments missing step_id={sid}")

    return by_id, seg_by_id, errors, warnings


def _check_stage3_and_final(
    video_out: str,
    draft: Dict[str, Any],
    seg_by_id: Dict[int, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    final_path = os.path.join(video_out, "causal_plan_with_keyframes.json")
    if not os.path.exists(final_path):
        errors.append(f"Missing: {final_path}")
        return errors, warnings
    final = read_json(final_path)

    high_level_goal = final.get("high_level_goal")
    if not (isinstance(high_level_goal, str) and high_level_goal.strip()):
        errors.append("Final plan missing/empty high_level_goal.")

    steps = final.get("steps", [])
    if not isinstance(steps, list) or not steps:
        errors.append("Final plan missing/empty steps list.")
        return errors, warnings

    expected = _draft_outline_by_id(draft)
    got: Dict[int, str] = {}
    step_in_final_by_id: Dict[int, Dict[str, Any]] = {}
    for st in steps:
        if not isinstance(st, dict) or st.get("step_id") is None:
            errors.append("Final plan contains a non-object step.")
            continue
        try:
            sid = int(st.get("step_id"))
        except Exception:
            errors.append(f"Final plan step has non-int step_id: {st.get('step_id')}")
            continue
        goal = str(st.get("step_goal", "")).strip()
        if sid in got:
            errors.append(f"Final plan duplicate step_id={sid}")
            continue
        got[sid] = goal
        step_in_final_by_id[sid] = st
        if expected.get(sid) != goal:
            errors.append(f"Final plan step_goal mismatch step_id={sid}: expected '{expected.get(sid)}', got '{goal}'")

    if expected and expected != got:
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        if missing:
            errors.append(f"Final plan missing step_ids from draft: {missing}")
        if extra:
            errors.append(f"Final plan has extra step_ids not in draft: {extra}")

    for sid, goal in sorted(got.items()):
        step_folder = os.path.join(video_out, f"{sid:02d}_{sanitize_filename(goal)}")
        if not os.path.isdir(step_folder):
            errors.append(f"Missing step folder: {step_folder}")
            continue
        step_meta_path = os.path.join(step_folder, "step_meta.json")
        manifest_path = os.path.join(step_folder, "frame_manifest.json")
        step_out_path = os.path.join(step_folder, "step_final.json")
        if not os.path.exists(manifest_path):
            errors.append(f"Missing: {manifest_path}")
            continue
        if not os.path.exists(step_out_path):
            errors.append(f"Missing: {step_out_path}")
            continue
        if not os.path.exists(step_meta_path):
            errors.append(f"Missing: {step_meta_path}")
            continue

        try:
            manifest = read_json(manifest_path)
            num_frames = int(manifest.get("num_frames", 0) or 0)
        except Exception:
            errors.append(f"Failed to read step manifest: {manifest_path}")
            continue
        if num_frames <= 0:
            errors.append(f"Invalid step manifest num_frames for step_id={sid}: {num_frames}")
            continue

        frames = manifest.get("frames", [])
        if not isinstance(frames, list) or len(frames) != num_frames:
            errors.append(f"Invalid step manifest frames list for step_id={sid} (len(frames) != num_frames).")
            continue
        by_idx: Dict[int, float] = {}
        frame_timestamps: List[float] = []
        for i, fr in enumerate(frames, start=1):
            if not isinstance(fr, dict):
                frame_timestamps.append(0.0)
                continue
            try:
                idx1 = int(fr.get("frame_index_1based"))
            except Exception:
                idx1 = i
            try:
                ts = float(fr.get("timestamp_sec", 0.0))
            except Exception:
                ts = 0.0
            by_idx[idx1] = ts
            frame_timestamps.append(ts)

        step_final = read_json(step_out_path)
        normalized_step_final, step_final_errors = normalize_stage3_step_output(
            step_final, sid, goal, num_frames, frame_timestamps=frame_timestamps
        )
        if normalized_step_final is None:
            errors.extend([f"Stage3 step_final.json error step_id={sid}: {e}" for e in step_final_errors[:30]])
            continue

        # Validate that the step object embedded in the final plan matches the per-step file on disk.
        step_in_final = step_in_final_by_id.get(sid) or {}
        normalized_in_final, in_final_errors = normalize_stage3_step_output(
            step_in_final, sid, goal, num_frames, frame_timestamps=frame_timestamps
        )
        if normalized_in_final is None:
            errors.extend([f"Final plan step schema error step_id={sid}: {e}" for e in in_final_errors[:30]])
            continue
        if normalized_in_final != normalized_step_final:
            errors.append(f"Final plan step differs from {step_out_path} for step_id={sid} (schema-normalized mismatch).")
            continue

        # Ensure step_meta.json matches Stage2 segment metadata (clip + start/end seconds).
        seg = seg_by_id.get(sid)
        if not isinstance(seg, dict):
            errors.append(f"Missing Stage2 segment for step_id={sid} (cannot validate step_meta.json).")
            continue
        stage2_dir = os.path.join(video_out, "stage2")
        clip_rel = seg.get("clip_relpath")
        if not isinstance(clip_rel, str) or not clip_rel:
            errors.append(f"Stage2 segment missing clip_relpath for step_id={sid} (cannot validate step_meta.json).")
            continue
        clip_abs = os.path.join(stage2_dir, clip_rel)
        try:
            seg_start_sec = float(seg.get("start_sec"))
            seg_end_sec = float(seg.get("end_sec"))
        except Exception:
            errors.append(f"Stage2 segment missing/invalid start_sec/end_sec for step_id={sid} (cannot validate step_meta.json).")
            continue
        try:
            meta = read_json(step_meta_path)
        except Exception:
            errors.append(f"Failed to read: {step_meta_path}")
            continue
        try:
            if int(meta.get("step_id")) != int(sid):
                errors.append(f"step_meta.json step_id mismatch for step_id={sid}: got {meta.get('step_id')}")
        except Exception:
            errors.append(f"step_meta.json missing/non-int step_id for step_id={sid}")
        if str(meta.get("step_goal", "")).strip() != str(goal).strip():
            errors.append(f"step_meta.json step_goal mismatch for step_id={sid}: expected '{goal}', got '{meta.get('step_goal')}'")
        rel = meta.get("clip_path")
        if not isinstance(rel, str) or not rel:
            errors.append(f"step_meta.json missing clip_path for step_id={sid}")
        else:
            meta_clip_abs = os.path.abspath(os.path.join(step_folder, rel))
            if os.path.abspath(clip_abs) != meta_clip_abs:
                errors.append(
                    f"step_meta.json clip_path mismatch for step_id={sid}: expected '{clip_abs}', got '{meta_clip_abs}'"
                )
        try:
            meta_s = float(meta.get("clip_start_sec"))
            meta_e = float(meta.get("clip_end_sec"))
            if abs(meta_s - seg_start_sec) > 1e-3 or abs(meta_e - seg_end_sec) > 1e-3:
                errors.append(
                    f"step_meta.json start/end sec mismatch for step_id={sid}: meta=[{meta_s:.3f},{meta_e:.3f}] vs seg=[{seg_start_sec:.3f},{seg_end_sec:.3f}]"
                )
        except Exception:
            errors.append(f"step_meta.json missing/invalid clip_start_sec/clip_end_sec for step_id={sid}")

        # Validate that keyframe images exist and are consistent with the step manifest.
        for i, cf in enumerate(normalized_step_final.get("critical_frames") or []):
            if not isinstance(cf, dict):
                continue
            try:
                fi = int(cf.get("frame_index", -1))
            except Exception:
                fi = -1
            if fi not in by_idx:
                errors.append(f"critical_frames[{i}].frame_index={fi} not found in step manifest for step_id={sid}")
                continue
            ts = float(by_idx[fi])
            expected_name = _keyframe_filename(fi, ts)
            expected_path = os.path.join(step_folder, expected_name)
            if not os.path.exists(expected_path):
                errors.append(f"Missing keyframe image for step_id={sid}, frame_index={fi}: {expected_path}")
                continue
            if os.path.getsize(expected_path) <= 0:
                errors.append(f"Keyframe image is empty for step_id={sid}, frame_index={fi}: {expected_path}")

            # Optional: check timestamp is within Stage2 segment [start_sec, end_sec] (tolerance for rounding).
            ts_2dp = round(ts, 2)
            if float(ts_2dp) + 0.05 < seg_start_sec or float(ts_2dp) - 0.05 > seg_end_sec:
                warnings.append(
                    f"Keyframe ts outside Stage2 segment for step_id={sid}: ts={ts_2dp:.2f}, seg=[{seg_start_sec:.2f},{seg_end_sec:.2f}]"
                )

    return errors, warnings


def check_dependencies() -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        import cv2  # noqa: F401
    except Exception as e:
        errors.append(f"Missing dependency: opencv-python (cv2). Install: pip install opencv-python. Detail: {e}")
    try:
        import openai  # noqa: F401
    except Exception as e:
        errors.append(f"Missing dependency: openai. Install: pip install openai. Detail: {e}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        errors.append("Missing dependency: ffmpeg (not found in PATH). Install ffmpeg or pass --ffmpeg-bin.")
    return errors, warnings


def validate_three_stage_video_output_dir(video_out: str, *, check_deps: bool = False) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if check_deps:
        dep_errs, dep_warns = check_dependencies()
        errors.extend(dep_errs)
        warnings.extend(dep_warns)

    if not os.path.isdir(video_out):
        errors.append(f"Not a directory: {video_out}")
        return False, errors, warnings

    if _looks_like_two_stage_output(video_out):
        errors.append(
            "This directory looks like a TWO-STAGE output (missing stage1/stage2/). "
            "Please validate a THREE-STAGE output directory (contains stage1/ and stage2/)."
        )
        return False, errors, warnings

    warnings.extend(_check_optional_root_compat(video_out))

    draft, _manifest, stage1_ts, s1_errs, s1_warns = _check_stage1(video_out)
    errors.extend(s1_errs)
    warnings.extend(s1_warns)
    if not draft or not stage1_ts:
        return False, errors, warnings

    stage1_num_frames = len(stage1_ts)
    _, seg_by_id, s2_errs, s2_warns = _check_stage2(video_out, draft, stage1_ts, stage1_num_frames)
    errors.extend(s2_errs)
    warnings.extend(s2_warns)

    s3_errs, s3_warns = _check_stage3_and_final(video_out, draft, seg_by_id)
    errors.extend(s3_errs)
    warnings.extend(s3_warns)

    return len(errors) == 0, errors, warnings


def _write_dummy_file(path: str, data: bytes = b"x") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _make_minimal_selftest_dir(tmp_root: str) -> str:
    video_out = os.path.join(tmp_root, "dummy_video_id")
    os.makedirs(video_out, exist_ok=True)

    # Stage1: 5-frame manifest + dummy frames.
    stage1_dir = os.path.join(video_out, "stage1")
    frames_dir = os.path.join(stage1_dir, "sampled_frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames: List[Dict[str, Any]] = []
    for i in range(1, 6):
        ts = float(i - 1)
        name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
        rel = os.path.join("sampled_frames", name)
        frames.append(
            {
                "frame_index_1based": i,
                "timestamp_sec": ts,
                "original_frame_index": i - 1,
                "image_relpath": rel,
            }
        )
        _write_dummy_file(os.path.join(stage1_dir, rel), data=b"jpg")
    write_json(
        os.path.join(stage1_dir, "frame_manifest.json"),
        {"num_frames": 5, "note": "selftest", "frames": frames},
    )

    draft = {
        "high_level_goal": "Demonstrate a minimal three-stage output folder for validation.",
        "steps": [],
    }
    for sid in range(1, 5):
        draft["steps"].append(
            {
                "step_id": sid,
                "step_goal": f"do_step_{sid}",
                "rationale": "A minimal rationale grounded in the sequence.",
                "causal_chain": {
                    "agent": "hands",
                    "action": "move",
                    "patient": "obj_a",
                    "causal_precondition_on_spatial": [
                        {"relation": "contacting", "objects": ["hands", "obj_a"], "truth": True}
                    ],
                    "causal_precondition_on_affordance": [
                        {
                            "object_name": "obj_a",
                            "affordance_types": ["graspable"],
                            "reasons": "The object is reachable and can be grasped.",
                        }
                    ],
                    "causal_effect_on_spatial": [{"relation": "on_top_of", "objects": ["obj_a", "obj_b"], "truth": True}],
                    "causal_effect_on_affordance": [
                        {
                            "object_name": "obj_a",
                            "affordance_types": ["positioned"],
                            "reasons": "After moving, the object ends up in the new position.",
                        }
                    ],
                },
                "counterfactual_challenge_question": "What if obj_a is missing?",
                "expected_challenge_outcome": "The step cannot be completed.",
                "failure_reflecting": {"reason": "obj_a is missing.", "recovery_strategy": "Find a replacement object."},
            }
        )
    write_json(os.path.join(stage1_dir, "draft_plan.json"), draft)

    # Root compat artifacts.
    os.makedirs(os.path.join(video_out, "sampled_frames"), exist_ok=True)
    _write_dummy_file(os.path.join(video_out, "sampled_frames", "sample_001_ts_0.00s.jpg"), data=b"jpg")
    write_json(os.path.join(video_out, "frame_manifest.json"), {"num_frames": 5, "note": "compat", "frames": frames})

    # Stage2: localization + segments + dummy clips.
    stage2_dir = os.path.join(video_out, "stage2")
    clips_dir = os.path.join(stage2_dir, "step_clips")
    os.makedirs(clips_dir, exist_ok=True)
    localization = {
        "steps": [
            {"step_id": 1, "start_frame_index": 1, "end_frame_index": 2},
            {"step_id": 2, "start_frame_index": 2, "end_frame_index": 3},
            {"step_id": 3, "start_frame_index": 3, "end_frame_index": 4},
            {"step_id": 4, "start_frame_index": 4, "end_frame_index": 5},
        ]
    }
    write_json(os.path.join(stage2_dir, "localization_raw.json"), localization)
    segments: List[Dict[str, Any]] = []
    for s in localization["steps"]:
        sid = int(s["step_id"])
        goal = f"do_step_{sid}"
        sidx = int(s["start_frame_index"])
        eidx = int(s["end_frame_index"])
        start_sec = float(sidx - 1)
        end_sec = float(eidx - 1)
        clip_rel = os.path.join("step_clips", f"step{sid:02d}_{sanitize_filename(goal)}.mp4")
        clip_abs = os.path.join(stage2_dir, clip_rel)
        _write_dummy_file(clip_abs, data=b"mp4")
        segments.append(
            {
                "step_id": sid,
                "step_goal": goal,
                "start_frame_index": sidx,
                "end_frame_index": eidx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_image_relpath": frames[sidx - 1]["image_relpath"],
                "end_image_relpath": frames[eidx - 1]["image_relpath"],
                "clip_relpath": clip_rel,
            }
        )
    write_json(
        os.path.join(stage2_dir, "step_segments.json"),
        {
            "source_video": "/abs/path/dummy.mp4",
            "video_id": "dummy_video_id",
            "generated_at_utc": "1970-01-01T00:00:00Z",
            "num_frames": 5,
            "cut": {"mode": "copy", "ffmpeg_bin": "ffmpeg"},
            "segments": segments,
        },
    )

    # Stage3 per-step folders + final plan.
    final_steps: List[Dict[str, Any]] = []
    for seg in segments:
        sid = int(seg["step_id"])
        goal = str(seg["step_goal"])
        step_folder = os.path.join(video_out, f"{sid:02d}_{sanitize_filename(goal)}")
        os.makedirs(step_folder, exist_ok=True)
        step_frames_dir = os.path.join(step_folder, "sampled_frames")
        os.makedirs(step_frames_dir, exist_ok=True)
        start_sec = float(seg["start_sec"])
        end_sec = float(seg["end_sec"])
        # 5 frames uniformly within [start_sec, end_sec].
        step_frames: List[Dict[str, Any]] = []
        for i in range(1, 6):
            alpha = (i - 1) / 4.0
            ts = start_sec + alpha * (end_sec - start_sec)
            name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
            rel = os.path.join("sampled_frames", name)
            step_frames.append(
                {
                    "frame_index_1based": i,
                    "timestamp_sec": ts,
                    "original_frame_index": i - 1,
                    "image_relpath": rel,
                }
            )
            _write_dummy_file(os.path.join(step_folder, rel), data=b"jpg")
        write_json(os.path.join(step_folder, "frame_manifest.json"), {"num_frames": 5, "note": "selftest", "frames": step_frames})

        # Two keyframes saved in the step folder root (matches stage3_refine_and_keyframes.py).
        fi1, fi2 = 2, 5
        for fi in (fi1, fi2):
            kf_ts = float(step_frames[fi - 1]["timestamp_sec"])
            kf_path = os.path.abspath(os.path.join(step_folder, _keyframe_filename(fi, kf_ts)))
            _write_dummy_file(kf_path, data=b"jpg")

        step_final = {
            "step_id": sid,
            "step_goal": goal,
            "rationale": "Refined rationale.",
            "causal_chain": {
                "agent": "hands",
                "action": "move",
                "patient": "obj_a",
                "causal_precondition_on_spatial": [{"relation": "contacting", "objects": ["hands", "obj_a"], "truth": True}],
                "causal_precondition_on_affordance": [
                    {
                        "object_name": "obj_a",
                        "affordance_types": ["graspable"],
                        "reasons": "The object is reachable and can be grasped.",
                    }
                ],
                "causal_effect_on_spatial": [{"relation": "on_top_of", "objects": ["obj_a", "obj_b"], "truth": True}],
                "causal_effect_on_affordance": [
                    {
                        "object_name": "obj_a",
                        "affordance_types": ["positioned"],
                        "reasons": "After moving, the object ends up in the new position.",
                    }
                ],
            },
            "counterfactual_challenge_question": "What if obj_a is missing?",
            "expected_challenge_outcome": "The step cannot be completed.",
            "failure_reflecting": {"reason": "obj_a is missing.", "recovery_strategy": "Find a replacement object."},
            "critical_frames": [
                {
                    "frame_index": fi1,
                    "action_state_change_description": "Initiate moving obj_a using hands; motion begins.",
                    "causal_chain": {
                        "agent": "hands",
                        "action": "move",
                        "patient": "obj_a",
                        "causal_precondition_on_spatial": [
                            {"relation": "contacting", "objects": ["hands", "obj_a"], "truth": True}
                        ],
                        "causal_precondition_on_affordance": [
                            {
                                "object_name": "obj_a",
                                "affordance_types": ["graspable"],
                                "reasons": "The object is reachable and can be grasped.",
                            }
                        ],
                        "causal_effect_on_spatial": [
                            {"relation": "contacting", "objects": ["hands", "obj_a"], "truth": True}
                        ],
                        "causal_effect_on_affordance": [
                            {
                                "object_name": "obj_a",
                                "affordance_types": ["moving"],
                                "reasons": "The object is being moved by the applied force.",
                            }
                        ],
                    },
                    "interaction": {
                        "tools": ["hands"],
                        "materials": ["obj_a"],
                        "hotspot": {
                            "description": "The graspable region of obj_a.",
                            "affordance_type": "graspable",
                            "mechanism": "Grip force transfers motion to the object.",
                        },
                    },
                },
                {
                    "frame_index": fi2,
                    "action_state_change_description": "Complete the movement; obj_a reaches the new position on obj_b.",
                    "causal_chain": {
                        "agent": "hands",
                        "action": "move_and_place",
                        "patient": "obj_a",
                        "causal_precondition_on_spatial": [
                            {"relation": "contacting", "objects": ["hands", "obj_a"], "truth": True}
                        ],
                        "causal_precondition_on_affordance": [
                            {
                                "object_name": "obj_a",
                                "affordance_types": ["movable"],
                                "reasons": "The object can be repositioned by applied force.",
                            }
                        ],
                        "causal_effect_on_spatial": [
                            {"relation": "on_top_of", "objects": ["obj_a", "obj_b"], "truth": True}
                        ],
                        "causal_effect_on_affordance": [
                            {
                                "object_name": "obj_a",
                                "affordance_types": ["positioned"],
                                "reasons": "The object ends in the target placement location.",
                            }
                        ],
                    },
                    "interaction": {
                        "tools": ["hands"],
                        "materials": ["obj_a"],
                        "hotspot": {
                            "description": "The region of obj_a contacted by the fingers/palm.",
                            "affordance_type": "graspable",
                            "mechanism": "Applied force controls the final placement against gravity.",
                        },
                    },
                }
            ],
        }
        write_json(os.path.join(step_folder, "step_final.json"), step_final)
        write_json(
            os.path.join(step_folder, "step_meta.json"),
            {
                "step_id": sid,
                "step_goal": goal,
                "clip_path": os.path.relpath(os.path.join(stage2_dir, str(seg.get("clip_relpath"))), step_folder),
                "clip_start_sec": float(seg.get("start_sec")),
                "clip_end_sec": float(seg.get("end_sec")),
                "num_frames": 5,
                "generated_at_utc": "1970-01-01T00:00:00Z",
                "manifest_path": "frame_manifest.json",
            },
        )
        final_steps.append(step_final)

    write_json(os.path.join(video_out, "causal_plan_with_keyframes.json"), {"high_level_goal": draft["high_level_goal"], "steps": final_steps})
    write_json(
        os.path.join(video_out, "run_summary.json"),
        {
            "source_video": "/abs/path/dummy.mp4",
            "video_id": "dummy_video_id",
            "stage1": {"status": "completed"},
            "stage2": {"status": "completed"},
            "stage3": {"status": "completed"},
        },
    )
    return video_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a THREE-STAGE video output directory (schema + files + index semantics).")
    parser.add_argument("--video-output-dir", help="Path to one <video_id>/ output directory.")
    parser.add_argument("--check-deps", action="store_true", help="Also verify runtime dependencies (opencv-python, openai, ffmpeg).")
    parser.add_argument("--self-test", action="store_true", help="Run an internal self-test (creates a temporary dummy output dir).")
    args = parser.parse_args()

    if not args.video_output_dir and not args.self_test:
        raise SystemExit("Provide --video-output-dir or use --self-test.")

    if args.self_test:
        with tempfile.TemporaryDirectory(prefix="three_stage_selftest_", dir=os.path.dirname(__file__)) as tmp:
            video_out = _make_minimal_selftest_dir(tmp)
            ok, errors, warnings = validate_three_stage_video_output_dir(video_out, check_deps=args.check_deps)
            if warnings:
                print("WARNINGS:")
                for w in warnings:
                    print(" - " + w)
            if errors:
                print("ERRORS:")
                for e in errors:
                    print(" - " + e)
            raise SystemExit(0 if ok else 1)

    ok, errors, warnings = validate_three_stage_video_output_dir(args.video_output_dir, check_deps=args.check_deps)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(" - " + w)
    if errors:
        print("ERRORS:")
        for e in errors:
            print(" - " + e)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
