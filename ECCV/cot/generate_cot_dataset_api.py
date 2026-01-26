#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THREE_STAGE_DIR = os.path.join(os.path.dirname(THIS_DIR), "three_stage")
if THREE_STAGE_DIR not in sys.path:
    sys.path.insert(0, THREE_STAGE_DIR)

from common import (  # noqa: E402
    ApiConfig,
    build_retry_prefix,
    call_chat_completion,
    extract_json_from_response,
    initialize_api_client,
    logger,
    read_json,
)


TASK_17 = "Task_17_Inter_Step_Dependency_Analysis"
TASK_18 = "Task_18_Next_Step_Goal_Prediction_From_Prefix"
TASK_19 = "Task_19_Middle_Steps_Infill_From_Head_Tail"
TASK_20 = "Task_20_Next_K_Steps_Prediction_From_Prefix_QA"
TASK_21 = "Task_21_Next_K_Steps_Reordering_From_Prefix"
TASK_22 = "Task_22_Failed_Planning_Flaw_Pointing"
TASK_23 = "Task_23_Plan_Repair_From_Flaw"
TASK_24 = "Task_24_Counterfactual_Prediction"
TASK_25 = "Task_25_Counterfactual_Outcome_QA"
TASK_26 = "Task_26_Failure_Recovery_Protocol"
TASK_27 = "Task_27_Next_Step_After_Recovery_QA"

ALL_TASKS = (
    TASK_17,
    TASK_18,
    TASK_19,
    TASK_20,
    TASK_21,
    TASK_22,
    TASK_23,
    TASK_24,
    TASK_25,
    TASK_26,
    TASK_27,
)

ALLOWED_FLAW_TYPES = (
    "tool_mismatch",
    "order_violation",
    "precondition_missing",
    "hallucinated_object",
    "goal_mismatch",
)


FRAME_LEAK_PATTERNS = [
    re.compile(r"\bframe_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bsample_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bts_\d", re.IGNORECASE),
    re.compile(r"\.(jpg|jpeg|png|mp4)\b", re.IGNORECASE),
    re.compile(r"\b(frame|image)\s*\d+\b", re.IGNORECASE),
]

MIN_CROSS_STEP_SUPPORT_SCORE = 0.06


SYSTEM_PROMPT_API_COT = """
You are an expert dataset annotation assistant.

You will be given:
1) A task question (human message).
2) Structured context extracted from a gold three-stage annotation.
3) An exact required final Answer text that MUST be preserved verbatim.

Your job:
- Write high-quality reasoning in English, strictly grounded in the provided context only.
- The reasoning MUST be one coherent natural-language paragraph (no bullet lists, no line breaks).
- In the reasoning, explicitly cover causal planning based on:
  1) spatial + affordance preconditions,
  2) spatial + affordance effects,
  3) failure reflecting (failure reason + recovery strategy),
  and conclude with why the final answer follows.
- Do NOT invent new objects, relations, affordances, steps, tools, failure modes, or outcomes.
- Do NOT reference frames/images/timestamps/file paths or any indexing.

Output format:
- Return STRICT JSON only (no markdown, no extra text).
- Schema: {"assistant_text": "<think>...</think> + <answer>"}.
- assistant_text MUST start with "<think>" and contain exactly one reasoning paragraph inside <think>...</think>.
- After </think>, output the exact required Answer text verbatim (do NOT add "Answer:" or any extra prefixes).
""".strip()


@dataclass(frozen=True)
class Evidence:
    evidence_type: str
    image: List[str]
    video: Optional[str] = None


@dataclass(frozen=True)
class BaseSample:
    task_name: str
    human_q: str
    evidence: Evidence
    step_index: int
    fields: Dict[str, Any]
    context: Dict[str, Any]
    answer_block: Optional[str] = None
    required_anchors: List[str] = field(default_factory=list)


def _has_frame_leak(text: str) -> bool:
    for pat in FRAME_LEAK_PATTERNS:
        if pat.search(text or ""):
            return True
    return False


def normalize_task_names(tasks: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for t in tasks:
        t = str(t or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _iter_video_dirs(input_root: str) -> Iterable[str]:
    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name)
        if os.path.isdir(path):
            yield path


def _load_final_plan(video_dir: str) -> Optional[Dict[str, Any]]:
    plan_path = os.path.join(video_dir, "causal_plan_with_keyframes.json")
    if not os.path.exists(plan_path):
        return None
    try:
        plan = read_json(plan_path)
    except Exception:
        return None
    return plan if isinstance(plan, dict) else None


def _validate_three_stage_plan_minimal(plan: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    high_level_goal = str(plan.get("high_level_goal", "")).strip()
    if not high_level_goal:
        errors.append("Missing/empty high_level_goal.")
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        errors.append("Missing/invalid steps list (expected list length>=2).")
        return errors
    for i, st in enumerate(steps, start=1):
        if not isinstance(st, dict):
            errors.append(f"steps[{i}] is not an object.")
            continue
        if not str(st.get("step_goal", "")).strip():
            errors.append(f"steps[{i}].step_goal is empty.")
        try:
            sid = int(st.get("step_id"))
        except Exception:
            sid = None
        if sid is None or sid <= 0:
            errors.append(f"steps[{i}].step_id missing/non-positive int.")
        cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        for k in ("agent", "action", "patient"):
            if not str(cc.get(k, "")).strip():
                errors.append(f"steps[{i}].causal_chain.{k} is empty.")
        for k in (
            "causal_precondition_on_spatial",
            "causal_precondition_on_affordance",
            "causal_effect_on_spatial",
            "causal_effect_on_affordance",
        ):
            v = cc.get(k)
            if not isinstance(v, str) or not v.strip():
                errors.append(f"steps[{i}].causal_chain.{k} missing/empty string.")
        fr = st.get("failure_reflecting") if isinstance(st.get("failure_reflecting"), dict) else {}
        if not str(fr.get("reason", "")).strip():
            errors.append(f"steps[{i}].failure_reflecting.reason is empty.")
        if not str(fr.get("recovery_strategy", "")).strip():
            errors.append(f"steps[{i}].failure_reflecting.recovery_strategy is empty.")
        if not str(st.get("counterfactual_challenge_question", "")).strip():
            errors.append(f"steps[{i}].counterfactual_challenge_question is empty.")
        if not str(st.get("expected_challenge_outcome", "")).strip():
            errors.append(f"steps[{i}].expected_challenge_outcome is empty.")
        cfs = st.get("critical_frames")
        if not isinstance(cfs, list) or len(cfs) != 2:
            errors.append(f"steps[{i}].critical_frames invalid (expected length==2).")
            continue
        for j, cf in enumerate(cfs):
            if not isinstance(cf, dict):
                errors.append(f"steps[{i}].critical_frames[{j}] is not an object.")
                continue
            try:
                fi = int(cf.get("frame_index"))
            except Exception:
                fi = None
            if fi is None or fi <= 0:
                errors.append(f"steps[{i}].critical_frames[{j}].frame_index missing/non-positive int.")
            ccf = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
            for k in (
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ):
                v = ccf.get(k)
                if not isinstance(v, str) or not v.strip():
                    errors.append(f"steps[{i}].critical_frames[{j}].causal_chain.{k} missing/empty string.")
    return errors


def _discover_step_folders(video_dir: str) -> Dict[int, str]:
    by_id: Dict[int, str] = {}
    for name in os.listdir(video_dir):
        if not re.match(r"^[0-9]{2}_.+", name):
            continue
        step_folder = os.path.join(video_dir, name)
        if not os.path.isdir(step_folder):
            continue
        meta_path = os.path.join(step_folder, "step_meta.json")
        if not os.path.exists(meta_path):
            continue
        step_id: Optional[int] = None
        try:
            meta = read_json(meta_path)
        except Exception:
            meta = None
        if isinstance(meta, dict):
            try:
                step_id = int(meta.get("step_id"))
            except Exception:
                step_id = None
        if step_id is not None and step_id > 0:
            by_id[step_id] = step_folder
    return by_id


def _find_keyframe_image(step_folder: str, frame_index_1based: int) -> Optional[str]:
    prefix = f"frame_{frame_index_1based:03d}_ts_"
    for name in os.listdir(step_folder):
        if name.startswith(prefix) and name.endswith(".jpg"):
            return os.path.join(step_folder, name)
    return None


def _extract_step_keyframes(video_dir: str, step_folder_by_id: Dict[int, str], step: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    try:
        step_id = int(step.get("step_id"))
    except Exception:
        return None, None
    step_folder = step_folder_by_id.get(step_id)
    if not step_folder:
        return None, None
    cfs = step.get("critical_frames")
    if not isinstance(cfs, list) or len(cfs) != 2:
        return None, None
    try:
        init_idx = int(cfs[0].get("frame_index") if isinstance(cfs[0], dict) else None)
        end_idx = int(cfs[1].get("frame_index") if isinstance(cfs[1], dict) else None)
    except Exception:
        return None, None
    init_img = _find_keyframe_image(step_folder, init_idx)
    end_img = _find_keyframe_image(step_folder, end_idx)
    if init_img:
        init_img = os.path.relpath(init_img, video_dir)
    if end_img:
        end_img = os.path.relpath(end_img, video_dir)
    return init_img, end_img


def _resolve_scene_frames_dir(video_dir: str) -> Optional[str]:
    cand = os.path.join(video_dir, "stage1", "sampled_frames")
    return cand if os.path.isdir(cand) else None


def _sample_evenly(items: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    idxs = [int(round(i * (len(items) - 1) / (k - 1))) for i in range(k)]
    out: List[str] = []
    for idx in idxs:
        if 0 <= idx < len(items):
            out.append(items[idx])
    seen: set[str] = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _sample_scene_images(video_dir: str, num_images: int = 4) -> List[str]:
    frames_dir = _resolve_scene_frames_dir(video_dir)
    if not frames_dir:
        return []
    names = sorted([n for n in os.listdir(frames_dir) if n.lower().endswith(".jpg")])
    rels = [os.path.relpath(os.path.join(frames_dir, n), video_dir) for n in names]
    return _sample_evenly(rels, int(num_images))


def _load_stage1_frame_paths(video_dir: str) -> List[str]:
    """Return ordered stage1 sampled frame paths (relative to video_dir)."""
    manifest_path = os.path.join(video_dir, "stage1", "frame_manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = read_json(manifest_path)
        except Exception:
            manifest = None
        if isinstance(manifest, dict):
            frames = manifest.get("frames", [])
            if isinstance(frames, list) and frames:
                out: List[str] = []
                for fr in frames:
                    if not isinstance(fr, dict):
                        continue
                    rel = fr.get("image_relpath")
                    if not isinstance(rel, str) or not rel.strip():
                        continue
                    rel = rel.strip().lstrip("./")
                    if rel.startswith("stage1/"):
                        out.append(rel)
                    else:
                        out.append(os.path.join("stage1", rel))
                seen: set[str] = set()
                uniq: List[str] = []
                for p in out:
                    if p not in seen:
                        seen.add(p)
                        uniq.append(p)
                return uniq

    frames_dir = _resolve_scene_frames_dir(video_dir)
    if not frames_dir:
        return []
    names = sorted([n for n in os.listdir(frames_dir) if n.lower().endswith(".jpg")])
    return [os.path.relpath(os.path.join(frames_dir, n), video_dir) for n in names]


def _load_stage2_segments_map(video_dir: str) -> Dict[int, Dict[str, Any]]:
    seg_path = os.path.join(video_dir, "stage2", "step_segments.json")
    if not os.path.exists(seg_path):
        return {}
    try:
        data = read_json(seg_path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    segs = data.get("segments", [])
    if not isinstance(segs, list):
        return {}
    by_id: Dict[int, Dict[str, Any]] = {}
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        try:
            sid = int(seg.get("step_id"))
        except Exception:
            continue
        if sid > 0 and sid not in by_id:
            by_id[sid] = seg
    return by_id


def _select_stage1_head_tail_images(
    frame_paths: Sequence[str],
    *,
    head_anchor_idx_1based: int,
    tail_anchor_idx_1based: int,
    head_k: int = 4,
    tail_k: int = 4,
) -> List[str]:
    """Select head/tail glimpses from stage1 frame pool around two anchor indices."""
    n = len(frame_paths)
    if n <= 0:
        return []
    head_anchor = max(1, min(int(head_anchor_idx_1based), n))
    tail_anchor = max(1, min(int(tail_anchor_idx_1based), n))
    if head_anchor > tail_anchor:
        head_anchor, tail_anchor = tail_anchor, head_anchor

    head_start = max(1, head_anchor - int(head_k) + 1)
    head_indices = list(range(head_start, head_anchor + 1))

    tail_end = min(n, tail_anchor + int(tail_k) - 1)
    tail_indices = list(range(tail_anchor, tail_end + 1))

    idxs = head_indices + tail_indices
    seen: set[int] = set()
    uniq: List[int] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return [frame_paths[i - 1] for i in uniq if 1 <= i <= n]


def _resolve_video_prefix_relpath(video_dir: str, step_id: int) -> Optional[str]:
    if step_id <= 0:
        return None
    cands = [
        os.path.join("cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}_last.mp4"),
        os.path.join("cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}.mp4"),
    ]
    for rel in cands:
        if os.path.exists(os.path.join(video_dir, rel)):
            return rel
    return None


_NUMBERED_POINT_RE = re.compile(r"^\s*\d+\s*[\.\)]\s*(.+?)\s*$")


def _parse_numbered_points(text: Any) -> List[str]:
    if not isinstance(text, str):
        return []
    out: List[str] = []
    for raw in str(text).splitlines():
        ln = str(raw).strip()
        if not ln:
            continue
        m = _NUMBERED_POINT_RE.match(ln)
        item = m.group(1).strip() if m else ln
        item = re.sub(r"\s+", " ", item).strip()
        if item:
            out.append(item)
    return out


def _first_numbered_point(text: Any) -> str:
    pts = _parse_numbered_points(text)
    return pts[0] if pts else ""


def _lowercase_first_char(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    return s[0].lower() + s[1:] if s[0].isalpha() else s


def _ensure_end_punct(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    return s if s.endswith((".", "!", "?")) else s + "."


def _build_required_anchors(
    *,
    pre_step: Dict[str, Any],
    eff_step: Dict[str, Any],
    failure_step: Dict[str, Any],
) -> List[str]:
    out: List[str] = []

    cc_pre = pre_step.get("causal_chain") if isinstance(pre_step.get("causal_chain"), dict) else {}
    cc_eff = eff_step.get("causal_chain") if isinstance(eff_step.get("causal_chain"), dict) else {}

    sp_pre = _first_numbered_point(cc_pre.get("causal_precondition_on_spatial"))
    if sp_pre:
        out.append(_ensure_end_punct(f"Spatially, {_lowercase_first_char(sp_pre)}"))

    ap_pre = _first_numbered_point(cc_pre.get("causal_precondition_on_affordance"))
    if ap_pre:
        out.append(_ensure_end_punct(f"Functionally, {_lowercase_first_char(ap_pre)}"))

    sp_eff = _first_numbered_point(cc_eff.get("causal_effect_on_spatial"))
    if sp_eff:
        out.append(_ensure_end_punct(f"After the action, spatially, {_lowercase_first_char(sp_eff)}"))

    ap_eff = _first_numbered_point(cc_eff.get("causal_effect_on_affordance"))
    if ap_eff:
        out.append(_ensure_end_punct(f"After the action, functionally, {_lowercase_first_char(ap_eff)}"))

    fr = failure_step.get("failure_reflecting") if isinstance(failure_step.get("failure_reflecting"), dict) else {}
    failure_reason = str(fr.get("reason", "")).strip()
    recovery_strategy = str(fr.get("recovery_strategy", "")).strip()

    def strip_tail_punct(s: str) -> str:
        return re.sub(r"[\s\.\!\?]+$", "", str(s or "").strip())

    if failure_reason:
        out.append(f"A likely failure is that {strip_tail_punct(failure_reason)}.")
    if recovery_strategy:
        out.append(f"If that happens, {strip_tail_punct(recovery_strategy)}.")
    return out


def _match_first_cross_step_support(step_i: Dict[str, Any], step_next: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cc_i = step_i.get("causal_chain") if isinstance(step_i.get("causal_chain"), dict) else {}
    cc_n = step_next.get("causal_chain") if isinstance(step_next.get("causal_chain"), dict) else {}

    sp_eff = _parse_numbered_points(cc_i.get("causal_effect_on_spatial"))
    sp_pre = _parse_numbered_points(cc_n.get("causal_precondition_on_spatial"))
    af_eff = _parse_numbered_points(cc_i.get("causal_effect_on_affordance"))
    af_pre = _parse_numbered_points(cc_n.get("causal_precondition_on_affordance"))

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "into",
        "with",
        "for",
        "from",
        "by",
        "as",
        "is",
        "are",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
        "these",
        "those",
    }

    def tokset(text: str) -> set[str]:
        words = [w.lower() for w in re.findall(r"[a-zA-Z]+", str(text or ""))]
        return {w for w in words if w and w not in stop}

    def best_pair(effects: List[str], preconds: List[str]) -> Tuple[str, str, float]:
        if not effects or not preconds:
            return "", "", -1.0
        best_e = ""
        best_p = ""
        best_s = -1.0
        for e in effects[: min(6, len(effects))]:
            te = tokset(e)
            for p in preconds[: min(6, len(preconds))]:
                tp = tokset(p)
                if not te or not tp:
                    continue
                inter = te & tp
                if not inter:
                    continue
                s = len(inter) / max(1, len(te | tp))
                if s > best_s:
                    best_s = s
                    best_e = e
                    best_p = p
        return best_e, best_p, best_s

    sp_e, sp_p, sp_s = best_pair(sp_eff, sp_pre)
    af_e, af_p, af_s = best_pair(af_eff, af_pre)

    cand: List[Tuple[str, str, str, float]] = []
    if sp_e and sp_p:
        cand.append(("spatial", sp_e, sp_p, float(sp_s)))
    if af_e and af_p:
        cand.append(("affordance", af_e, af_p, float(af_s)))
    if not cand:
        return None

    # Prefer higher similarity; tie-break with spatial.
    cand.sort(key=lambda x: (x[3], 1 if x[0] == "spatial" else 0), reverse=True)
    support_type, effect, precondition, score = cand[0]
    if score < MIN_CROSS_STEP_SUPPORT_SCORE:
        return None
    return {"type": support_type, "effect": effect, "precondition": precondition, "score": score}


_CONTINUE_KEYWORDS = ("continue", "proceed", "move on", "go on", "resume", "next step")
_RETRY_KEYWORDS = ("retry", "try again", "re-try", "repeat", "attempt again", "restart", "redo")
_FIX_KEYWORDS = (
    "reposition",
    "realign",
    "adjust",
    "stabilize",
    "secure",
    "tighten",
    "loosen",
    "wipe",
    "dry",
    "clear",
    "remove",
    "add",
    "increase",
    "decrease",
    "hold",
    "grip",
    "regrasp",
    "re-grasp",
    "reset",
    "replace",
)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "into",
    "with",
    "for",
    "from",
    "by",
    "as",
    "is",
    "are",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
}


def _tokenize_words(text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[a-zA-Z]+", str(text or ""))]
    return [w for w in words if w and w not in _STOPWORDS]


def _infer_retry_or_continue_label(current_step: Dict[str, Any]) -> str:
    fr = current_step.get("failure_reflecting", {}) if isinstance(current_step.get("failure_reflecting"), dict) else {}
    recovery = str(fr.get("recovery_strategy", "")).strip()
    if not recovery:
        return "retry_current_step"
    rec_l = recovery.lower()
    if any(k in rec_l for k in _CONTINUE_KEYWORDS):
        return "continue_next_step"
    if any(k in rec_l for k in _RETRY_KEYWORDS):
        return "retry_current_step"
    if any(k in rec_l for k in _FIX_KEYWORDS):
        return "retry_current_step"
    cc = current_step.get("causal_chain") if isinstance(current_step.get("causal_chain"), dict) else {}
    action = str(cc.get("action", "")).strip()
    step_goal = str(current_step.get("step_goal", "")).strip()
    action_words = set(_tokenize_words(action) + _tokenize_words(step_goal))
    rec_words = set(_tokenize_words(recovery))
    if action_words:
        overlap = len(action_words & rec_words) / max(1, len(action_words))
        if overlap >= 0.45:
            return "continue_next_step"
    return "retry_current_step"


def _to_single_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _format_plan_steps_inline(steps: Sequence[str], *, quote: bool) -> str:
    parts: List[str] = []
    for i, s in enumerate(steps, start=1):
        item = _to_single_line(s)
        if quote:
            item = f'"{item}"'
        parts.append(f"{i}) {item}")
    return " ".join(parts)


def _prompt_task_17(high_level_goal: str, previous_step_goal: str, next_step_goal: str) -> str:
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Previous step goal: "{previous_step_goal}" '
        f'Next step goal: "{next_step_goal}" How does the outcome of the previous step satisfy the preconditions for the next step?'
    )


def _prompt_task_18(high_level_goal: str, prefix_end_step_goal: str) -> str:
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Last completed step (in this prefix): "{prefix_end_step_goal}" '
        "What is the next step goal?"
    )


def _prompt_task_19(high_level_goal: str) -> str:
    return _to_single_line(
        f'High-level goal: "{high_level_goal}" Based on the beginning/end glimpses of the video, infer the missing middle steps in order.'
    )


def _prompt_task_20(high_level_goal: str, prefix_end_step_goal: str, k: int) -> str:
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Last completed step (in this prefix): "{prefix_end_step_goal}" '
        f"Based on this prefix, predict the next K={int(k)} step goals in order."
    )


def _prompt_task_21(high_level_goal: str, prefix_end_step_goal: str, presented_steps: Sequence[str]) -> str:
    cand = _format_plan_steps_inline(presented_steps, quote=True)
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Last completed step (in this prefix): "{prefix_end_step_goal}" '
        f"Reorder the shuffled candidate steps {cand} into the most plausible next-step sequence."
    )


def _prompt_task_22(high_level_goal: str, prefix_end_step_goal: str, bad_plan_steps: Sequence[str]) -> str:
    bad_str = _format_plan_steps_inline(bad_plan_steps, quote=True)
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Last completed step (in this prefix): "{prefix_end_step_goal}" '
        f"Based on this prefix, the following bad_plan_steps are proposed as the next steps: {bad_str} "
        "Identify the flaw in the bad plan. Answer as: FlawStep=<int>; FlawType=<type>; Reason=<one sentence>."
    )


def _prompt_task_23(high_level_goal: str, prefix_end_step_goal: str, bad_plan_steps: Sequence[str]) -> str:
    bad_str = _format_plan_steps_inline(bad_plan_steps, quote=True)
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Last completed step (in this prefix): "{prefix_end_step_goal}" '
        f"Based on this prefix, bad_plan_steps are proposed as the next steps: {bad_str} "
        f"Repair the plan by outputting the corrected {len(bad_plan_steps)}-step sequence."
    )


def _prompt_task_24(step_goal: str, counterfactual_q: str) -> str:
    return _to_single_line(
        f'Context: Step goal: "{step_goal}" Counterfactual: {_to_single_line(counterfactual_q)} '
        "From a spatial & affordance perspective, what would likely happen? Only predict the outcome; do not propose any recovery actions."
    )


def _prompt_task_25(step_goal: str, counterfactual_q: str) -> str:
    q = _to_single_line(counterfactual_q)
    m = re.match(r"^\s*what\s+if\s+(.+?)\s*\??\s*$", q, re.IGNORECASE)
    cond = (m.group(1) if m else q).rstrip("?").strip()
    return _to_single_line(
        f'Context: Step goal: "{step_goal}" What is the most likely outcome if {cond}? '
        "Answer with a short outcome prediction grounded in spatial setup and affordance, and do not propose any recovery actions."
    )


def _prompt_task_26(step_goal: str, failure_reason: str) -> str:
    return _to_single_line(
        f'Context: Step goal: "{step_goal}" Failure reason: "{_to_single_line(failure_reason)}" '
        "What is a plausible recovery strategy? Answer with the recovery strategy only."
    )


def _prompt_task_27(high_level_goal: str, failure_reason: str, recovery_strategy: str) -> str:
    return _to_single_line(
        f'Context: High-level goal: "{high_level_goal}" Failure reason: "{_to_single_line(failure_reason)}" '
        f'Recovery strategy: "{_to_single_line(recovery_strategy)}" '
        "After applying the recovery strategy, what is the most appropriate next step? Answer as a single step_goal."
    )


def _validate_api_payload(payload: Any, *, answer_block: Optional[str], required_anchors: Sequence[str]) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["Output must be a JSON object."]
    if set(payload.keys()) != {"assistant_text"}:
        return [f"JSON must contain exactly one key: assistant_text (got keys={sorted(payload.keys())})."]
    text = payload.get("assistant_text")
    if not isinstance(text, str) or not text.strip():
        return ["assistant_text must be a non-empty string."]
    assistant_text = str(text).rstrip()
    if not assistant_text.startswith("<think>"):
        errors.append("assistant_text must start with '<think>'.")
    if _has_frame_leak(assistant_text):
        errors.append("assistant_text contains forbidden frame/image/file references.")

    close_idx = assistant_text.find("</think>")
    if close_idx == -1:
        errors.append("assistant_text must contain a closing '</think>' tag.")
        return errors

    reasoning_body = assistant_text[len("<think>") : close_idx].strip()
    if not reasoning_body:
        errors.append("CoT reasoning inside <think> must be non-empty.")
    if "\n" in reasoning_body or "\r" in reasoning_body:
        errors.append("CoT reasoning must be a single paragraph (no newlines) inside <think>.")
    # Required anchors must appear verbatim AND in the same order as listed.
    cursor = 0
    for a in required_anchors:
        if not a:
            continue
        pos = reasoning_body.find(a, cursor)
        if pos < 0:
            if reasoning_body.find(a) >= 0:
                errors.append(f"Required anchor sentences must appear in order; out-of-order anchor: {a}")
            else:
                errors.append(f"CoT reasoning must include this exact sentence verbatim: {a}")
        else:
            cursor = pos + len(a)

    tail = assistant_text[close_idx + len("</think>") :]
    if tail.startswith("\r\n"):
        tail = tail[2:]
    elif tail.startswith("\n") or tail.startswith("\r"):
        tail = tail[1:]

    if answer_block:
        want = str(answer_block).rstrip()
        got = str(tail).rstrip()
        if got != want:
            errors.append("assistant_text must end with the exact required Answer text after </think>.")
    else:
        if not tail.strip():
            errors.append("assistant_text must contain the final Answer text after </think>.")
    return errors


def _build_api_user_prompt(sample: BaseSample) -> str:
    parts: List[str] = []
    parts.append(f"Task name: {sample.task_name}\n")
    parts.append("Task question (human message):\n" + sample.human_q.strip() + "\n")
    parts.append("Structured context (gold; do NOT invent new facts):\n")
    parts.append(json.dumps(sample.context, ensure_ascii=False, indent=2))
    parts.append("")
    if sample.answer_block:
        parts.append("Exact required final Answer text (copy verbatim after </think>; do NOT change anything inside it):\n")
        parts.append(sample.answer_block.strip())
        parts.append("")
    if sample.required_anchors:
        parts.append("You MUST include the following exact sentences verbatim in the <think> paragraph:\n")
        for a in sample.required_anchors:
            if a:
                parts.append(f"- {a}")
        parts.append("")
    parts.append("Constraints:\n")
    parts.append('- Output STRICT JSON only, schema: {"assistant_text": "..."}')
    parts.append('- assistant_text MUST start with "<think>" and contain exactly one paragraph inside <think>...</think>.')
    parts.append("- The CoT inside <think> must be ONE paragraph (single line; no bullet lists; no extra headings).")
    parts.append("- In the CoT, analyze: spatial+affordance preconditions -> spatial+affordance effects -> failure reflecting (failure+recovery) -> conclude.")
    if sample.answer_block:
        parts.append("- After </think>, output the exact required Answer text above verbatim (no 'Answer:' prefix).")
    else:
        parts.append("- After </think>, output the final Answer text that follows the task format (no 'Answer:' prefix).")
    parts.append("- Do NOT mention frames/images/timestamps/file paths or any indexing")
    return "\n".join(parts).rstrip() + "\n"


def _call_api_with_retries(
    *,
    client: Any,
    api_cfg: ApiConfig,
    sample: BaseSample,
    max_attempts: int,
    max_tokens: int,
) -> Tuple[Optional[str], List[str]]:
    base_prompt = _build_api_user_prompt(sample)
    last_errors: List[str] = []
    prev_output = ""
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        if attempt == 1:
            user_prompt = base_prompt
        else:
            user_prompt = build_retry_prefix(last_errors, prev_output) + base_prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_API_COT},
            {"role": "user", "content": user_prompt},
        ]
        raw = call_chat_completion(client, api_cfg, messages=messages, max_tokens=int(max_tokens))
        prev_output = raw
        try:
            s = extract_json_from_response(raw)
            payload = json.loads(s)
        except Exception as e:
            last_errors = [f"Invalid JSON: {type(e).__name__}: {e}"]
            continue
        last_errors = _validate_api_payload(payload, answer_block=sample.answer_block, required_anchors=sample.required_anchors)
        if last_errors:
            continue
        # Canonicalize formatting: enforce exactly one newline after </think> and end with newline.
        raw_text = str(payload.get("assistant_text", "")).rstrip()
        close_idx = raw_text.find("</think>")
        if close_idx != -1:
            head = raw_text[: close_idx + len("</think>")]
            tail = raw_text[close_idx + len("</think>") :]
            if tail.startswith("\r\n"):
                tail = tail[2:]
            elif tail.startswith("\n") or tail.startswith("\r"):
                tail = tail[1:]
            raw_text = head + "\n" + tail
        return raw_text.rstrip() + "\n", []
    return None, last_errors


def _write_jsonl(path: str, entries: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_sharegpt_entry(
    *,
    input_root: str,
    video_dir: str,
    abs_paths: bool,
    sample: BaseSample,
    assistant_text: str,
    api_cfg: ApiConfig,
) -> Dict[str, Any]:
    input_root_abs = os.path.abspath(input_root)
    video_dir_abs = os.path.abspath(video_dir)
    source_path_abs = os.path.join(video_dir_abs, "causal_plan_with_keyframes.json")

    def norm(path_rel: str) -> str:
        abs_p = os.path.join(video_dir_abs, path_rel)
        return abs_p if abs_paths else os.path.relpath(abs_p, input_root_abs)

    images = [norm(p) for p in sample.evidence.image]
    video = norm(sample.evidence.video) if sample.evidence.video else None

    source_path = source_path_abs if abs_paths else os.path.relpath(source_path_abs, input_root_abs)
    evidence_files = list(images) + ([video] if video else [])

    meta: Dict[str, Any] = {
        "task_name": sample.task_name,
        "item_type": "three_stage",
        "evidence_type": sample.evidence.evidence_type,
        "source_path": source_path,
        "step_index": int(sample.step_index),
        "fields": sample.fields,
        "evidence_files": evidence_files,
        "assistant_generator": {
            "type": "api_generate_v1",
            "api_base_url": api_cfg.api_base_url,
            "model_provider_id": api_cfg.model_provider_id,
            "model_name": api_cfg.model_name,
        },
    }
    out: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "image": images,
        "conversations": [
            {"from": "human", "value": sample.human_q},
            {"from": "gpt", "value": assistant_text},
        ],
        "meta": meta,
    }
    if video:
        out["video"] = video
    return out


def generate_base_samples_for_video(
    *,
    video_dir: str,
    tasks: Sequence[str],
    rng: random.Random,
    require_video_prefix: bool,
) -> List[BaseSample]:
    out: List[BaseSample] = []

    plan = _load_final_plan(video_dir)
    if not plan:
        return out
    schema_errors = _validate_three_stage_plan_minimal(plan)
    if schema_errors:
        logger.warning("[cot_api] Skip video (schema mismatch): video_dir=%s errors=%s", video_dir, " | ".join(schema_errors[:6]))
        return out

    high_level_goal = str(plan.get("high_level_goal", "")).strip()
    steps = plan.get("steps", [])
    if not high_level_goal or not isinstance(steps, list) or len(steps) < 2:
        return out

    step_folder_by_id = _discover_step_folders(video_dir)
    scene_images = _sample_scene_images(video_dir, num_images=4)
    stage1_frame_paths = _load_stage1_frame_paths(video_dir)
    stage2_segments_by_id = _load_stage2_segments_map(video_dir)

    def strip_tail_punct(s: str) -> str:
        return re.sub(r"[\s\.\!\?]+$", "", str(s or "").strip())

    # Task 17: inter-step dependency analysis (adjacent steps).
    if TASK_17 in tasks:
        for idx in range(len(steps) - 1):
            previous_step = steps[idx]
            next_step = steps[idx + 1]
            if not isinstance(previous_step, dict) or not isinstance(next_step, dict):
                continue
            previous_goal = str(previous_step.get("step_goal", "")).strip()
            next_goal = str(next_step.get("step_goal", "")).strip()
            if not previous_goal or not next_goal:
                continue

            support = _match_first_cross_step_support(previous_step, next_step)
            if not support:
                continue
            support_type = str(support.get("type", "")).strip()
            effect_raw = str(support.get("effect", "")).strip()
            precond_raw = str(support.get("precondition", "")).strip()
            effect = _lowercase_first_char(strip_tail_punct(effect_raw))
            precond = _lowercase_first_char(strip_tail_punct(precond_raw))
            if support_type not in {"spatial", "affordance"} or not effect or not precond:
                continue

            try:
                previous_step_id = int(previous_step.get("step_id", idx + 1))
            except Exception:
                previous_step_id = idx + 1

            _init_prev, end_prev = _extract_step_keyframes(video_dir, step_folder_by_id, previous_step)
            init_next, _end_next = _extract_step_keyframes(video_dir, step_folder_by_id, next_step)
            if not end_prev:
                continue
            imgs = [end_prev] + ([init_next] if init_next else [])
            evidence = Evidence(evidence_type="keyframe_single", image=imgs)
            human_q = _prompt_task_17(high_level_goal, previous_goal, next_goal)
            answer_block = (
                f'Completing "{previous_goal}" yields the {support_type} effect that {effect}, '
                f'which satisfies the precondition that {precond} needed for "{next_goal}".'
            )
            out.append(
                BaseSample(
                    task_name=TASK_17,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=previous_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "previous_step_goal": previous_goal,
                        "next_step_goal": next_goal,
                        "dependency_support": {
                            "type": support_type,
                            "effect": effect_raw,
                            "precondition": precond_raw,
                            "score": float(support.get("score", 0.0) or 0.0),
                        },
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "previous_step": {
                            "step_id": previous_step.get("step_id"),
                            "step_goal": previous_goal,
                            "causal_chain": previous_step.get("causal_chain", {}),
                        },
                        "next_step": {
                            "step_id": next_step.get("step_id"),
                            "step_goal": next_goal,
                            "causal_chain": next_step.get("causal_chain", {}),
                            "failure_reflecting": next_step.get("failure_reflecting", {}),
                            "counterfactual_challenge_question": next_step.get("counterfactual_challenge_question", ""),
                            "expected_challenge_outcome": next_step.get("expected_challenge_outcome", ""),
                        },
                        "dependency_support": {
                            "type": support_type,
                            "effect": effect_raw,
                            "precondition": precond_raw,
                            "score": float(support.get("score", 0.0) or 0.0),
                        },
                    },
                    answer_block=answer_block,
                    required_anchors=_build_required_anchors(pre_step=next_step, eff_step=previous_step, failure_step=next_step),
                )
            )

    # Task 18: next step goal prediction from prefix.
    if TASK_18 in tasks:
        for idx in range(len(steps) - 1):
            current_step = steps[idx]
            next_step = steps[idx + 1]
            if not isinstance(current_step, dict) or not isinstance(next_step, dict):
                continue
            current_goal = str(current_step.get("step_goal", "")).strip()
            next_goal = str(next_step.get("step_goal", "")).strip()
            if not current_goal or not next_goal:
                continue
            try:
                current_step_id = int(current_step.get("step_id", idx + 1))
            except Exception:
                current_step_id = idx + 1

            _init_img, end_img = _extract_step_keyframes(video_dir, step_folder_by_id, current_step)
            prefix_video = _resolve_video_prefix_relpath(video_dir, current_step_id)
            if require_video_prefix and not prefix_video:
                continue

            imgs: List[str] = []
            if end_img:
                imgs.append(end_img)
            for p in scene_images:
                if p not in imgs:
                    imgs.append(p)
                if len(imgs) >= 4:
                    break
            if not imgs:
                continue

            evidence = Evidence(
                evidence_type="video_prefix" if prefix_video else ("images_uniform_scene" if len(imgs) > 1 else "keyframe_single"),
                image=imgs,
                video=prefix_video,
            )
            human_q = _prompt_task_18(high_level_goal, current_goal)
            out.append(
                BaseSample(
                    task_name=TASK_18,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "prefix_end_step": current_step_id,
                        "prefix_end_step_goal": current_goal,
                        "next_step_goal": next_goal,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "prefix_end_step": current_step_id,
                        "prefix_end_step_goal": current_goal,
                        "current_step": {
                            "step_id": current_step_id,
                            "step_goal": current_goal,
                            "causal_chain": current_step.get("causal_chain", {}),
                        },
                        "next_step": {
                            "step_id": next_step.get("step_id"),
                            "step_goal": next_goal,
                            "causal_chain": next_step.get("causal_chain", {}),
                            "failure_reflecting": next_step.get("failure_reflecting", {}),
                        },
                    },
                    answer_block=next_goal,
                    required_anchors=_build_required_anchors(pre_step=next_step, eff_step=next_step, failure_step=next_step),
                )
            )

    # Task 19: infill missing middle steps given head/tail anchors.
    if TASK_19 in tasks:
        candidates: List[Tuple[int, int]] = []
        for head_idx in range(0, len(steps) - 2):
            for tail_idx in range(head_idx + 2, len(steps)):
                middle = [s for s in steps[head_idx + 1 : tail_idx] if isinstance(s, dict)]
                if 1 <= len(middle) <= 3:
                    candidates.append((head_idx, tail_idx))
        rng.shuffle(candidates)
        for head_idx, tail_idx in candidates[: min(2, len(candidates))]:
            head_step = steps[head_idx]
            tail_step = steps[tail_idx]
            if not isinstance(head_step, dict) or not isinstance(tail_step, dict):
                continue
            head_goal = str(head_step.get("step_goal", "")).strip()
            tail_goal = str(tail_step.get("step_goal", "")).strip()
            if not head_goal or not tail_goal:
                continue
            middle_steps = [s for s in steps[head_idx + 1 : tail_idx] if isinstance(s, dict)]
            middle_goals = [str(s.get("step_goal", "")).strip() for s in middle_steps]
            if any(not g for g in middle_goals):
                continue

            imgs: List[str] = []
            if stage1_frame_paths:
                # Prefer true head/tail glimpses from stage1 frame pool, aligned by Stage2 localization if available.
                n_frames = len(stage1_frame_paths)
                head_anchor: Optional[int] = None
                tail_anchor: Optional[int] = None
                try:
                    head_sid = int(head_step.get("step_id"))
                    tail_sid = int(tail_step.get("step_id"))
                except Exception:
                    head_sid = None
                    tail_sid = None
                if head_sid and tail_sid:
                    head_seg = stage2_segments_by_id.get(head_sid)
                    tail_seg = stage2_segments_by_id.get(tail_sid)
                    if isinstance(head_seg, dict) and isinstance(tail_seg, dict):
                        try:
                            head_end_idx = int(head_seg.get("end_frame_index"))
                            head_anchor = max(1, min(int(head_end_idx), n_frames))
                        except Exception:
                            head_anchor = None
                        try:
                            tail_start_idx = int(tail_seg.get("start_frame_index"))
                            tail_anchor = max(1, min(int(tail_start_idx), n_frames))
                        except Exception:
                            tail_anchor = None
                if head_anchor is None or tail_anchor is None:
                    # Fallback: approximate by step order when Stage2 metadata is unavailable.
                    num_steps = max(1, len(steps))
                    head_anchor = max(1, min(n_frames, int(round((head_idx + 1) * n_frames / num_steps))))
                    tail_anchor = max(1, min(n_frames, int(round((tail_idx + 1) * n_frames / num_steps))))
                imgs = _select_stage1_head_tail_images(
                    stage1_frame_paths,
                    head_anchor_idx_1based=int(head_anchor),
                    tail_anchor_idx_1based=int(tail_anchor),
                    head_k=4,
                    tail_k=4,
                )

            if not imgs:
                # Legacy fallback: use step keyframes + a few uniform scene frames.
                _hi, head_end = _extract_step_keyframes(video_dir, step_folder_by_id, head_step)
                _ti, tail_end = _extract_step_keyframes(video_dir, step_folder_by_id, tail_step)
                if head_end:
                    imgs.append(head_end)
                if tail_end and tail_end not in imgs:
                    imgs.append(tail_end)
                for p in scene_images:
                    if p not in imgs:
                        imgs.append(p)
                    if len(imgs) >= 6:
                        break

            if len(imgs) < 2:
                continue
            evidence = Evidence(evidence_type="images_uniform_scene", image=imgs)
            human_q = _prompt_task_19(high_level_goal)
            answer_lines = "\n".join([f"{i}) {g}" for i, g in enumerate(middle_goals, start=1)])
            out.append(
                BaseSample(
                    task_name=TASK_19,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=int(head_step.get("step_id", head_idx + 1)),
                    fields={
                        "high_level_goal": high_level_goal,
                        "head_step_goal": head_goal,
                        "tail_step_goal": tail_goal,
                        "middle_steps": middle_goals,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "head_step": {"step_id": head_step.get("step_id"), "step_goal": head_goal},
                        "tail_step": {"step_id": tail_step.get("step_id"), "step_goal": tail_goal},
                        "middle_steps": middle_goals,
                        "first_missing_step_detail": {
                            "step_id": middle_steps[0].get("step_id"),
                            "step_goal": str(middle_steps[0].get("step_goal", "")).strip(),
                            "causal_chain": middle_steps[0].get("causal_chain", {}),
                            "failure_reflecting": middle_steps[0].get("failure_reflecting", {}),
                        },
                    },
                    answer_block=answer_lines,
                    required_anchors=_build_required_anchors(pre_step=middle_steps[0], eff_step=middle_steps[0], failure_step=middle_steps[0]),
                )
            )

    # Task 20/21/22/23: prefix-based planning (predict/reorder/diagnose/repair).
    min_k = 3
    max_k_global = 6
    if TASK_20 in tasks or TASK_21 in tasks or TASK_22 in tasks or TASK_23 in tasks:
        for prefix_end_idx in range(0, len(steps) - min_k):
            completed_prefix_step = steps[prefix_end_idx]
            if not isinstance(completed_prefix_step, dict):
                continue
            completed_prefix_goal = str(completed_prefix_step.get("step_goal", "")).strip()
            if not completed_prefix_goal:
                continue
            try:
                completed_step_id = int(completed_prefix_step.get("step_id", prefix_end_idx + 1))
            except Exception:
                completed_step_id = prefix_end_idx + 1
            prefix_video = _resolve_video_prefix_relpath(video_dir, completed_step_id)
            if require_video_prefix and not prefix_video:
                continue

            remaining = len(steps) - (prefix_end_idx + 1)
            k_hi = min(max_k_global, remaining)
            if k_hi < min_k:
                continue
            k = rng.randint(min_k, k_hi)
            gold_steps = [s for s in steps[prefix_end_idx + 1 : prefix_end_idx + 1 + k] if isinstance(s, dict)]
            if len(gold_steps) != k:
                continue
            gold_goals = [str(s.get("step_goal", "")).strip() for s in gold_steps]
            if any(not g for g in gold_goals) or len(set(gold_goals)) != len(gold_goals):
                continue

            _pi, prefix_end_img = _extract_step_keyframes(video_dir, step_folder_by_id, completed_prefix_step)
            imgs: List[str] = []
            if prefix_end_img:
                imgs.append(prefix_end_img)
            for p in scene_images:
                if p not in imgs:
                    imgs.append(p)
                if len(imgs) >= 4:
                    break
            if not imgs:
                continue
            evidence = Evidence(
                evidence_type="video_prefix" if prefix_video else ("images_uniform_scene" if len(imgs) > 1 else "keyframe_single"),
                image=imgs,
                video=prefix_video,
            )

            if TASK_20 in tasks:
                human_q = _prompt_task_20(high_level_goal, completed_prefix_goal, k)
                answer_lines = "\n".join([f"{i}) {g}" for i, g in enumerate(gold_goals, start=1)])
                anchor_step = gold_steps[0]
                out.append(
                    BaseSample(
                        task_name=TASK_20,
                        human_q=human_q,
                        evidence=evidence,
                        step_index=completed_step_id,
                        fields={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
                            "prefix_end_step_goal": completed_prefix_goal,
                            "K": k,
                            "next_k_step_goals": gold_goals,
                        },
                        context={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
                            "prefix_end_step_goal": completed_prefix_goal,
                            "K": k,
                            "next_k_step_goals": gold_goals,
                            "next_step_detail": {
                                "step_id": anchor_step.get("step_id"),
                                "step_goal": str(anchor_step.get("step_goal", "")).strip(),
                                "causal_chain": anchor_step.get("causal_chain", {}),
                                "failure_reflecting": anchor_step.get("failure_reflecting", {}),
                            },
                        },
                        answer_block=answer_lines,
                        required_anchors=_build_required_anchors(pre_step=anchor_step, eff_step=anchor_step, failure_step=anchor_step),
                    )
                )

            if TASK_21 in tasks:
                presented = list(gold_goals)
                rng.shuffle(presented)
                if presented == gold_goals:
                    rng.shuffle(presented)
                human_q = _prompt_task_21(high_level_goal, completed_prefix_goal, presented)
                answer_lines = "\n".join([f"{i}) {g}" for i, g in enumerate(gold_goals, start=1)])
                anchor_step = gold_steps[0]
                out.append(
                    BaseSample(
                        task_name=TASK_21,
                        human_q=human_q,
                        evidence=evidence,
                        step_index=completed_step_id,
                        fields={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
                            "prefix_end_step_goal": completed_prefix_goal,
                            "K": k,
                            "presented_steps": presented,
                            "gold_next_k_step_goals": gold_goals,
                            "label": gold_goals,
                        },
                        context={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
                            "prefix_end_step_goal": completed_prefix_goal,
                            "K": k,
                            "presented_steps": presented,
                            "gold_next_k_step_goals": gold_goals,
                            "first_step_in_correct_order_detail": {
                                "step_id": anchor_step.get("step_id"),
                                "step_goal": str(anchor_step.get("step_goal", "")).strip(),
                                "causal_chain": anchor_step.get("causal_chain", {}),
                                "failure_reflecting": anchor_step.get("failure_reflecting", {}),
                            },
                        },
                        answer_block=answer_lines,
                        required_anchors=_build_required_anchors(pre_step=anchor_step, eff_step=anchor_step, failure_step=anchor_step),
                    )
                )

            # Task 22/23: flawed plan + repair.
            if TASK_22 in tasks or TASK_23 in tasks:
                if k < 3:
                    continue
                best_swap_idx: Optional[int] = None
                best_support: Optional[Dict[str, Any]] = None
                best_score = -1.0
                for j in range(len(gold_steps) - 1):
                    support = _match_first_cross_step_support(gold_steps[j], gold_steps[j + 1])
                    if not support:
                        continue
                    try:
                        score = float(support.get("score", -1.0))
                    except Exception:
                        score = -1.0
                    if score > best_score:
                        best_score = score
                        best_swap_idx = j
                        best_support = support
                if best_swap_idx is None or not best_support:
                    continue
                swap_idx = best_swap_idx
                dep_support = best_support

                prereq_step = gold_steps[swap_idx]
                flawed_step = gold_steps[swap_idx + 1]

                bad_goals = list(gold_goals)
                bad_goals[swap_idx], bad_goals[swap_idx + 1] = bad_goals[swap_idx + 1], bad_goals[swap_idx]

                flaw_step_pos = swap_idx + 1  # 1-based index in bad_plan_steps
                flaw_type = "precondition_missing"
                dep_precond = _lowercase_first_char(strip_tail_punct(str(dep_support.get("precondition", "")).strip()))
                flawed_goal = bad_goals[swap_idx]
                prereq_goal = bad_goals[swap_idx + 1]
                reason = f'You cannot "{flawed_goal}" before "{prereq_goal}" because it requires the precondition that {dep_precond}.'
                label_str = f"FlawStep={flaw_step_pos}; FlawType={flaw_type}; Reason={reason}"

                if TASK_22 in tasks:
                    human_q = _prompt_task_22(high_level_goal, completed_prefix_goal, bad_goals)
                    out.append(
                        BaseSample(
                            task_name=TASK_22,
                            human_q=human_q,
                            evidence=evidence,
                            step_index=completed_step_id,
                            fields={
                                "high_level_goal": high_level_goal,
                                "prefix_end_step": completed_step_id,
                                "prefix_end_step_goal": completed_prefix_goal,
                                "K": k,
                                "bad_plan_steps": bad_goals,
                                "gold_plan_steps": gold_goals,
                                "flaw_step": flaw_step_pos,
                                "flaw_type": flaw_type,
                                "dependency_support": dep_support,
                                "label": label_str,
                            },
                            context={
                                "high_level_goal": high_level_goal,
                                "prefix_end_step": completed_step_id,
                                "prefix_end_step_goal": completed_prefix_goal,
                                "bad_plan_steps": bad_goals,
                                "gold_plan_steps": gold_goals,
                                "dependency_support": dep_support,
                                "flaw": {"flaw_step_pos": flaw_step_pos, "flaw_type": flaw_type, "reason": reason},
                                "flawed_step_detail": {
                                    "step_id": flawed_step.get("step_id"),
                                    "step_goal": str(flawed_step.get("step_goal", "")).strip(),
                                    "causal_chain": flawed_step.get("causal_chain", {}),
                                    "failure_reflecting": flawed_step.get("failure_reflecting", {}),
                                },
                                "prerequisite_step_detail": {
                                    "step_id": prereq_step.get("step_id"),
                                    "step_goal": str(prereq_step.get("step_goal", "")).strip(),
                                    "causal_chain": prereq_step.get("causal_chain", {}),
                                    "failure_reflecting": prereq_step.get("failure_reflecting", {}),
                                },
                            },
                            answer_block=label_str,
                            required_anchors=_build_required_anchors(pre_step=flawed_step, eff_step=prereq_step, failure_step=flawed_step),
                        )
                    )

                if TASK_23 in tasks:
                    human_q = _prompt_task_23(high_level_goal, completed_prefix_goal, bad_goals)
                    answer_lines = "\n".join([f'{i}) "{g}"' for i, g in enumerate(gold_goals, start=1)])
                    out.append(
                        BaseSample(
                            task_name=TASK_23,
                            human_q=human_q,
                            evidence=evidence,
                            step_index=completed_step_id,
                            fields={
                                "high_level_goal": high_level_goal,
                                "prefix_end_step": completed_step_id,
                                "prefix_end_step_goal": completed_prefix_goal,
                                "K": k,
                                "bad_plan_steps": bad_goals,
                                "gold_plan_steps": gold_goals,
                                "dependency_support": dep_support,
                                "label": gold_goals,
                            },
                            context={
                                "high_level_goal": high_level_goal,
                                "prefix_end_step": completed_step_id,
                                "prefix_end_step_goal": completed_prefix_goal,
                                "bad_plan_steps": bad_goals,
                                "gold_plan_steps": gold_goals,
                                "dependency_support": dep_support,
                                "flaw": {"flaw_step_pos": flaw_step_pos, "flaw_type": flaw_type, "reason": reason},
                                "flawed_step_detail": {
                                    "step_id": flawed_step.get("step_id"),
                                    "step_goal": str(flawed_step.get("step_goal", "")).strip(),
                                    "causal_chain": flawed_step.get("causal_chain", {}),
                                    "failure_reflecting": flawed_step.get("failure_reflecting", {}),
                                },
                                "prerequisite_step_detail": {
                                    "step_id": prereq_step.get("step_id"),
                                    "step_goal": str(prereq_step.get("step_goal", "")).strip(),
                                    "causal_chain": prereq_step.get("causal_chain", {}),
                                    "failure_reflecting": prereq_step.get("failure_reflecting", {}),
                                },
                            },
                            answer_block=answer_lines,
                            required_anchors=_build_required_anchors(
                                pre_step=flawed_step,
                                eff_step=prereq_step,
                                failure_step=flawed_step,
                            ),
                        )
                    )

    # Task 24/25/26: counterfactual + failure recovery (per step).
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        step_goal = str(step.get("step_goal", "")).strip()
        if not step_goal:
            continue
        try:
            step_id = int(step.get("step_id", idx))
        except Exception:
            step_id = idx
        init_img, _end_img = _extract_step_keyframes(video_dir, step_folder_by_id, step)
        if not init_img:
            continue
        evidence = Evidence(evidence_type="keyframe_single", image=[init_img])

        fr = step.get("failure_reflecting", {}) if isinstance(step.get("failure_reflecting"), dict) else {}
        fr_reason = str(fr.get("reason", "")).strip()
        fr_recovery = str(fr.get("recovery_strategy", "")).strip()

        cf_q = str(step.get("counterfactual_challenge_question", "")).strip()
        cf_out = str(step.get("expected_challenge_outcome", "")).strip()

        if TASK_24 in tasks and cf_q and cf_out:
            human_q = _prompt_task_24(step_goal, cf_q)
            out.append(
                BaseSample(
                    task_name=TASK_24,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "step_goal": step_goal,
                        "counterfactual_challenge_question": cf_q,
                        "expected_challenge_outcome": cf_out,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "step": {"step_id": step_id, "step_goal": step_goal, "causal_chain": step.get("causal_chain", {})},
                        "counterfactual_challenge_question": cf_q,
                        "expected_challenge_outcome": cf_out,
                        "failure_reflecting": fr,
                    },
                    answer_block=cf_out,
                    required_anchors=_build_required_anchors(pre_step=step, eff_step=step, failure_step=step),
                )
            )

        if TASK_25 in tasks and cf_q and cf_out:
            human_q = _prompt_task_25(step_goal, cf_q)
            out.append(
                BaseSample(
                    task_name=TASK_25,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "step_goal": step_goal,
                        "counterfactual_challenge_question": cf_q,
                        "expected_challenge_outcome": cf_out,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "step": {"step_id": step_id, "step_goal": step_goal, "causal_chain": step.get("causal_chain", {})},
                        "counterfactual_challenge_question": cf_q,
                        "expected_challenge_outcome": cf_out,
                        "failure_reflecting": fr,
                    },
                    answer_block=cf_out,
                    required_anchors=_build_required_anchors(pre_step=step, eff_step=step, failure_step=step),
                )
            )

        if TASK_26 in tasks and fr_reason and fr_recovery:
            human_q = _prompt_task_26(step_goal, fr_reason)
            out.append(
                BaseSample(
                    task_name=TASK_26,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "step_goal": step_goal,
                        "failure_reason": fr_reason,
                        "recovery_strategy": fr_recovery,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "step": {"step_id": step_id, "step_goal": step_goal, "causal_chain": step.get("causal_chain", {})},
                        "failure_reason": fr_reason,
                        "failure_reflecting": fr,
                    },
                    answer_block=fr_recovery,
                    required_anchors=_build_required_anchors(pre_step=step, eff_step=step, failure_step=step),
                )
            )

    # Task 27: failure-driven replanning (next step after recovery).
    if TASK_27 in tasks:
        for idx in range(len(steps) - 1):
            current_step = steps[idx]
            next_step = steps[idx + 1]
            if not isinstance(current_step, dict) or not isinstance(next_step, dict):
                continue
            current_goal = str(current_step.get("step_goal", "")).strip()
            next_goal = str(next_step.get("step_goal", "")).strip()
            if not current_goal or not next_goal:
                continue

            fr = current_step.get("failure_reflecting", {}) if isinstance(current_step.get("failure_reflecting"), dict) else {}
            fr_reason = str(fr.get("reason", "")).strip()
            fr_recovery = str(fr.get("recovery_strategy", "")).strip()
            if not fr_reason or not fr_recovery:
                continue

            try:
                current_step_id = int(current_step.get("step_id", idx + 1))
            except Exception:
                current_step_id = idx + 1

            prefix_video = _resolve_video_prefix_relpath(video_dir, current_step_id)
            if require_video_prefix and not prefix_video:
                continue

            init_img, _end_img = _extract_step_keyframes(video_dir, step_folder_by_id, current_step)
            imgs: List[str] = []
            if init_img:
                imgs.append(init_img)
            for p in scene_images:
                if p not in imgs:
                    imgs.append(p)
                if len(imgs) >= 4:
                    break
            if not imgs:
                continue

            evidence = Evidence(
                evidence_type="video_prefix" if prefix_video else ("images_uniform_scene" if len(imgs) > 1 else "keyframe_single"),
                image=imgs,
                video=prefix_video,
            )

            decision = _infer_retry_or_continue_label(current_step)
            target_goal = current_goal if decision == "retry_current_step" else next_goal

            human_q = _prompt_task_27(high_level_goal, fr_reason, fr_recovery)
            out.append(
                BaseSample(
                    task_name=TASK_27,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "prefix_end_step": current_step_id,
                        "prefix_end_step_goal": current_goal,
                        "failure_reason": fr_reason,
                        "recovery_strategy": fr_recovery,
                        "current_step_goal": current_goal,
                        "next_step_goal": next_goal,
                        "decision": decision,
                        "gold_next_step_goal": target_goal,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "prefix_end_step": current_step_id,
                        "current_step": {
                            "step_id": current_step_id,
                            "step_goal": current_goal,
                            "causal_chain": current_step.get("causal_chain", {}),
                            "failure_reflecting": current_step.get("failure_reflecting", {}),
                        },
                        "next_step": {
                            "step_id": next_step.get("step_id"),
                            "step_goal": next_goal,
                            "causal_chain": next_step.get("causal_chain", {}),
                            "failure_reflecting": next_step.get("failure_reflecting", {}),
                        },
                        "failure_reason": fr_reason,
                        "recovery_strategy": fr_recovery,
                        "gold_next_step_goal": target_goal,
                    },
                    answer_block=target_goal,
                    required_anchors=_build_required_anchors(pre_step=current_step, eff_step=current_step, failure_step=current_step),
                )
            )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="API-only CoT generator for three-stage outputs (Task_17–Task_27; planning-only).")
    parser.add_argument("--input-root", required=True, help="Three-stage output root containing multiple <video_id>/ folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory to write <task_name>/data.jsonl.")
    parser.add_argument("--tasks", default=",".join(ALL_TASKS), help="Comma-separated task names to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling K and plan variants.")
    parser.add_argument("--abs-paths", action="store_true", help="Write absolute paths in image/video/source_path.")
    parser.add_argument("--require-video-prefix", action="store_true", help="Skip prefix-based samples when prefix mp4 is missing.")

    # API config (OpenAI-compatible).
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", "http://model.mify.ai.srv/v1"))
    parser.add_argument("--provider", default=os.environ.get("MODEL_PROVIDER_ID", "vertex_ai"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemini-3-pro-preview"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "2048")))
    parser.add_argument("--api-call-retries", type=int, default=int(os.environ.get("API_CALL_RETRIES", "3")))
    parser.add_argument("--api-call-retry-backoff-sec", type=float, default=float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0")))
    parser.add_argument("--max-sample-attempts", type=int, default=3, help="Max attempts per sample for API output validation/retry.")
    parser.add_argument("--post-validate", action="store_true", help="Run validate_cot_dataset.py after generation.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging for API calls.")

    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir)
    tasks = normalize_task_names([t.strip() for t in str(args.tasks).split(",") if t.strip()])
    if not tasks:
        raise SystemExit("No tasks selected.")
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"Unknown/unsupported tasks: {unknown}. Supported tasks: {list(ALL_TASKS)}")

    api_cfg = ApiConfig(
        api_key=str(args.api_key),
        api_base_url=str(args.api_base),
        model_provider_id=str(args.provider),
        model_name=str(args.model),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        api_call_retries=int(args.api_call_retries),
        api_call_retry_backoff_sec=float(args.api_call_retry_backoff_sec),
        embed_index_on_api_images=False,
        verbose=bool(args.verbose),
    )
    client = initialize_api_client(api_cfg)
    if client is None:
        raise SystemExit("Failed to initialize API client. Ensure `openai` is installed and API config is valid.")

    rng = random.Random(int(args.seed))
    all_entries_by_task: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tasks}

    kept = 0
    dropped = 0
    for video_dir in _iter_video_dirs(input_root):
        per_video_rng = random.Random(rng.randint(0, 2**31 - 1))
        base_samples = generate_base_samples_for_video(
            video_dir=video_dir,
            tasks=tasks,
            rng=per_video_rng,
            require_video_prefix=bool(args.require_video_prefix),
        )
        for sample in base_samples:
            assistant_text, errors = _call_api_with_retries(
                client=client,
                api_cfg=api_cfg,
                sample=sample,
                max_attempts=int(args.max_sample_attempts),
                max_tokens=int(args.max_tokens),
            )
            if assistant_text is None or errors:
                dropped += 1
                continue
            entry = _build_sharegpt_entry(
                input_root=input_root,
                video_dir=video_dir,
                abs_paths=bool(args.abs_paths),
                sample=sample,
                assistant_text=assistant_text,
                api_cfg=api_cfg,
            )
            all_entries_by_task[sample.task_name].append(entry)
            kept += 1

    for task_name, entries in all_entries_by_task.items():
        out_path = os.path.join(output_dir, task_name, "data.jsonl")
        _write_jsonl(out_path, entries)
        print(f"[OK] {task_name}: {len(entries)} samples -> {out_path}")

    logger.info("[cot_api] Done. kept=%d dropped=%d output_dir=%s", kept, dropped, output_dir)

    if args.post_validate:
        validator = os.path.join(os.path.dirname(__file__), "validate_cot_dataset.py")
        cmd = [
            sys.executable,
            validator,
            "--input-root",
            input_root,
            "--cot-root",
            output_dir,
            "--strict",
        ]
        logger.info(f"[cot_api] Post-validate: {' '.join(cmd)}")
        import subprocess

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
