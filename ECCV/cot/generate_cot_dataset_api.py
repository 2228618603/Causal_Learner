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


TASK_28 = "Task_28_Inter_Step_Dependency_Analysis"
TASK_29 = "Task_29_Next_Action_Prediction"
TASK_30 = "Task_30_Next_Step_Goal_Prediction_From_Prefix"
TASK_31 = "Task_31_Prefix_Completed_Steps_MultiSelect"
TASK_32 = "Task_32_Middle_Steps_Infill_From_Head_Tail"
TASK_33 = "Task_33_Next_K_Steps_MultiSelect_From_Prefix"
TASK_34 = "Task_34_Next_K_Steps_Reordering_From_Prefix"
TASK_35 = "Task_35_Failed_Planning_Flaw_Pointing"
TASK_36 = "Task_36_Plan_Repair_From_Flaw"
TASK_37 = "Task_37_Counterfactual_Prediction"
TASK_38 = "Task_38_Counterfactual_Outcome_MCQ"
TASK_39 = "Task_39_Failure_Recovery_Protocol"
TASK_40 = "Task_40_Recovery_Strategy_MCQ"
TASK_41 = "Task_41_Recovery_then_Retry_or_Continue"
TASK_42 = "Task_42_Next_Step_After_Recovery"

ALL_TASKS = (
    TASK_28,
    TASK_29,
    TASK_30,
    TASK_31,
    TASK_32,
    TASK_33,
    TASK_34,
    TASK_35,
    TASK_36,
    TASK_37,
    TASK_38,
    TASK_39,
    TASK_40,
    TASK_41,
    TASK_42,
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
            if not isinstance(v, list) or not v:
                errors.append(f"steps[{i}].causal_chain.{k} missing/empty list.")
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


def _spatial_fact_key(item: Dict[str, Any]) -> Optional[Tuple[str, Tuple[str, ...], bool]]:
    rel = str(item.get("relation", "")).strip()
    objs = item.get("objects", [])
    truth = item.get("truth")
    if not rel or not isinstance(objs, list) or not objs or not isinstance(truth, bool):
        return None
    obj_tuple = tuple(str(o).strip() for o in objs if str(o).strip())
    if not obj_tuple:
        return None
    return rel, obj_tuple, truth


def _spatial_fact_str(item: Dict[str, Any]) -> str:
    rel = str(item.get("relation", "")).strip()
    objs = item.get("objects", [])
    truth = item.get("truth")
    obj_str = ", ".join(str(o).strip() for o in objs) if isinstance(objs, list) else str(objs).strip()
    return f"{rel}({obj_str}) => {truth}"


def _humanize_token(token: str) -> str:
    return re.sub(r"_+", " ", str(token or "")).strip()


def _pick_first_spatial_fact(items: Any) -> Optional[Tuple[str, List[str], bool]]:
    if not isinstance(items, list):
        return None
    for it in items:
        if not isinstance(it, dict):
            continue
        rel = str(it.get("relation", "")).strip()
        objs = it.get("objects", [])
        truth = it.get("truth")
        if not rel or not isinstance(objs, list) or not objs or not isinstance(truth, bool):
            continue
        obj_list = [str(o).strip() for o in objs if str(o).strip()]
        if not obj_list:
            continue
        return rel, obj_list, truth
    return None


def _pick_first_affordance_pair(items: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(items, list):
        return None
    for it in items:
        if not isinstance(it, dict):
            continue
        obj = str(it.get("object_name", "")).strip()
        affs = it.get("affordance_types", [])
        if not obj or not isinstance(affs, list) or not affs:
            continue
        for a in affs:
            a = str(a).strip()
            if a:
                return obj, a
    return None


def _spatial_precondition_sentence(rel: str, objs: Sequence[str], truth: bool) -> str:
    rel_words = _humanize_token(rel)
    obj_words = [_humanize_token(o) for o in objs]
    obj_words = [o for o in obj_words if o]
    if not rel_words or not obj_words:
        return ""
    if len(obj_words) == 1:
        return f"Spatially, the {obj_words[0]} must be {rel_words}." if truth else f"Spatially, the {obj_words[0]} must not be {rel_words}."
    if len(obj_words) == 2:
        subj, obj2 = obj_words
        return (
            f"Spatially, the {subj} must be {rel_words} the {obj2}."
            if truth
            else f"Spatially, the {subj} must not be {rel_words} the {obj2}."
        )
    obj_list = ", ".join([f"the {o}" for o in obj_words])
    if truth:
        return f"Spatially, the relation {rel_words} must hold among {obj_list}."
    return f"Spatially, the relation {rel_words} must not hold among {obj_list}."


def _spatial_effect_sentence(rel: str, objs: Sequence[str], truth: bool) -> str:
    rel_words = _humanize_token(rel)
    obj_words = [_humanize_token(o) for o in objs]
    obj_words = [o for o in obj_words if o]
    if not rel_words or not obj_words:
        return ""
    if len(obj_words) == 1:
        return (
            f"After the action, spatially, the {obj_words[0]} will be {rel_words}."
            if truth
            else f"After the action, spatially, the {obj_words[0]} will not be {rel_words}."
        )
    if len(obj_words) == 2:
        subj, obj2 = obj_words
        return (
            f"After the action, spatially, the {subj} will be {rel_words} the {obj2}."
            if truth
            else f"After the action, spatially, the {subj} will not be {rel_words} the {obj2}."
        )
    obj_list = ", ".join([f"the {o}" for o in obj_words])
    if truth:
        return f"After the action, spatially, the relation {rel_words} will hold among {obj_list}."
    return f"After the action, spatially, the relation {rel_words} will not hold among {obj_list}."


def _affordance_precondition_sentence(obj: str, aff: str) -> str:
    obj_words = _humanize_token(obj)
    aff_words = _humanize_token(aff)
    if not obj_words or not aff_words:
        return ""
    return f"Functionally, the {obj_words} must be {aff_words}."


def _affordance_effect_sentence(obj: str, aff: str) -> str:
    obj_words = _humanize_token(obj)
    aff_words = _humanize_token(aff)
    if not obj_words or not aff_words:
        return ""
    return f"After the action, functionally, the {obj_words} will be {aff_words}."


def _build_required_anchors(
    *,
    pre_step: Dict[str, Any],
    eff_step: Dict[str, Any],
    failure_step: Dict[str, Any],
) -> List[str]:
    out: List[str] = []

    cc_pre = pre_step.get("causal_chain") if isinstance(pre_step.get("causal_chain"), dict) else {}
    cc_eff = eff_step.get("causal_chain") if isinstance(eff_step.get("causal_chain"), dict) else {}

    sp_pre = _pick_first_spatial_fact(cc_pre.get("causal_precondition_on_spatial", []))
    if sp_pre:
        sent = _spatial_precondition_sentence(sp_pre[0], sp_pre[1], sp_pre[2])
        if sent:
            out.append(sent)

    ap_pre = _pick_first_affordance_pair(cc_pre.get("causal_precondition_on_affordance", []))
    if ap_pre:
        sent = _affordance_precondition_sentence(ap_pre[0], ap_pre[1])
        if sent:
            out.append(sent)

    sp_eff = _pick_first_spatial_fact(cc_eff.get("causal_effect_on_spatial", []))
    if sp_eff:
        sent = _spatial_effect_sentence(sp_eff[0], sp_eff[1], sp_eff[2])
        if sent:
            out.append(sent)

    ap_eff = _pick_first_affordance_pair(cc_eff.get("causal_effect_on_affordance", []))
    if ap_eff:
        sent = _affordance_effect_sentence(ap_eff[0], ap_eff[1])
        if sent:
            out.append(sent)

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


def _match_first_cross_step_support(step_i: Dict[str, Any], step_next: Dict[str, Any]) -> Tuple[str, str]:
    cc_i = step_i.get("causal_chain") if isinstance(step_i.get("causal_chain"), dict) else {}
    cc_n = step_next.get("causal_chain") if isinstance(step_next.get("causal_chain"), dict) else {}

    effects_sp = cc_i.get("causal_effect_on_spatial", [])
    pre_sp = cc_n.get("causal_precondition_on_spatial", [])
    eff_keys: Dict[Tuple[str, Tuple[str, ...], bool], Dict[str, Any]] = {}
    if isinstance(effects_sp, list):
        for it in effects_sp:
            if not isinstance(it, dict):
                continue
            k = _spatial_fact_key(it)
            if k is not None and k not in eff_keys:
                eff_keys[k] = it
    if isinstance(pre_sp, list):
        for it in pre_sp:
            if not isinstance(it, dict):
                continue
            k = _spatial_fact_key(it)
            if k is not None and k in eff_keys:
                return _spatial_fact_str(it), ""

    effects_aff = cc_i.get("causal_effect_on_affordance", [])
    pre_aff = cc_n.get("causal_precondition_on_affordance", [])
    eff_pairs: set[Tuple[str, str]] = set()
    if isinstance(effects_aff, list):
        for it in effects_aff:
            if not isinstance(it, dict):
                continue
            obj = str(it.get("object_name", "")).strip()
            affs = it.get("affordance_types", [])
            if not obj or not isinstance(affs, list):
                continue
            for a in affs:
                a = str(a).strip()
                if a:
                    eff_pairs.add((obj, a))
    if isinstance(pre_aff, list):
        for it in pre_aff:
            if not isinstance(it, dict):
                continue
            obj = str(it.get("object_name", "")).strip()
            affs = it.get("affordance_types", [])
            if not obj or not isinstance(affs, list):
                continue
            for a in affs:
                a = str(a).strip()
                if a and (obj, a) in eff_pairs:
                    return "", f"{obj}: {a}"

    return "", ""


def _build_mc_options(rng: random.Random, correct: str, distractor_pool: Sequence[str], num_options: int = 4) -> Tuple[List[str], str]:
    pool = [s for s in distractor_pool if s and s != correct]
    if len({correct, *pool}) < num_options:
        return [], ""
    rng.shuffle(pool)
    distractors = pool[: max(0, num_options - 1)]
    options = [correct] + distractors
    if len(options) != num_options:
        return [], ""
    rng.shuffle(options)
    label = "ABCD"[options.index(correct)]
    return options, label


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


def _prompt_task_29(high_level_goal: str, current_step_goal: str) -> str:
    _ = high_level_goal
    _ = current_step_goal
    return "What is the next planned action?"


def _prompt_task_30(high_level_goal: str, prefix_end_step_goal: str) -> str:
    _ = high_level_goal
    _ = prefix_end_step_goal
    return "What is the next step goal?"


def _prompt_task_31(high_level_goal: str, plan_steps: Sequence[str]) -> str:
    _ = high_level_goal
    _ = plan_steps
    return "Up to which step_id has the plan been completed in this prefix?"


def _prompt_task_32(high_level_goal: str, head_step_goal: str, tail_step_goal: str, num_missing: int) -> str:
    _ = high_level_goal
    _ = head_step_goal
    _ = tail_step_goal
    _ = num_missing
    return "Based on the beginning/end glimpses of the video, infer the missing middle steps in order."


def _prompt_task_33(high_level_goal: str, completed_prefix_step_goal: str, k: int, options: Sequence[str]) -> str:
    _ = high_level_goal
    _ = completed_prefix_step_goal
    _ = options
    return f"Select all steps that will occur in the next {int(k)} steps (order not required)."


def _prompt_task_34(high_level_goal: str, completed_prefix_step_goal: str, presented_steps: Sequence[str]) -> str:
    _ = high_level_goal
    _ = completed_prefix_step_goal
    _ = presented_steps
    return "Reorder the shuffled candidate steps into the most plausible next-step sequence."


def _prompt_task_35(high_level_goal: str, completed_prefix_step_goal: str, bad_plan_steps: Sequence[str]) -> str:
    _ = high_level_goal
    _ = completed_prefix_step_goal
    _ = bad_plan_steps
    return "Identify the flaw in the bad plan."


def _prompt_task_36(high_level_goal: str, completed_prefix_step_goal: str, bad_plan_steps: Sequence[str]) -> str:
    _ = high_level_goal
    _ = completed_prefix_step_goal
    return f"Repair the plan by outputting the corrected {len(bad_plan_steps)}-step sequence."


def _prompt_task_37(high_level_goal: str, step_goal: str, counterfactual_q: str) -> str:
    _ = high_level_goal
    _ = step_goal
    return _to_single_line(counterfactual_q)


def _prompt_task_38(high_level_goal: str, step_goal: str, counterfactual_q: str, options: Sequence[str]) -> str:
    _ = high_level_goal
    _ = step_goal
    _ = options
    q = _to_single_line(counterfactual_q)
    m = re.match(r"^\s*what\s+if\s+(.+?)\s*\??\s*$", q, re.IGNORECASE)
    cond = (m.group(1) if m else q).rstrip("?").strip()
    return f"What is the most likely outcome if {cond}?"


def _prompt_task_39(failure_reason: str) -> str:
    reason = _to_single_line(failure_reason)
    return f"If the step fails because {reason}, what is a plausible recovery strategy?"


def _prompt_task_40(high_level_goal: str, step_goal: str, failure_reason: str, options: Sequence[str]) -> str:
    _ = high_level_goal
    _ = step_goal
    _ = options
    reason = _to_single_line(failure_reason)
    return (
        f"Which recovery strategy best resolves the failure where {reason}?"
    )


def _prompt_task_41(
    *,
    high_level_goal: str,
    current_step_goal: str,
    next_step_goal: str,
    failure_reason: str,
    recovery_strategy: str,
) -> str:
    _ = high_level_goal
    _ = current_step_goal
    _ = next_step_goal
    _ = failure_reason
    _ = recovery_strategy
    return "After applying the recovery strategy, should we retry the current step or continue to the next step?"


def _prompt_task_42(high_level_goal: str, failure_reason: str, recovery_strategy: str, options: Sequence[str]) -> str:
    _ = high_level_goal
    _ = failure_reason
    _ = recovery_strategy
    _ = options
    return "After applying the recovery strategy, what is the most appropriate next step?"


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
    for a in required_anchors:
        if a and a not in reasoning_body:
            errors.append(f"CoT reasoning must include this exact sentence verbatim: {a}")

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
        return str(payload.get("assistant_text", "")).rstrip() + "\n", []
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
    all_step_goals = [str(s.get("step_goal", "")).strip() for s in steps if isinstance(s, dict)]
    all_recovery_strategies: List[str] = []
    all_expected_outcomes: List[str] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        fr = st.get("failure_reflecting", {}) if isinstance(st.get("failure_reflecting"), dict) else {}
        rec = str(fr.get("recovery_strategy", "")).strip()
        if rec:
            all_recovery_strategies.append(rec)
        outc = str(st.get("expected_challenge_outcome", "")).strip()
        if outc:
            all_expected_outcomes.append(outc)

    # Task 28: inter-step dependency analysis (adjacent steps).
    if TASK_28 in tasks:
        for idx in range(len(steps) - 1):
            step_i = steps[idx]
            step_n = steps[idx + 1]
            if not isinstance(step_i, dict) or not isinstance(step_n, dict):
                continue
            step_i_goal = str(step_i.get("step_goal", "")).strip()
            step_n_goal = str(step_n.get("step_goal", "")).strip()
            if not step_i_goal or not step_n_goal:
                continue
            spatial_support, affordance_support = _match_first_cross_step_support(step_i, step_n)
            if not spatial_support and not affordance_support:
                continue
            _init_i, end_i = _extract_step_keyframes(video_dir, step_folder_by_id, step_i)
            init_n, _end_n = _extract_step_keyframes(video_dir, step_folder_by_id, step_n)
            if not end_i:
                continue
            imgs = [end_i] + ([init_n] if init_n else [])
            evidence = Evidence(evidence_type="keyframe_single", image=imgs)
            human_q = "How does the outcome of the previous step satisfy the preconditions for the next step?"
            support_bits = [s for s in (spatial_support, affordance_support) if s]
            support_text = " and ".join(support_bits)
            answer_block = (
                f'Completing "{step_i_goal}" establishes {support_text}, which directly supports executing "{step_n_goal}" '
                "by satisfying its corresponding preconditions."
            )
            out.append(
                BaseSample(
                    task_name=TASK_28,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=int(step_i.get("step_id", idx + 1)),
                    fields={
                        "high_level_goal": high_level_goal,
                        "step_n_goal": step_i_goal,
                        "step_next_goal": step_n_goal,
                        "dependency_support": {"spatial": spatial_support, "affordance": affordance_support},
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "step_i": {
                            "step_id": step_i.get("step_id"),
                            "step_goal": step_i_goal,
                            "causal_chain": step_i.get("causal_chain", {}),
                        },
                        "step_i_plus_1": {
                            "step_id": step_n.get("step_id"),
                            "step_goal": step_n_goal,
                            "causal_chain": step_n.get("causal_chain", {}),
                            "failure_reflecting": step_n.get("failure_reflecting", {}),
                            "counterfactual_challenge_question": step_n.get("counterfactual_challenge_question", ""),
                            "expected_challenge_outcome": step_n.get("expected_challenge_outcome", ""),
                        },
                        "dependency_support": {"spatial": spatial_support, "affordance": affordance_support},
                    },
                    answer_block=answer_block,
                    required_anchors=_build_required_anchors(pre_step=step_n, eff_step=step_i, failure_step=step_n),
                )
            )

    # Task 29/30/31: next-step prediction + prefix tracking.
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

        init_img, end_img = _extract_step_keyframes(video_dir, step_folder_by_id, current_step)
        prefix_video = _resolve_video_prefix_relpath(video_dir, current_step_id)

        if TASK_29 in tasks:
            if not end_img:
                continue
            evidence = Evidence(evidence_type="keyframe_single", image=[end_img])
            human_q = _prompt_task_29(high_level_goal, current_goal)
            out.append(
                BaseSample(
                    task_name=TASK_29,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "current_step_goal": current_goal,
                        "next_step_goal": next_goal,
                    },
                    context={
                        "high_level_goal": high_level_goal,
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

        if TASK_30 in tasks:
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
            human_q = _prompt_task_30(high_level_goal, current_goal)
            out.append(
                BaseSample(
                    task_name=TASK_30,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "current_step_goal": current_goal,
                        "next_step_goal": next_goal,
                        "prefix_end_step": current_step_id,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "prefix_end_step": current_step_id,
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

        if TASK_31 in tasks:
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
            human_q = _prompt_task_31(high_level_goal, all_step_goals)
            out.append(
                BaseSample(
                    task_name=TASK_31,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "all_steps": all_step_goals,
                        "prefix_end_step": current_step_id,
                        "label": current_step_id,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "prefix_end_step": current_step_id,
                        "all_steps": all_step_goals,
                        "prefix_end_step_goal": current_goal,
                        "prefix_end_step_detail": {
                            "step_id": current_step_id,
                            "step_goal": current_goal,
                            "causal_chain": current_step.get("causal_chain", {}),
                            "failure_reflecting": current_step.get("failure_reflecting", {}),
                        },
                    },
                    answer_block=str(current_step_id),
                    required_anchors=_build_required_anchors(pre_step=current_step, eff_step=current_step, failure_step=current_step),
                )
            )

    # Task 32: infill missing middle steps given head/tail anchors.
    if TASK_32 in tasks:
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

            _hi, head_end = _extract_step_keyframes(video_dir, step_folder_by_id, head_step)
            _ti, tail_end = _extract_step_keyframes(video_dir, step_folder_by_id, tail_step)
            imgs: List[str] = []
            if head_end:
                imgs.append(head_end)
            if tail_end and tail_end not in imgs:
                imgs.append(tail_end)
            for p in scene_images:
                if p not in imgs:
                    imgs.append(p)
                if len(imgs) >= 6:
                    break
            if not imgs:
                continue
            evidence = Evidence(evidence_type="images_uniform_scene" if len(imgs) > 1 else "keyframe_single", image=imgs)
            human_q = _prompt_task_32(high_level_goal, head_goal, tail_goal, num_missing=len(middle_goals))
            answer_lines = "\n".join([f"{i}) {g}" for i, g in enumerate(middle_goals, start=1)])
            out.append(
                BaseSample(
                    task_name=TASK_32,
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

    # Task 33/34: next-K steps multi-select + reordering (prefix-based).
    min_k = 3
    max_k_global = 6
    if TASK_33 in tasks or TASK_34 in tasks:
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

            if TASK_33 in tasks:
                option_pool = [g for g in all_step_goals if g and g not in gold_goals]
                rng.shuffle(option_pool)
                options = list(gold_goals)
                target_n = min(8, len(set([*options, *option_pool])))
                for cand in option_pool:
                    if cand not in options:
                        options.append(cand)
                    if len(options) >= target_n:
                        break
                if len(options) < len(gold_goals) + 2:
                    continue
                rng.shuffle(options)
                letters = [chr(ord("A") + i) for i in range(len(options))]
                gold_letters_list = [letters[options.index(g)] for g in gold_goals if g in options]
                gold_letters = ",".join(sorted(set(gold_letters_list)))
                human_q = _prompt_task_33(high_level_goal, completed_prefix_goal, k, options)
                anchor_step = gold_steps[0]
                out.append(
                    BaseSample(
                        task_name=TASK_33,
                        human_q=human_q,
                        evidence=evidence,
                        step_index=completed_step_id,
                        fields={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
                            "K": k,
                            "options": options,
                            "label_set": gold_letters,
                            "gold_next_k_step_goals": gold_goals,
                        },
                        context={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
                            "prefix_end_step_goal": completed_prefix_goal,
                            "K": k,
                            "options": options,
                            "gold_next_k_step_goals": gold_goals,
                            "next_step_detail": {
                                "step_id": anchor_step.get("step_id"),
                                "step_goal": str(anchor_step.get("step_goal", "")).strip(),
                                "causal_chain": anchor_step.get("causal_chain", {}),
                                "failure_reflecting": anchor_step.get("failure_reflecting", {}),
                            },
                        },
                        answer_block=gold_letters,
                        required_anchors=_build_required_anchors(pre_step=anchor_step, eff_step=anchor_step, failure_step=anchor_step),
                    )
                )

            if TASK_34 in tasks:
                presented = list(gold_goals)
                rng.shuffle(presented)
                if presented == gold_goals:
                    rng.shuffle(presented)
                human_q = _prompt_task_34(high_level_goal, completed_prefix_goal, presented)
                answer_lines = "\n".join([f"{i}) {g}" for i, g in enumerate(gold_goals, start=1)])
                anchor_step = gold_steps[0]
                out.append(
                    BaseSample(
                        task_name=TASK_34,
                        human_q=human_q,
                        evidence=evidence,
                        step_index=completed_step_id,
                        fields={
                            "high_level_goal": high_level_goal,
                            "prefix_end_step": completed_step_id,
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

            # Task 35/36: flawed plan + repair.
            if TASK_35 in tasks or TASK_36 in tasks:
                if k < 3:
                    continue
                swap_idx: Optional[int] = None
                dep_support = ""
                for j in range(len(gold_steps) - 1):
                    sp_supp, aff_supp = _match_first_cross_step_support(gold_steps[j], gold_steps[j + 1])
                    supp = sp_supp or aff_supp
                    if supp:
                        swap_idx = j
                        dep_support = supp
                        break
                if swap_idx is None:
                    continue

                prereq_step = gold_steps[swap_idx]
                flawed_step = gold_steps[swap_idx + 1]

                bad_goals = list(gold_goals)
                bad_goals[swap_idx], bad_goals[swap_idx + 1] = bad_goals[swap_idx + 1], bad_goals[swap_idx]

                flaw_step_pos = swap_idx + 1  # 1-based index in bad_plan_steps
                flaw_type = "precondition_missing"
                reason = f'It requires "{dep_support}" as a causal precondition, but this bad plan only establishes it later.'
                label_str = f"FlawStep={flaw_step_pos}; FlawType={flaw_type}; Reason={reason}"

                if TASK_35 in tasks:
                    human_q = _prompt_task_35(high_level_goal, completed_prefix_goal, bad_goals)
                    out.append(
                        BaseSample(
                            task_name=TASK_35,
                            human_q=human_q,
                            evidence=evidence,
                            step_index=completed_step_id,
                            fields={
                                "high_level_goal": high_level_goal,
                                "prefix_end_step": completed_step_id,
                                "K": k,
                                "bad_plan_steps": bad_goals,
                                "gold_plan_steps": gold_goals,
                                "flaw_step": flaw_step_pos,
                                "flaw_type": flaw_type,
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

                if TASK_36 in tasks:
                    human_q = _prompt_task_36(high_level_goal, completed_prefix_goal, bad_goals)
                    answer_lines = "\n".join([f"{i}) {g}" for i, g in enumerate(gold_goals, start=1)])
                    out.append(
                        BaseSample(
                            task_name=TASK_36,
                            human_q=human_q,
                            evidence=evidence,
                            step_index=completed_step_id,
                            fields={
                                "high_level_goal": high_level_goal,
                                "prefix_end_step": completed_step_id,
                                "K": k,
                                "bad_plan_steps": bad_goals,
                                "gold_plan_steps": gold_goals,
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
                                "prerequisite_step_detail": {
                                    "step_id": prereq_step.get("step_id"),
                                    "step_goal": str(prereq_step.get("step_goal", "")).strip(),
                                    "causal_chain": prereq_step.get("causal_chain", {}),
                                    "failure_reflecting": prereq_step.get("failure_reflecting", {}),
                                },
                            },
                            answer_block=answer_lines,
                            required_anchors=_build_required_anchors(pre_step=prereq_step, eff_step=prereq_step, failure_step=prereq_step),
                        )
                    )

    # Task 37/38/39/40: counterfactual + failure recovery (per step).
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

        if TASK_37 in tasks and cf_q and cf_out:
            human_q = _prompt_task_37(high_level_goal, step_goal, cf_q)
            out.append(
                BaseSample(
                    task_name=TASK_37,
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

        if TASK_38 in tasks and cf_q and cf_out:
            options, label = _build_mc_options(rng, correct=cf_out, distractor_pool=all_expected_outcomes, num_options=4)
            if options and label:
                human_q = _prompt_task_38(high_level_goal, step_goal, cf_q, options)
                out.append(
                    BaseSample(
                        task_name=TASK_38,
                        human_q=human_q,
                        evidence=evidence,
                        step_index=step_id,
                        fields={
                            "high_level_goal": high_level_goal,
                            "step_goal": step_goal,
                            "counterfactual_challenge_question": cf_q,
                            "options": options,
                            "label": label,
                            "expected_challenge_outcome": cf_out,
                        },
                        context={
                            "high_level_goal": high_level_goal,
                            "step": {"step_id": step_id, "step_goal": step_goal, "causal_chain": step.get("causal_chain", {})},
                            "counterfactual_challenge_question": cf_q,
                            "options": options,
                            "expected_challenge_outcome": cf_out,
                            "failure_reflecting": fr,
                        },
                        answer_block=label,
                        required_anchors=_build_required_anchors(pre_step=step, eff_step=step, failure_step=step),
                    )
                )

        if TASK_39 in tasks and fr_reason and fr_recovery:
            human_q = _prompt_task_39(fr_reason)
            out.append(
                BaseSample(
                    task_name=TASK_39,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "step_goal": step_goal,
                        "reason": fr_reason,
                        "recovery_strategy": fr_recovery,
                    },
                    context={
                        "high_level_goal": high_level_goal,
                        "step": {"step_id": step_id, "step_goal": step_goal, "causal_chain": step.get("causal_chain", {})},
                        "failure_reflecting": fr,
                    },
                    answer_block=fr_recovery,
                    required_anchors=_build_required_anchors(pre_step=step, eff_step=step, failure_step=step),
                )
            )

        if TASK_40 in tasks and fr_reason and fr_recovery:
            options, label = _build_mc_options(rng, correct=fr_recovery, distractor_pool=all_recovery_strategies, num_options=4)
            if options and label:
                human_q = _prompt_task_40(high_level_goal, step_goal, fr_reason, options)
                out.append(
                    BaseSample(
                        task_name=TASK_40,
                        human_q=human_q,
                        evidence=evidence,
                        step_index=step_id,
                        fields={
                            "high_level_goal": high_level_goal,
                            "step_goal": step_goal,
                            "failure_reason": fr_reason,
                            "options": options,
                            "label": label,
                            "gold_recovery_strategy": fr_recovery,
                        },
                        context={
                            "high_level_goal": high_level_goal,
                            "step": {"step_id": step_id, "step_goal": step_goal, "causal_chain": step.get("causal_chain", {})},
                            "failure_reason": fr_reason,
                            "options": options,
                            "gold_recovery_strategy": fr_recovery,
                        },
                        answer_block=label,
                        required_anchors=_build_required_anchors(pre_step=step, eff_step=step, failure_step=step),
                    )
                )

    # Task 41/42: recovery then retry/continue (binary + MC).
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

        if TASK_41 in tasks:
            label = _infer_retry_or_continue_label(current_step)
            human_q = _prompt_task_41(
                high_level_goal=high_level_goal,
                current_step_goal=current_goal,
                next_step_goal=next_goal,
                failure_reason=fr_reason,
                recovery_strategy=fr_recovery,
            )
            out.append(
                BaseSample(
                    task_name=TASK_41,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "current_step_goal": current_goal,
                        "next_step_goal": next_goal,
                        "failure_reason": fr_reason,
                        "recovery_strategy": fr_recovery,
                        "label": label,
                    },
                    context={
                        "high_level_goal": high_level_goal,
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
                    },
                    answer_block=label,
                    required_anchors=_build_required_anchors(pre_step=current_step, eff_step=current_step, failure_step=current_step),
                )
            )

        if TASK_42 in tasks:
            decision = _infer_retry_or_continue_label(current_step)
            target_goal = current_goal if decision == "retry_current_step" else next_goal
            options = [current_goal, next_goal]
            pool = [g for g in all_step_goals if g and g not in options]
            rng.shuffle(pool)
            options.extend(pool[:2])
            if len(options) != 4 or len(set(options)) != 4 or target_goal not in options:
                continue
            rng.shuffle(options)
            label = "ABCD"[options.index(target_goal)]
            human_q = _prompt_task_42(high_level_goal, fr_reason, fr_recovery, options)
            out.append(
                BaseSample(
                    task_name=TASK_42,
                    human_q=human_q,
                    evidence=evidence,
                    step_index=current_step_id,
                    fields={
                        "high_level_goal": high_level_goal,
                        "failure_reason": fr_reason,
                        "recovery_strategy": fr_recovery,
                        "prefix_end_step": current_step_id,
                        "options": options,
                        "label": label,
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
                        "options": options,
                        "gold_next_step_goal": target_goal,
                    },
                    answer_block=label,
                    required_anchors=_build_required_anchors(pre_step=current_step, eff_step=current_step, failure_step=current_step),
                )
            )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="API-only CoT generator for three-stage outputs (Task_28–Task_42).")
    parser.add_argument("--input-root", required=True, help="Three-stage output root containing multiple <video_id>/ folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory to write <task_name>/data.jsonl.")
    parser.add_argument("--tasks", default=",".join(ALL_TASKS), help="Comma-separated task names to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling distractors/options.")
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
