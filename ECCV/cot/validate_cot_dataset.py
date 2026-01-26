#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FRAME_LEAK_PATTERNS = [
    re.compile(r"\bframe_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bsample_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bts_\d", re.IGNORECASE),
    re.compile(r"\.(jpg|jpeg|png|mp4)\b", re.IGNORECASE),
    re.compile(r"\b(frame|image)\s*\d+\b", re.IGNORECASE),
]


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

KNOWN_TASKS = {
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
}


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN"
    task: str
    file: str
    line_no: int
    entry_id: str
    message: str


def _iter_task_jsonl_files(cot_root: str) -> Iterable[Tuple[str, str]]:
    for name in sorted(os.listdir(cot_root)):
        task_dir = os.path.join(cot_root, name)
        if not os.path.isdir(task_dir):
            continue
        jsonl = os.path.join(task_dir, "data.jsonl")
        if os.path.exists(jsonl):
            yield name, jsonl


def _load_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield line_no, json.loads(s)


def _resolve_path(input_root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(input_root, p)


def _has_frame_leak(text: str) -> bool:
    for pat in FRAME_LEAK_PATTERNS:
        if pat.search(text or ""):
            return True
    return False


def _required_keys(d: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    missing = [k for k in keys if k not in d]
    if missing:
        return f"Missing keys: {missing}"
    return None


# ------------------------
# Anchor derivation helpers
# ------------------------

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


def _build_required_anchors(*, pre_step: Dict[str, Any], eff_step: Dict[str, Any], failure_step: Dict[str, Any]) -> List[str]:
    """Must match ECCV/cot/generate_cot_dataset_api.py behavior exactly."""
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


def _find_all_anchors_in_order(text: str, anchors: Sequence[str]) -> Optional[str]:
    start = 0
    for a in anchors:
        if not a:
            continue
        pos = str(text or "").find(a, start)
        if pos < 0:
            return f"Missing required anchor sentence in <think>: {a}"
        start = pos + len(a)
    return None


@dataclass
class _SourcePlanIndex:
    path: str
    high_level_goal: str
    steps: List[Dict[str, Any]]
    step_id_to_pos: Dict[int, int]  # step_id -> 0-based index in steps


def _load_source_plan_index(
    *,
    input_root: str,
    source_path: str,
    cache: Dict[str, Optional[_SourcePlanIndex]],
) -> Tuple[Optional[_SourcePlanIndex], str]:
    sp = str(source_path or "").strip()
    if not sp:
        return None, "meta.source_path is empty."
    abs_p = os.path.abspath(_resolve_path(input_root, sp))

    if abs_p in cache:
        idx = cache[abs_p]
        return idx, "" if idx else f"Failed to load source plan: {sp}"

    try:
        with open(abs_p, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        cache[abs_p] = None
        return None, f"Failed to load source plan JSON: {sp} ({type(e).__name__}: {e})"

    if not isinstance(plan, dict):
        cache[abs_p] = None
        return None, f"Source plan is not a JSON object: {sp}"

    high_level_goal = str(plan.get("high_level_goal", "")).strip()
    steps_raw = plan.get("steps", [])
    if not isinstance(steps_raw, list) or not steps_raw:
        cache[abs_p] = None
        return None, f"Source plan missing/invalid steps list: {sp}"

    steps: List[Dict[str, Any]] = [s for s in steps_raw if isinstance(s, dict)]
    step_id_to_pos: Dict[int, int] = {}
    for pos, st in enumerate(steps):
        try:
            sid = int(st.get("step_id"))
        except Exception:
            continue
        if sid > 0 and sid not in step_id_to_pos:
            step_id_to_pos[sid] = pos

    idx = _SourcePlanIndex(path=abs_p, high_level_goal=high_level_goal, steps=steps, step_id_to_pos=step_id_to_pos)
    cache[abs_p] = idx
    return idx, ""


def _step_goal(step: Dict[str, Any]) -> str:
    return str(step.get("step_goal", "")).strip()


def _list_str(xs: Any) -> Optional[List[str]]:
    if not isinstance(xs, list) or not xs:
        return None
    out: List[str] = []
    for x in xs:
        if not isinstance(x, str) or not x.strip():
            return None
        out.append(str(x).strip())
    return out


def _validate_task_against_source_and_get_anchors(
    *,
    task_dir_name: str,
    plan_idx: _SourcePlanIndex,
    meta: Dict[str, Any],
    fields: Dict[str, Any],
) -> Tuple[Optional[List[str]], List[str]]:
    errors: List[str] = []

    try:
        step_id = int(meta.get("step_index"))
    except Exception:
        return None, ["meta.step_index must be an int (required for anchor checks)."]

    pos = plan_idx.step_id_to_pos.get(step_id)
    if pos is None:
        return None, [f"Cannot locate meta.step_index={step_id} in source plan steps."]

    steps = plan_idx.steps
    if not steps:
        return None, ["Source plan has empty steps list after filtering."]

    f_hlg = fields.get("high_level_goal")
    if isinstance(f_hlg, str) and f_hlg.strip() and plan_idx.high_level_goal and f_hlg.strip() != plan_idx.high_level_goal:
        errors.append("meta.fields.high_level_goal mismatch vs source plan high_level_goal.")

    if task_dir_name == TASK_17:
        if pos + 1 >= len(steps):
            return None, errors + ["TASK_17 expects an adjacent next step, but current step is last in source plan."]
        prev_step = steps[pos]
        next_step = steps[pos + 1]
        if str(fields.get("previous_step_goal", "")).strip() and str(fields.get("previous_step_goal", "")).strip() != _step_goal(prev_step):
            errors.append("meta.fields.previous_step_goal mismatch vs source plan step_goal.")
        if str(fields.get("next_step_goal", "")).strip() and str(fields.get("next_step_goal", "")).strip() != _step_goal(next_step):
            errors.append("meta.fields.next_step_goal mismatch vs source plan next step_goal.")
        anchors = _build_required_anchors(pre_step=next_step, eff_step=prev_step, failure_step=next_step)
        return anchors, errors

    if task_dir_name == TASK_18:
        if pos + 1 >= len(steps):
            return None, errors + ["TASK_18 expects a next step after the prefix end, but current step is last in source plan."]
        cur_step = steps[pos]
        next_step = steps[pos + 1]
        if str(fields.get("prefix_end_step_goal", "")).strip() and str(fields.get("prefix_end_step_goal", "")).strip() != _step_goal(cur_step):
            errors.append("meta.fields.prefix_end_step_goal mismatch vs source plan step_goal.")
        if str(fields.get("next_step_goal", "")).strip() and str(fields.get("next_step_goal", "")).strip() != _step_goal(next_step):
            errors.append("meta.fields.next_step_goal mismatch vs source plan next step_goal.")
        anchors = _build_required_anchors(pre_step=next_step, eff_step=next_step, failure_step=next_step)
        return anchors, errors

    if task_dir_name == TASK_19:
        mid = _list_str(fields.get("middle_steps"))
        if not mid:
            return None, errors + ["TASK_19 requires meta.fields.middle_steps as non-empty list[str] for anchor checks."]
        head_goal = str(fields.get("head_step_goal", "")).strip()
        if head_goal and head_goal != _step_goal(steps[pos]):
            errors.append("meta.fields.head_step_goal mismatch vs source plan head step_goal.")
        tail_goal = str(fields.get("tail_step_goal", "")).strip()
        tail_pos = pos + 1 + len(mid)
        if tail_pos >= len(steps):
            return None, errors + ["TASK_19 tail step index out of range for source plan."]
        expected_mid_steps = steps[pos + 1 : pos + 1 + len(mid)]
        expected_mid_goals = [_step_goal(s) for s in expected_mid_steps]
        if expected_mid_goals != mid:
            errors.append("meta.fields.middle_steps mismatch vs source plan derived middle steps.")
        tail_step = steps[tail_pos]
        if tail_goal and tail_goal != _step_goal(tail_step):
            errors.append("meta.fields.tail_step_goal mismatch vs source plan tail step_goal.")
        first_missing_step = steps[pos + 1]
        anchors = _build_required_anchors(pre_step=first_missing_step, eff_step=first_missing_step, failure_step=first_missing_step)
        return anchors, errors

    if task_dir_name in (TASK_20, TASK_21, TASK_22, TASK_23):
        try:
            k = int(fields.get("K"))
        except Exception:
            return None, errors + [f"{task_dir_name} requires meta.fields.K as int for anchor checks."]
        if k <= 0:
            return None, errors + [f"{task_dir_name} meta.fields.K must be positive."]

        try:
            prefix_end = int(fields.get("prefix_end_step")) if fields.get("prefix_end_step") is not None else None
        except Exception:
            prefix_end = None
        if prefix_end is not None and prefix_end != step_id:
            errors.append("meta.fields.prefix_end_step mismatch vs meta.step_index.")
        prefix_goal = str(fields.get("prefix_end_step_goal", "")).strip()
        if prefix_goal and prefix_goal != _step_goal(steps[pos]):
            errors.append("meta.fields.prefix_end_step_goal mismatch vs source plan prefix end step_goal.")

        gold_steps = steps[pos + 1 : pos + 1 + k]
        if len(gold_steps) != k:
            return None, errors + [f"{task_dir_name} gold steps out of range in source plan (K={k})."]

        if task_dir_name == TASK_20:
            exp = _list_str(fields.get("next_k_step_goals"))
            if not exp:
                return None, errors + ["TASK_20 requires meta.fields.next_k_step_goals as list[str] for anchor checks."]
            if len(exp) != k:
                errors.append("meta.fields.next_k_step_goals length mismatch vs meta.fields.K.")
            got = [_step_goal(s) for s in gold_steps]
            if got != exp:
                errors.append("meta.fields.next_k_step_goals mismatch vs source plan next K step_goals.")
            anchor_step = gold_steps[0]
            anchors = _build_required_anchors(pre_step=anchor_step, eff_step=anchor_step, failure_step=anchor_step)
            return anchors, errors

        if task_dir_name == TASK_21:
            exp = _list_str(fields.get("gold_next_k_step_goals")) or _list_str(fields.get("label"))
            if not exp:
                return None, errors + ["TASK_21 requires meta.fields.gold_next_k_step_goals (or label) as list[str] for anchor checks."]
            if len(exp) != k:
                errors.append("meta.fields.gold_next_k_step_goals length mismatch vs meta.fields.K.")
            got = [_step_goal(s) for s in gold_steps]
            if got != exp:
                errors.append("meta.fields.gold_next_k_step_goals mismatch vs source plan next K step_goals.")
            presented = _list_str(fields.get("presented_steps"))
            if not presented:
                errors.append("TASK_21 requires meta.fields.presented_steps as list[str].")
            else:
                if len(presented) != k:
                    errors.append("meta.fields.presented_steps length mismatch vs meta.fields.K.")
                elif sorted(presented) != sorted(exp):
                    errors.append("meta.fields.presented_steps must be a permutation of gold_next_k_step_goals.")
            anchor_step = gold_steps[0]
            anchors = _build_required_anchors(pre_step=anchor_step, eff_step=anchor_step, failure_step=anchor_step)
            return anchors, errors

        gold_goals = _list_str(fields.get("gold_plan_steps"))
        bad_goals = _list_str(fields.get("bad_plan_steps"))
        if not gold_goals or not bad_goals or len(gold_goals) != k or len(bad_goals) != k:
            return None, errors + [f"{task_dir_name} requires meta.fields.gold_plan_steps and bad_plan_steps as list[str] (len==K)."]

        gold_from_plan = [_step_goal(s) for s in gold_steps]
        if gold_from_plan != gold_goals:
            errors.append("meta.fields.gold_plan_steps mismatch vs source plan next K step_goals.")

        mism = [i for i in range(k) if gold_goals[i] != bad_goals[i]]
        if len(mism) != 2 or mism[1] != mism[0] + 1:
            return None, errors + [f"{task_dir_name} bad_plan_steps must differ from gold_plan_steps by exactly one adjacent swap."]
        swap_idx = mism[0]
        if bad_goals[swap_idx] != gold_goals[swap_idx + 1] or bad_goals[swap_idx + 1] != gold_goals[swap_idx]:
            return None, errors + [f"{task_dir_name} bad_plan_steps is not a valid adjacent swap of gold_plan_steps."]

        try:
            flaw_step = int(fields.get("flaw_step")) if fields.get("flaw_step") is not None else None
        except Exception:
            flaw_step = None
        if flaw_step is not None and flaw_step != swap_idx + 1:
            errors.append("meta.fields.flaw_step mismatch vs derived flaw position from bad_plan_steps.")

        prereq_step = gold_steps[swap_idx]
        flawed_step = gold_steps[swap_idx + 1]
        anchors = _build_required_anchors(pre_step=flawed_step, eff_step=prereq_step, failure_step=flawed_step)
        return anchors, errors

    if task_dir_name in (TASK_24, TASK_25):
        step = steps[pos]
        cf_q = str(fields.get("counterfactual_challenge_question", "")).strip()
        cf_out = str(fields.get("expected_challenge_outcome", "")).strip()
        if cf_q and cf_q != str(step.get("counterfactual_challenge_question", "")).strip():
            errors.append("meta.fields.counterfactual_challenge_question mismatch vs source plan.")
        if cf_out and cf_out != str(step.get("expected_challenge_outcome", "")).strip():
            errors.append("meta.fields.expected_challenge_outcome mismatch vs source plan.")
        anchors = _build_required_anchors(pre_step=step, eff_step=step, failure_step=step)
        return anchors, errors

    if task_dir_name == TASK_26:
        step = steps[pos]
        fr = step.get("failure_reflecting") if isinstance(step.get("failure_reflecting"), dict) else {}
        fr_reason = str(fr.get("reason", "")).strip()
        fr_recovery = str(fr.get("recovery_strategy", "")).strip()
        if str(fields.get("failure_reason", "")).strip() and str(fields.get("failure_reason", "")).strip() != fr_reason:
            errors.append("meta.fields.failure_reason mismatch vs source plan failure_reflecting.reason.")
        if str(fields.get("recovery_strategy", "")).strip() and str(fields.get("recovery_strategy", "")).strip() != fr_recovery:
            errors.append("meta.fields.recovery_strategy mismatch vs source plan failure_reflecting.recovery_strategy.")
        anchors = _build_required_anchors(pre_step=step, eff_step=step, failure_step=step)
        return anchors, errors

    if task_dir_name == TASK_27:
        if pos + 1 >= len(steps):
            return None, errors + ["TASK_27 expects a next step after the current step, but current step is last in source plan."]
        cur_step = steps[pos]
        next_step = steps[pos + 1]
        if str(fields.get("current_step_goal", "")).strip() and str(fields.get("current_step_goal", "")).strip() != _step_goal(cur_step):
            errors.append("meta.fields.current_step_goal mismatch vs source plan step_goal.")
        if str(fields.get("next_step_goal", "")).strip() and str(fields.get("next_step_goal", "")).strip() != _step_goal(next_step):
            errors.append("meta.fields.next_step_goal mismatch vs source plan next step_goal.")
        fr = cur_step.get("failure_reflecting") if isinstance(cur_step.get("failure_reflecting"), dict) else {}
        if str(fields.get("failure_reason", "")).strip() and str(fields.get("failure_reason", "")).strip() != str(fr.get("reason", "")).strip():
            errors.append("meta.fields.failure_reason mismatch vs source plan failure_reflecting.reason.")
        if str(fields.get("recovery_strategy", "")).strip() and str(fields.get("recovery_strategy", "")).strip() != str(fr.get("recovery_strategy", "")).strip():
            errors.append("meta.fields.recovery_strategy mismatch vs source plan failure_reflecting.recovery_strategy.")
        decision = str(fields.get("decision", "")).strip()
        gold_next = str(fields.get("gold_next_step_goal", "")).strip()
        if decision == "retry_current_step" and gold_next and gold_next != _step_goal(cur_step):
            errors.append("meta.fields.gold_next_step_goal mismatch vs decision=retry_current_step.")
        if decision == "continue_next_step" and gold_next and gold_next != _step_goal(next_step):
            errors.append("meta.fields.gold_next_step_goal mismatch vs decision=continue_next_step.")
        anchors = _build_required_anchors(pre_step=cur_step, eff_step=cur_step, failure_step=cur_step)
        return anchors, errors

    return None, errors + [f"Anchor derivation not implemented for task: {task_dir_name}"]


# ------------------------
# Answer parsing utilities
# ------------------------

def _split_think_and_answer(assistant_text: str) -> Tuple[Optional[str], Optional[str], str]:
    s = str(assistant_text or "")
    if not s.strip():
        return None, None, "Assistant text is empty."
    if not s.startswith("<think>"):
        return None, None, "Assistant text must start with '<think>'."
    end_idx = s.find("</think>")
    if end_idx == -1:
        return None, None, "Assistant text must contain a closing '</think>' tag."
    reasoning = s[len("<think>") : end_idx].strip()
    tail = s[end_idx + len("</think>") :]
    if tail.startswith("\r\n"):
        tail = tail[2:]
    elif tail.startswith("\n") or tail.startswith("\r"):
        tail = tail[1:]
    answer = tail.rstrip("\n\r")
    if not answer.strip():
        return None, None, "Answer text after </think> is empty."
    return reasoning, answer, ""


def _parse_answer_text(answer_text: str) -> Optional[str]:
    s = str(answer_text or "").strip()
    return s if s else None


def _parse_numbered_step_list(answer_text: str) -> List[str]:
    lines = [ln.rstrip() for ln in str(answer_text or "").splitlines()]
    out: List[str] = []
    for ln in lines:
        m = re.match(r"^\s*\d+\)\s*(.+?)\s*$", ln)
        if m:
            s = str(m.group(1)).strip()
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                s = s[1:-1].strip()
            out.append(s)
    return out


def _expected_task17_answer(fields: Dict[str, Any]) -> Optional[str]:
    prev_goal = str(fields.get("previous_step_goal", "")).strip()
    next_goal = str(fields.get("next_step_goal", "")).strip()
    dep = fields.get("dependency_support", {})
    if not isinstance(dep, dict):
        return None
    dep_type = str(dep.get("type", "")).strip()
    effect = re.sub(r"[\s\.\!\?]+$", "", str(dep.get("effect", "")).strip())
    precond = re.sub(r"[\s\.\!\?]+$", "", str(dep.get("precondition", "")).strip())
    if effect and effect[0].isalpha():
        effect = effect[0].lower() + effect[1:]
    if precond and precond[0].isalpha():
        precond = precond[0].lower() + precond[1:]
    if not prev_goal or not next_goal or dep_type not in {"spatial", "affordance"} or not effect or not precond:
        return None
    return (
        f'Completing "{prev_goal}" yields the {dep_type} effect that {effect}, '
        f'which satisfies the precondition that {precond} needed for "{next_goal}".'
    )


def validate_entry(
    *,
    input_root: str,
    task_dir_name: str,
    jsonl_path: str,
    line_no: int,
    entry: Dict[str, Any],
    strict: bool,
    anchor_check: bool,
    plan_cache: Dict[str, Optional[_SourcePlanIndex]],
) -> List[Issue]:
    issues: List[Issue] = []
    entry_id = str(entry.get("id", "")).strip() or "<missing_id>"

    if not isinstance(entry, dict):
        return [Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Entry is not a JSON object.")]
    missing = _required_keys(entry, ["id", "image", "conversations", "meta"])
    if missing:
        return [Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, missing)]

    try:
        uuid.UUID(entry_id)
    except Exception:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "id must be a valid UUID string."))

    image = entry.get("image")
    if not isinstance(image, list) or not image or any(not isinstance(x, str) or not x.strip() for x in image):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "image must be a non-empty list[str]."))

    video = entry.get("video")
    if video is not None and (not isinstance(video, str) or not video.strip()):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "video must be a non-empty string when present."))

    conv = entry.get("conversations")
    if not isinstance(conv, list) or len(conv) != 2:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "conversations must be a list of length 2."))
        return issues
    if not all(isinstance(t, dict) for t in conv):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "conversations[*] must be objects."))
        return issues

    human = conv[0]
    gpt = conv[1]
    if human.get("from") != "human" or not isinstance(human.get("value"), str):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "conversations[0] must be {from:'human', value:str}."))
    if gpt.get("from") != "gpt" or not isinstance(gpt.get("value"), str):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "conversations[1] must be {from:'gpt', value:str}."))

    human_text = str(human.get("value", ""))
    gpt_text = str(gpt.get("value", ""))
    if _has_frame_leak(human_text) or _has_frame_leak(gpt_text):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Frame/path leak detected in conversations text."))
    human_text_stripped = human_text.strip()
    if not human_text_stripped:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Human question text is empty."))
    if "\n" in human_text_stripped or "\r" in human_text_stripped:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Human question must be a single line (no newlines)."))
    if "fields." in human_text_stripped:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Human question must not contain any 'fields.*' lines."))

    reasoning_text, answer_text, split_err = _split_think_and_answer(gpt_text)
    if split_err:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, split_err))
        reasoning_text = ""
        answer_text = ""
    else:
        if not reasoning_text:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "CoT reasoning inside <think> must be non-empty."))
        if "\n" in reasoning_text or "\r" in reasoning_text:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "CoT reasoning must be one paragraph (no newlines)."))
        template_markers = (
            "Spatially,",
            "Functionally,",
            "After the action, spatially,",
            "After the action, functionally,",
            "A likely failure is that",
            "If that happens,",
        )
        missing_markers = [m for m in template_markers if m not in reasoning_text]
        if missing_markers:
            issues.append(
                Issue(
                    "ERROR",
                    task_dir_name,
                    jsonl_path,
                    line_no,
                    entry_id,
                    f"CoT reasoning missing required style markers: {missing_markers}",
                )
            )

    meta = entry.get("meta")
    if not isinstance(meta, dict):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta must be an object."))
        return issues

    if str(meta.get("task_name", "")).strip() != task_dir_name:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.task_name must match directory name."))
    if str(meta.get("item_type", "")).strip() != "three_stage":
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.item_type must be 'three_stage'."))
    if not str(meta.get("evidence_type", "")).strip():
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.evidence_type must be non-empty."))
    if not str(meta.get("source_path", "")).strip():
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.source_path must be non-empty."))
    try:
        int(meta.get("step_index"))
    except Exception:
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.step_index must be an int."))

    fields = meta.get("fields")
    if not isinstance(fields, dict):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.fields must be an object."))
        return issues

    ag = meta.get("assistant_generator")
    if not isinstance(ag, dict) or str(ag.get("type", "")).strip() != "api_generate_v1":
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.assistant_generator.type must be 'api_generate_v1'."))

    ev_files = meta.get("evidence_files")
    if ev_files is not None:
        if not isinstance(ev_files, list) or any(not isinstance(x, str) or not x.strip() for x in ev_files):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.evidence_files must be list[str] when present."))
        else:
            if isinstance(image, list) and image and all(isinstance(x, str) and x.strip() for x in image):
                exp = list(image) + ([video] if isinstance(video, str) and video.strip() else [])
                if [str(x) for x in ev_files] != [str(x) for x in exp]:
                    issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.evidence_files must equal image + (video) in order."))

    if strict:
        for p in (image or []):
            if isinstance(p, str) and p.strip() and not os.path.exists(_resolve_path(input_root, p)):
                issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, f"Missing evidence image file: {p}"))
                break
        if isinstance(video, str) and video.strip() and not os.path.exists(_resolve_path(input_root, video)):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, f"Missing evidence video file: {video}"))
        sp = str(meta.get("source_path", "")).strip()
        if sp and not os.path.exists(_resolve_path(input_root, sp)):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, f"Missing source_path file: {sp}"))

    if anchor_check and reasoning_text:
        plan_idx, plan_err = _load_source_plan_index(
            input_root=input_root,
            source_path=str(meta.get("source_path", "")).strip(),
            cache=plan_cache,
        )
        if plan_err:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, plan_err))
        elif plan_idx:
            anchors, anchor_errors = _validate_task_against_source_and_get_anchors(
                task_dir_name=task_dir_name,
                plan_idx=plan_idx,
                meta=meta,
                fields=fields,
            )
            for msg in anchor_errors:
                issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, msg))
            if anchors:
                order_err = _find_all_anchors_in_order(reasoning_text, anchors)
                if order_err:
                    issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, order_err))

    if task_dir_name == TASK_17:
        exp = _expected_task17_answer(fields)
        got = _parse_answer_text(answer_text)
        if not got:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer must be non-empty text."))
        elif exp and got != exp:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer mismatch vs meta.fields.dependency_support."))

    if task_dir_name == TASK_18:
        exp = str(fields.get("next_step_goal", "")).strip()
        got = _parse_answer_text(answer_text)
        if not exp or not got:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Missing next_step_goal or invalid Answer."))
        elif got != exp:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer mismatch vs meta.fields.next_step_goal."))

    if task_dir_name == TASK_19:
        exp = fields.get("middle_steps")
        got = _parse_numbered_step_list(answer_text)
        if not isinstance(exp, list) or not exp or any(not isinstance(x, str) or not x.strip() for x in exp):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.fields.middle_steps must be list[str]."))
        elif got != [str(x).strip() for x in exp]:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer list mismatch vs meta.fields.middle_steps."))

    if task_dir_name == TASK_20:
        exp = fields.get("next_k_step_goals")
        got = _parse_numbered_step_list(answer_text)
        if not isinstance(exp, list) or not exp or any(not isinstance(x, str) or not x.strip() for x in exp):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.fields.next_k_step_goals must be list[str]."))
        elif got != [str(x).strip() for x in exp]:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer list mismatch vs meta.fields.next_k_step_goals."))

    if task_dir_name == TASK_21:
        exp = fields.get("label")
        if not isinstance(exp, list):
            exp = fields.get("gold_next_k_step_goals")
        got = _parse_numbered_step_list(answer_text)
        if not isinstance(exp, list) or not exp or any(not isinstance(x, str) or not x.strip() for x in exp):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.fields.label must be list[str]."))
        elif got != [str(x).strip() for x in exp]:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer list mismatch vs meta.fields.label."))

    if task_dir_name == TASK_22:
        exp = str(fields.get("label", "")).strip()
        got = _parse_answer_text(answer_text)
        if not exp or not got:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Missing meta.fields.label or invalid Answer."))
        elif got != exp:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer mismatch vs meta.fields.label."))

    if task_dir_name == TASK_23:
        exp = fields.get("label")
        got = _parse_numbered_step_list(answer_text)
        if not isinstance(exp, list) or not exp or any(not isinstance(x, str) or not x.strip() for x in exp):
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta.fields.label must be list[str]."))
        elif got != [str(x).strip() for x in exp]:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer list mismatch vs meta.fields.label."))

    if task_dir_name in (TASK_24, TASK_25):
        exp = str(fields.get("expected_challenge_outcome", "")).strip()
        got = _parse_answer_text(answer_text)
        if not exp or not got:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Missing expected_challenge_outcome or invalid Answer."))
        elif got != exp:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer mismatch vs meta.fields.expected_challenge_outcome."))

    if task_dir_name == TASK_26:
        exp = str(fields.get("recovery_strategy", "")).strip()
        got = _parse_answer_text(answer_text)
        if not exp or not got:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Missing recovery_strategy or invalid Answer."))
        elif got != exp:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer mismatch vs meta.fields.recovery_strategy."))

    if task_dir_name == TASK_27:
        exp = str(fields.get("gold_next_step_goal", "")).strip()
        got = _parse_answer_text(answer_text)
        if not exp or not got:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Missing gold_next_step_goal or invalid Answer."))
        elif got != exp:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Answer mismatch vs meta.fields.gold_next_step_goal."))

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CoT(JSONL) outputs generated from three-stage items (Task_17–Task_27; planning-only).")
    parser.add_argument("--input-root", required=True, help="Root dir containing multiple <video_id>/ folders.")
    parser.add_argument("--cot-root", required=True, help="Root dir containing <task_name>/data.jsonl.")
    parser.add_argument("--strict", action="store_true", help="Check file existence for evidence/source_path.")
    parser.add_argument("--no-anchor-check", action="store_true", help="Disable anchor derivation checks against source plans.")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    cot_root = os.path.abspath(args.cot_root)

    issues: List[Issue] = []
    counts: Dict[str, int] = {}
    plan_cache: Dict[str, Optional[_SourcePlanIndex]] = {}

    for task_dir_name, jsonl_path in _iter_task_jsonl_files(cot_root):
        counts[task_dir_name] = 0
        if task_dir_name not in KNOWN_TASKS:
            issues.append(Issue("WARN", task_dir_name, jsonl_path, 0, "<n/a>", "Unknown task directory name."))
        try:
            for line_no, entry in _load_jsonl(jsonl_path):
                counts[task_dir_name] += 1
                issues.extend(
                    validate_entry(
                        input_root=input_root,
                        task_dir_name=task_dir_name,
                        jsonl_path=jsonl_path,
                        line_no=line_no,
                        entry=entry,
                        strict=bool(args.strict),
                        anchor_check=not bool(args.no_anchor_check),
                        plan_cache=plan_cache,
                    )
                )
        except Exception as e:
            issues.append(Issue("ERROR", task_dir_name, jsonl_path, 0, "<n/a>", f"Failed to load/validate jsonl: {type(e).__name__}: {e}"))

    err_n = sum(1 for it in issues if it.level == "ERROR")
    warn_n = sum(1 for it in issues if it.level == "WARN")

    print("=== CoT Dataset Validation Summary ===")
    for task, n in sorted(counts.items()):
        print(f"- {task}: {n} samples")
    print(f"- Errors: {err_n}")
    print(f"- Warnings: {warn_n}")

    if issues:
        print("\n=== Issues (first 200) ===")
        for it in issues[:200]:
            loc = f"{os.path.relpath(it.file, cot_root)}:{it.line_no}" if it.line_no else os.path.relpath(it.file, cot_root)
            print(f"[{it.level}] task={it.task} file={loc} id={it.entry_id} :: {it.message}")

    raise SystemExit(1 if err_n else 0)


if __name__ == "__main__":
    main()

