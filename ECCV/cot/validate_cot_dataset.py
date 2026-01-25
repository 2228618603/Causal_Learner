#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
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
) -> List[Issue]:
    issues: List[Issue] = []
    entry_id = str(entry.get("id", "")).strip()
    if not entry_id:
        entry_id = "<missing_id>"

    # Top-level structure
    if not isinstance(entry, dict):
        return [Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "Entry is not a JSON object.")]
    missing = _required_keys(entry, ["id", "image", "conversations", "meta"])
    if missing:
        return [Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, missing)]

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

    reasoning_text: str = ""
    answer_text: str = ""
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
        missing = [m for m in template_markers if m not in reasoning_text]
        if missing:
            issues.append(
                Issue(
                    "ERROR",
                    task_dir_name,
                    jsonl_path,
                    line_no,
                    entry_id,
                    f"CoT reasoning missing required style markers: {missing}",
                )
            )

    meta = entry.get("meta")
    if not isinstance(meta, dict):
        issues.append(Issue("ERROR", task_dir_name, jsonl_path, line_no, entry_id, "meta must be an object."))
        return issues

    # Meta checks
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

    # Evidence existence checks (strict).
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

    # Task-specific Answer checks (internal consistency against meta.fields).
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
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    cot_root = os.path.abspath(args.cot_root)

    issues: List[Issue] = []
    counts: Dict[str, int] = {}

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
