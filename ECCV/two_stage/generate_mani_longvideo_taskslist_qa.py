#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Mani-LongVideo QA dataset from `causal_plan_with_keyframes.json`.

This script converts the QA-only tasks defined in:
  ECCV/tasklist/mani_longvideo_taskslist_final.md (Task_01–Task_27)
into ShareGPT-style JSONL entries, following the spirit of:
  ECCV/two_stage_old/generate_phyplan_api.py

Key requirements:
- Single type only.
- Source of truth is `<ITEM_DIR>/causal_plan_with_keyframes.json` and its schema must align with the canonical mani_longvideo
  schema described in `ECCV/tasklist/mani_longvideo_taskslist_final.md`.
- Strict final-schema validation is enabled by default (disable via `--no-strict-schema` only if you know you need legacy tolerance).
- Each sample must declare the multimodal evidence type (4 types) and include evidence file paths.
- Keep meta minimal (avoid redundant fields).
- Avoid leaking filenames/frame indices/timestamps into the Q/A text.

LLM two-stage rewrite (optional, recommended for better naturalness):
- By default, the script will use an OpenAI-compatible API when `API_KEY` is set (disable via `--no-api`).
- Control which tasks are rewritten via `--llm-tasks` (default: a small subset that benefits most).

Example:
  python ECCV/two_stage/generate_mani_longvideo_taskslist_qa.py --input-root <DATA_ROOT> --output-dir <OUT_ROOT> --require-videos
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mani_longvideo_taskslist_qa")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


TASK_01 = "Task_01_Goal_Recognition_From_Full_Video"
TASK_02 = "Task_02_Macro_Anchor_Extraction"
TASK_03 = "Task_03_Clip_to_StepGoal_Statement"
TASK_04 = "Task_04_Patient_Identification_QA"
TASK_05 = "Task_05_Action_Phrase_QA"
TASK_06 = "Task_06_Hotspot_AffordanceType_QA"
TASK_07 = "Task_07_Hotspot_Mechanism_QA"
TASK_08 = "Task_08_Micro_Affordance_Visual_Semantics"
TASK_09 = "Task_09_State_Evolution_Description"
TASK_10 = "Task_10_Holistic_Causal_Chain_Analysis"
TASK_11 = "Task_11_Strategic_Rationale_Justification"
TASK_12 = "Task_12_Spatial_Precondition_Description"
TASK_13 = "Task_13_Affordance_Precondition_Description"
TASK_14 = "Task_14_Physical_Feasibility_Verification_QA"
TASK_15 = "Task_15_Spatial_Postcondition_Description"
TASK_16 = "Task_16_Affordance_Postcondition_Description"
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


ALL_TASKS: Tuple[str, ...] = (
    TASK_01,
    TASK_02,
    TASK_03,
    TASK_04,
    TASK_05,
    TASK_06,
    TASK_07,
    TASK_08,
    TASK_09,
    TASK_10,
    TASK_11,
    TASK_12,
    TASK_13,
    TASK_14,
    TASK_15,
    TASK_16,
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


EVIDENCE_KEYFRAME = "keyframe_single"
EVIDENCE_UNIFORM = "images_uniform_scene"
EVIDENCE_CLIP = "video_clip"
EVIDENCE_PREFIX = "video_prefix"


FRAME_LEAK_PATTERNS = [
    re.compile(r"\bframe_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bsample_\d{3}\b", re.IGNORECASE),
    re.compile(r"\bts_\d", re.IGNORECASE),
    re.compile(r"\.(jpg|jpeg|png|mp4)\b", re.IGNORECASE),
    re.compile(r"\b(frame|image)\s*\d+\b", re.IGNORECASE),
]


_KEY_MOMENT_PREFIX_RE = re.compile(r"^\s*Key moment\s*\d+\s*\([^)]*\)\s*:\s*", re.IGNORECASE)
_TS_RE = re.compile(r"_ts_(\d+(?:\.\d+)?)s\b", re.IGNORECASE)
_SNAKE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_LOWER_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_]*\b")


_GENERIC_OBJECT_TOKENS = {
    "hand",
    "hands",
    "left_hand",
    "right_hand",
    "person",
    "human",
    "agent",
    "body",
    "arm",
    "finger",
    "fingers",
}


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
    "at",
    "is",
    "are",
    "be",
    "by",
    "for",
    "with",
    "then",
    "into",
    "from",
    "that",
    "this",
    "it",
    "as",
    "after",
    "before",
    "when",
    "while",
    "so",
    "can",
    "cannot",
    "not",
    "no",
    "yes",
    "will",
    "would",
    "should",
    "could",
    "must",
    "may",
    "might",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "have",
    "has",
    "had",
}


_VERBISH_TOKENS = {
    "aligned",
    "contact",
    "contacting",
    "supports",
    "support",
    "supported",
    "stabilize",
    "stabilized",
    "stabilizing",
    "position",
    "positioned",
    "place",
    "placed",
    "placing",
    "remove",
    "removed",
    "removing",
    "open",
    "opened",
    "opening",
    "close",
    "closed",
    "closing",
    "tilt",
    "tilted",
    "tilting",
    "pour",
    "pours",
    "pouring",
    "lift",
    "lifted",
    "lifting",
    "rotate",
    "rotates",
    "rotating",
    "hold",
    "holds",
    "holding",
    "begin",
    "begins",
    "beginning",
    "start",
    "starts",
    "starting",
}

_DISTRACTOR_OBJECTS = [
    "microwave",
    "dish_soap",
    "blender",
    "hammer",
    "screwdriver",
    "toothbrush",
    "laptop",
    "phone",
    "remote_control",
    "shampoo_bottle",
    "shoe",
    "book",
]


_RELATION_TOKENS = {
    "on_top_of",
    "inside",
    "inside_of",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "above",
    "below",
    "next_to",
    "relative_to",
    "separated_from",
    "connected_to",
    "disconnected_from",
    "aligned_with",
    "tilted_toward",
    "centered_on",
    "filled_with",
    "covered_by",
    "supported_by",
    "stabilized_on",
    "contacting",
}


_GENERIC_PART_TOKENS = {
    "opening",
    "mouth",
    "rim",
    "edge",
    "surface",
    "point",
    "region",
    "area",
    "contact",
    "grip",
    "handle",
}


@dataclass(frozen=True)
class Sample:
    task_name: str
    evidence_type: str
    image: List[str]
    video: Optional[str]
    question: str
    answer: str
    source_path: str
    llm_fields: Optional[Dict[str, Any]] = None


@dataclass
class ApiConfig:
    api_key: str = os.environ.get("API_KEY", "EMPTY")
    api_base_url: str = os.environ.get("API_BASE_URL", "http://model.mify.ai.srv/v1")
    model_provider_id: str = os.environ.get("MODEL_PROVIDER_ID", "vertex_ai")
    model_name: str = os.environ.get("MODEL_NAME", "gemini-3-pro-preview")
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "4096"))
    request_images_limit: int = int(os.environ.get("REQUEST_IMAGES_LIMIT", "1000000"))
    max_retries: int = int(os.environ.get("MAX_RETRIES", "3"))
    retry_backoff_sec: float = float(os.environ.get("RETRY_BACKOFF_SEC", "1.5"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.3"))


def initialize_api_client(cfg: ApiConfig) -> Any:
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.api_base_url,
            default_headers={"X-Model-Provider-Id": cfg.model_provider_id},
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"OpenAI-compatible client init failed: {e}. Falling back to non-API mode.")
        return None


SYSTEM_PROMPT = """You are an expert Embodied AI Analyst and Physics Consultant.
Your task is to synthesize structured data fields into high-quality, natural language answers for a QA dataset.

### Core Objectives:
1. Naturalness: Do not just list the fields. Weave the data into fluent, professional English with logical connectors.
2. Strict Grounding: The output must be based ONLY on the provided Input Data. Do not hallucinate details.
3. Detail & Rigor: Preserve all technical details provided in the input fields. Do not simplify if it loses precision.
4. Professional Tone: Keep the language objective and academic. Avoid conversational fillers.
5. Follow Constraints: Obey the task-specific constraints exactly (e.g., one sentence, no newlines, required phrases).

### Additional Constraints (for dataset safety):
- Do NOT mention any filenames, paths, extensions, timestamps, or frame numbers.
- Do NOT output markdown, code fences, or bullet lists unless explicitly required by the task.

### Output Format:
- Return ONLY the final answer text.
"""


TWO_PARAGRAPH_TASKS: set[str] = set()
SINGLE_SENTENCE_TASKS = {TASK_10, TASK_14}


LLM_ANSWER_PROMPTS: Dict[str, str] = {
    TASK_08: """Input Data:
Step Goal: {step_goal}
Hotspot Description: {hotspot_description}
Affordance Type: {affordance_type}
Mechanism: {mechanism}

Instruction: The user asks: "Locate the interaction hotspot area first, then describe its affordance_type and mechanism."
Synthesize the fields into a single, natural paragraph. Explicitly mention the hotspot region using the Hotspot Description.""",
    TASK_10: """Input Data:
High-Level Goal: {high_level_goal}
Step Goal: {step_goal}
Agent: {agent}
Action: {action}
Patient: {patient}
Spatial Preconditions: {spatial_preconditions}
Affordance Preconditions: {affordance_preconditions}
Hotspot Description: {hotspot_description}
Affordance Type: {affordance_type}
Mechanism: {mechanism}
Spatial Effects: {spatial_effects}
Affordance Effects: {affordance_effects}

Instruction: The user asks: "Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects. Answer in one English sentence."
Write ONE long English sentence (no newlines) that forms a causal loop:
preconditions → action/interaction mechanism → immediate effects.
If a required detail is not supported by a single keyframe or is an internal/latent property, explicitly use the exact phrase: "not directly observable".
Do NOT use bullet points, numbering, or multiple sentences.""",
    TASK_11: """Input Data:
High-Level Goal: {high_level_goal}
Step Goal: {step_goal}
Rationale: {rationale}

Instruction: The user asks: "Why is this step necessary for the overall goal?"
Write a concise justification grounded in the rationale and explicitly link it to the high-level goal.""",
    TASK_17: """Input Data:
High-Level Goal: {high_level_goal}
Previous Step Goal: {prev_step_goal}
Previous Step Effects: {prev_step_effects}
Next Step Goal: {next_step_goal}
Next Step Preconditions: {next_step_preconditions}

Instruction: The user asks: "How does the outcome of the previous step satisfy the preconditions for the next step?"
Explain the dependency in 1–2 sentences by connecting effects to preconditions. Do not add extra steps.""",
}


LLM_REASON_PROMPTS: Dict[str, str] = {
    TASK_22: """Input Data:
High-Level Goal: {high_level_goal}
Bad Plan Steps (proposed): {bad_plan_steps}
Gold Plan Steps (reference): {gold_plan_steps}
Flaw Type: {flaw_type}
Flawed Step (1-based within bad plan): {flaw_step}

Instruction: Provide ONE concise sentence explaining why the flawed step is incorrect, grounded in missing preconditions or goal mismatch.
Be specific: name the missing/out-of-order prerequisite step(s) when possible using the provided plan steps.
Return ONLY the reason sentence, without labels or numbering.""",
    TASK_26: """Input Data:
Step Goal: {step_goal}
Failure Reason: {failure_reason}
Recovery Strategy (must be followed): {recovery_strategy}

Instruction: Explain briefly (1-2 sentences) why the recovery strategy helps, grounded in spatial stability and affordance/mechanism.
Explicitly connect the Failure Reason to what spatial/affordance condition the strategy restores.
Return ONLY the explanation text (do NOT restate the recovery strategy).""",
}

DEFAULT_LLM_TASKS: Tuple[str, ...] = tuple(sorted(set(LLM_ANSWER_PROMPTS.keys()) | set(LLM_REASON_PROMPTS.keys())))


def _sanitize_text_single_line(text: str) -> str:
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_answer(task_name: str, text: str) -> str:
    if not text:
        return ""
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"```[a-zA-Z]*\s*", "", s)
    s = s.replace("```", "")
    s = re.sub(r"(?m)^\s*([\-\*•\>]+|\d+[\.\)])\s+", "", s)
    s = s.strip()
    if task_name in TWO_PARAGRAPH_TASKS:
        parts = re.split(r"\n\s*\n", s)
        parts = [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts[0] + "\n\n" + "\n\n".join(parts[1:2])
        return _sanitize_space(s)
    return _sanitize_text_single_line(s)


def _defluff_text(text: str) -> str:
    s = str(text or "")
    patterns = [
        r"^\s*(In summary|In conclusion|To summarize|Overall|In general|Generally),\s*",
        r"^\s*(In this (scene|image|frame|step)),\s*",
        r"^\s*(It should be noted that|Note that)\s*",
        r"^\s*(Here is the answer|Answer)\s*[:\-]\s*",
    ]
    for pat in patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class TwoStageLlm:
    def __init__(self, cfg: ApiConfig):
        self.cfg = cfg
        self.client = initialize_api_client(cfg)

    def enabled(self) -> bool:
        return self.client is not None and (self.cfg.api_key or "EMPTY") != "EMPTY"

    def call(self, *, system_prompt: str, user_text: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        if not self.enabled():
            return ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(max(1, int(self.cfg.max_retries))):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=messages,
                    max_tokens=max_tokens or self.cfg.max_tokens,
                    temperature=float(self.cfg.temperature if temperature is None else temperature),
                    top_p=0.9,
                    presence_penalty=0,
                )
                dt = time.time() - t0
                if not (resp and getattr(resp, "choices", None)):
                    raise RuntimeError("Empty response or missing choices")
                choice = resp.choices[0]
                out = getattr(getattr(choice, "message", None), "content", "") or ""
                logger.info(f">>> [LLM] ok dt={dt:.2f}s len={len(out)}")
                return out
            except Exception as e:  # pragma: no cover
                last_err = e
                logger.warning(f">>> [LLM] failed attempt={attempt+1}: {e}")
                if attempt + 1 < max(1, int(self.cfg.max_retries)):
                    time.sleep(float(self.cfg.retry_backoff_sec) * (attempt + 1))
        if last_err is not None:
            logger.error(f">>> [LLM] final failure: {last_err}")
        return ""

    def generate_answer(self, *, task_name: str, fields: Dict[str, Any], draft_answer: str, two_pass: bool) -> str:
        prompt = LLM_ANSWER_PROMPTS.get(task_name)
        if not prompt:
            return draft_answer
        user_text = prompt.format(**fields)
        raw = self.call(system_prompt=SYSTEM_PROMPT, user_text=user_text)
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            return draft_answer
        if not two_pass:
            out = _defluff_text(_sanitize_answer(task_name, raw))
            if task_name in SINGLE_SENTENCE_TASKS:
                out = _enforce_single_sentence(out)
            return out

        polish_user = (
            "### INPUT DATA (Ground Truth):\n"
            + json.dumps(fields, ensure_ascii=False)
            + "\n\n### DRAFT ANSWER (For Reference):\n"
            + raw
            + "\n\n### INSTRUCTIONS:\n"
            "Rewrite the draft to improve fluency while staying strictly grounded in the Input Data.\n"
            "Do NOT add any details not in the Input Data.\n"
            "Do NOT mention filenames, timestamps, or frame numbers.\n"
            "Do NOT use bullet points or lists.\n"
            + (
                "Produce exactly TWO paragraphs.\n"
                if task_name in TWO_PARAGRAPH_TASKS
                else (
                    "Produce exactly ONE sentence (no newlines).\n"
                    if task_name in SINGLE_SENTENCE_TASKS
                    else "Produce a single paragraph.\n"
                )
            )
            + "\n### POLISHED OUTPUT:"
        )
        polished = self.call(system_prompt=SYSTEM_PROMPT, user_text=polish_user, temperature=0.2)
        polished = polished.strip() if isinstance(polished, str) else ""
        if not polished:
            polished = raw
        out = _defluff_text(_sanitize_answer(task_name, polished))
        if task_name in SINGLE_SENTENCE_TASKS:
            out = _enforce_single_sentence(out)
        return out

    def generate_reason_only(self, *, task_name: str, fields: Dict[str, Any], two_pass: bool) -> str:
        prompt = LLM_REASON_PROMPTS.get(task_name)
        if not prompt:
            return ""
        user_text = prompt.format(**fields)
        raw = self.call(system_prompt=SYSTEM_PROMPT, user_text=user_text)
        raw = raw.strip() if isinstance(raw, str) else ""
        if not raw:
            return ""
        if not two_pass:
            return _defluff_text(_sanitize_text_single_line(raw))

        polish_user = (
            "### INPUT DATA (Ground Truth):\n"
            + json.dumps(fields, ensure_ascii=False)
            + "\n\n### DRAFT TEXT (For Reference):\n"
            + raw
            + "\n\n### INSTRUCTIONS:\n"
            "Rewrite the draft into ONE concise sentence, strictly grounded in the Input Data.\n"
            "Do NOT mention filenames, timestamps, or frame numbers.\n"
            "Return ONLY the sentence.\n\n"
            "### POLISHED OUTPUT:"
        )
        polished = self.call(system_prompt=SYSTEM_PROMPT, user_text=polish_user, temperature=0.2)
        polished = polished.strip() if isinstance(polished, str) else ""
        return _defluff_text(_sanitize_text_single_line(polished or raw))


def _apply_llm(samples: List[Sample], llm: TwoStageLlm, llm_tasks: set[str], *, two_pass: bool) -> List[Sample]:
    if not samples or not llm_tasks or not llm.enabled():
        return samples

    out: List[Sample] = []
    for s in samples:
        if s.task_name not in llm_tasks:
            out.append(s)
            continue
        fields = s.llm_fields or {}
        if not fields:
            out.append(s)
            continue

        original_answer = s.answer
        new_answer = original_answer
        try:
            if s.task_name == TASK_22:
                reason = llm.generate_reason_only(task_name=TASK_22, fields=fields, two_pass=two_pass)
                if reason:
                    new_answer = f"FlawStep={fields.get('flaw_step')}; FlawType={fields.get('flaw_type')}; Reason={reason}"
            elif s.task_name == TASK_26:
                expl = llm.generate_reason_only(task_name=TASK_26, fields=fields, two_pass=two_pass)
                if expl:
                    strat = str(fields.get("recovery_strategy") or "").strip()
                    if strat and not strat.endswith((".", "!", "?")):
                        strat = strat + "."
                    new_answer = _sanitize_space(f"{strat} {expl}")
            else:
                new_answer = llm.generate_answer(task_name=s.task_name, fields=fields, draft_answer=original_answer, two_pass=two_pass)
        except Exception as e:  # pragma: no cover
            logger.warning(f"LLM postprocess failed for task={s.task_name}: {e}")
            new_answer = original_answer

        if _has_frame_leak(new_answer):
            new_answer = original_answer

        out.append(
            Sample(
                task_name=s.task_name,
                evidence_type=s.evidence_type,
                image=s.image,
                video=s.video,
                question=s.question,
                answer=new_answer,
                source_path=s.source_path,
                llm_fields=s.llm_fields,
            )
        )
    return out


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_final_plan_schema(plan: Dict[str, Any], *, source: str, strict: bool) -> None:
    problems: List[str] = []

    def _add(msg: str) -> None:
        if len(problems) < 80:
            problems.append(msg)

    def _is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _check_exact_keys(obj: Any, *, allowed: set[str], path: str) -> None:
        if not strict or not isinstance(obj, dict):
            return
        extra = sorted([k for k in obj.keys() if k not in allowed])
        if extra:
            _add(f"{path} has extra keys: {extra}")
        missing = sorted([k for k in allowed if k not in obj])
        if missing:
            _add(f"{path} missing keys: {missing}")

    def _require_obj(obj: Any, path: str) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            _add(f"{path} must be an object")
            return {}
        return obj

    def _require_list(obj: Any, path: str) -> List[Any]:
        if not isinstance(obj, list):
            _add(f"{path} must be a list")
            return []
        return obj

    def _require_str_field(d: Dict[str, Any], key: str, path: str) -> str:
        if strict and key not in d:
            _add(f"{path}.{key} missing")
            return ""
        v = d.get(key)
        if key in d and not isinstance(v, str):
            _add(f"{path}.{key} must be a string")
            return ""
        return v.strip() if isinstance(v, str) else ""

    plan_obj = _require_obj(plan, "top")
    _check_exact_keys(plan_obj, allowed={"high_level_goal", "steps"}, path="top")

    hl = plan_obj.get("high_level_goal")
    if not isinstance(hl, str):
        _add("top.high_level_goal must be a string")

    steps = _require_list(plan_obj.get("steps"), "top.steps")

    allowed_step_keys = {
        "step_id",
        "step_goal",
        "rationale",
        "causal_chain",
        "counterfactual_challenge_question",
        "expected_challenge_outcome",
        "failure_reflecting",
        "critical_frames",
    }
    allowed_step_cc_keys = {
        "agent",
        "action",
        "patient",
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",
    }
    allowed_failure_keys = {"reason", "recovery_strategy"}
    allowed_cf_keys = {"frame_index", "action_state_change_description", "causal_chain", "interaction"}
    allowed_frame_cc_keys = {
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",
    }
    allowed_interaction_keys = {"description", "affordance_type", "mechanism"}

    step_ids: List[int] = []
    for idx, st_any in enumerate(steps):
        path = f"steps[{idx}]"
        if not isinstance(st_any, dict):
            _add(f"{path} must be an object")
            continue
        st = st_any
        _check_exact_keys(st, allowed=allowed_step_keys, path=path)

        sid = st.get("step_id")
        if not _is_int(sid):
            _add(f"{path}.step_id must be int")
        else:
            if int(sid) <= 0:
                _add(f"{path}.step_id must be >= 1")
            step_ids.append(int(sid))

        _require_str_field(st, "step_goal", path)
        _require_str_field(st, "rationale", path)
        q_cf = _require_str_field(st, "counterfactual_challenge_question", path)
        _require_str_field(st, "expected_challenge_outcome", path)
        if strict and q_cf and not re.match(r"^\s*What\s+if\b", q_cf, flags=re.IGNORECASE):
            _add(f"{path}.counterfactual_challenge_question must start with 'What if'")

        cc = _require_obj(st.get("causal_chain"), f"{path}.causal_chain")
        _check_exact_keys(cc, allowed=allowed_step_cc_keys, path=f"{path}.causal_chain")
        for k in sorted(allowed_step_cc_keys):
            _require_str_field(cc, k, f"{path}.causal_chain")

        fr = _require_obj(st.get("failure_reflecting"), f"{path}.failure_reflecting")
        _check_exact_keys(fr, allowed=allowed_failure_keys, path=f"{path}.failure_reflecting")
        for k in sorted(allowed_failure_keys):
            _require_str_field(fr, k, f"{path}.failure_reflecting")

        cfs = st.get("critical_frames")
        if not isinstance(cfs, list):
            _add(f"{path}.critical_frames must be a list")
            continue
        if strict and len(cfs) != 2:
            _add(f"{path}.critical_frames must have length 2 (got {len(cfs)})")

        fi0: Optional[int] = None
        fi1: Optional[int] = None
        for j, cf_any in enumerate(cfs):
            cf_path = f"{path}.critical_frames[{j}]"
            if not isinstance(cf_any, dict):
                _add(f"{cf_path} must be an object")
                continue
            cf = cf_any
            _check_exact_keys(cf, allowed=allowed_cf_keys, path=cf_path)

            fi = cf.get("frame_index")
            if not _is_int(fi):
                _add(f"{cf_path}.frame_index must be int")
            else:
                if int(fi) <= 0:
                    _add(f"{cf_path}.frame_index must be >= 1")
                if j == 0:
                    fi0 = int(fi)
                elif j == 1:
                    fi1 = int(fi)

            _require_str_field(cf, "action_state_change_description", cf_path)

            fcc = _require_obj(cf.get("causal_chain"), f"{cf_path}.causal_chain")
            _check_exact_keys(fcc, allowed=allowed_frame_cc_keys, path=f"{cf_path}.causal_chain")
            for k in sorted(allowed_frame_cc_keys):
                _require_str_field(fcc, k, f"{cf_path}.causal_chain")

            intr = _require_obj(cf.get("interaction"), f"{cf_path}.interaction")
            _check_exact_keys(intr, allowed=allowed_interaction_keys, path=f"{cf_path}.interaction")
            for k in sorted(allowed_interaction_keys):
                _require_str_field(intr, k, f"{cf_path}.interaction")

        if strict and fi0 is not None and fi1 is not None:
            if fi0 == fi1:
                _add(f"{path}.critical_frames frame_index must be distinct (got {fi0} and {fi1})")
            if fi0 > fi1:
                _add(f"{path}.critical_frames must be in increasing time order (frame_index {fi0} then {fi1})")

    if strict and step_ids:
        dup = sorted({sid for sid in step_ids if step_ids.count(sid) > 1})
        if dup:
            _add(f"steps.step_id must be unique (duplicates: {dup})")

    if problems:
        raise ValueError(f"Final schema validation failed: source={source} problems=" + " | ".join(problems[:10]))


def _has_frame_leak(text: str) -> bool:
    s = str(text or "")
    for pat in FRAME_LEAK_PATTERNS:
        if pat.search(s):
            return True
    return False


def _sanitize_space(text: str) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_key_moment_prefix(text: str) -> str:
    s = str(text or "").strip()
    s = _KEY_MOMENT_PREFIX_RE.sub("", s)
    return s.strip()


def _split_numbered_block(text: str) -> List[str]:
    raw = str(text or "").replace("\\n", "\n").strip()
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*\d+\s*[\.\)、]\s*", "", ln).strip()
        if ln:
            out.append(ln)
    if not out and raw:
        out = [raw]
    return out


def _pick_first_points(text: str, max_points: int) -> str:
    pts = _split_numbered_block(text)
    if not pts:
        return ""
    return " ".join(pts[: max(1, int(max_points))]).strip()


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _parse_spatial_relations(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            out.append(item)
    return out


def _parse_affordance_states(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            out.append(item)
    return out


def _relation_phrase(token: str) -> str:
    return str(token or "").strip().replace("_", " ")

def _coerce_truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0", ""):
            return False
    return True


def _format_spatial_relation(rel: Dict[str, Any]) -> str:
    relation = _relation_phrase(rel.get("relation", ""))
    if not relation:
        return ""
    objects = rel.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    objects = [str(o).strip() for o in objects if isinstance(o, (str, int, float)) and str(o).strip()]
    truth_bool = _coerce_truth(rel.get("truth", True))

    if not objects:
        base = f"the relevant objects are {relation}"
    elif len(objects) == 1:
        base = f"{objects[0]} is {relation}"
    else:
        base = f"{objects[0]} is {relation} " + " and ".join(objects[1:])

    if not truth_bool:
        base = base.replace(" is ", " is not ", 1)
    return base.strip()


def _format_affordance_state(st: Dict[str, Any]) -> str:
    obj = str(st.get("object_name", "") or "").strip()
    affs = st.get("affordance_types", [])
    if not isinstance(affs, list):
        affs = []
    affs = [str(a).strip() for a in affs if isinstance(a, (str, int, float)) and str(a).strip()]
    reasons = str(st.get("reasons", "") or "").strip()
    if not obj and not affs:
        return ""
    if obj and affs:
        base = f"{obj} has affordance/state " + ", ".join(affs)
    elif obj:
        base = f"{obj} has the required affordance/state"
    else:
        base = "the object has affordance/state " + ", ".join(affs)
    if reasons:
        base = f"{base} because {reasons}"
    return base.strip()


def _format_affordance_state_compact(st: Dict[str, Any]) -> str:
    obj = str(st.get("object_name", "") or "").strip()
    affs = st.get("affordance_types", [])
    if not isinstance(affs, list):
        affs = []
    affs = [str(a).strip() for a in affs if isinstance(a, (str, int, float)) and str(a).strip()]
    if obj and affs:
        return f"{obj} " + ", ".join(affs)
    if obj:
        return obj
    return ", ".join(affs)


def _format_spatial(value: Any, *, max_items: int) -> str:
    if isinstance(value, str):
        return _pick_first_points(value, max_items)
    rels = _parse_spatial_relations(value)
    phrases = [_format_spatial_relation(r) for r in rels]
    phrases = [p for p in phrases if p]
    return " ".join(phrases[: max(1, int(max_items))]).strip()


def _format_affordance(value: Any, *, max_items: int) -> str:
    if isinstance(value, str):
        return _pick_first_points(value, max_items)
    sts = _parse_affordance_states(value)
    phrases = [_format_affordance_state(s) for s in sts]
    phrases = [p for p in phrases if p]
    return " ".join(phrases[: max(1, int(max_items))]).strip()


def _terms_from_spatial(value: Any) -> set[str]:
    if isinstance(value, str):
        return _normalize_terms(value)
    out: set[str] = set()
    for rel in _parse_spatial_relations(value):
        rel_token = rel.get("relation")
        if isinstance(rel_token, str) and rel_token.strip():
            out.add(rel_token.strip().lower())
        objs = rel.get("objects", [])
        if isinstance(objs, list):
            for o in objs:
                if isinstance(o, str) and o.strip():
                    out.add(o.strip().lower())
    out -= _GENERIC_OBJECT_TOKENS
    return out


def _terms_from_affordance(value: Any) -> set[str]:
    if isinstance(value, str):
        return _normalize_terms(value)
    out: set[str] = set()
    for st in _parse_affordance_states(value):
        obj = st.get("object_name")
        if isinstance(obj, str) and obj.strip():
            out.add(obj.strip().lower())
        affs = st.get("affordance_types", [])
        if isinstance(affs, list):
            for a in affs:
                if isinstance(a, str) and a.strip():
                    out.add(a.strip().lower())
    out -= _GENERIC_OBJECT_TOKENS
    return out

def _parse_timestamp_from_path(path: str) -> Optional[float]:
    m = _TS_RE.search(os.path.basename(path or ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_relpath(path: str, root: str) -> str:
    ap = os.path.abspath(path)
    ar = os.path.abspath(root)
    try:
        rel = os.path.relpath(ap, ar)
        return rel.replace("\\", "/")
    except Exception:
        return ap.replace("\\", "/")


def _list_item_dirs(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "causal_plan_with_keyframes.json" in filenames:
            out.append(dirpath)
            dirnames[:] = []
    return sorted(out)


def _list_sampled_frames(item_dir: str) -> List[str]:
    dirs = [
        os.path.join(item_dir, "sampled_frames"),
        os.path.join(item_dir, "stage1", "sampled_frames"),
    ]
    patterns = [
        "sample_*.jpg",
        "sample_*.jpeg",
        "sample_*.png",
    ]
    for d in dirs:
        paths: List[str] = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(d, pat)))
        if paths:
            return sorted(paths)
    return []


def _pick_uniform(frames: Sequence[str], k: int) -> List[str]:
    n = len(frames)
    if n == 0:
        return []
    k = max(1, int(k))
    if n <= k:
        return list(frames)
    if k == 1:
        return [frames[n // 2]]
    idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    uniq = []
    for i in idxs:
        if not uniq or uniq[-1] != i:
            uniq.append(i)
    return [frames[i] for i in uniq]


def _pick_head_tail(frames: Sequence[str], head: int, tail: int) -> List[str]:
    head = max(0, int(head))
    tail = max(0, int(tail))
    if not frames:
        return []
    if head + tail <= 0:
        return []
    if len(frames) <= head + tail:
        return list(frames)
    return list(frames[:head]) + list(frames[-tail:])


def _resolve_video_prefix(item_dir: str, step_id: int) -> Optional[str]:
    cands = [
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}_last.mp4"),
        os.path.join(item_dir, "cumulative_last_frame_segments", f"segment_start_to_step{step_id:02d}.mp4"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _resolve_video_clip(item_dir: str, step_id: int) -> Optional[str]:
    if step_id <= 0:
        return None
    cands: List[str] = []
    if step_id == 1:
        cands.append(os.path.join(item_dir, "last_frame_segments", "segment_start_to_step01.mp4"))
        cands.append(os.path.join(item_dir, "last_frame_segments", "segment_start_to_step1.mp4"))
    else:
        cands.append(os.path.join(item_dir, "last_frame_segments", f"segment_step{step_id - 1:02d}_to_step{step_id:02d}.mp4"))
        cands.append(os.path.join(item_dir, "last_frame_segments", f"segment_step{step_id - 1}_to_step{step_id}.mp4"))
    for p in cands:
        if os.path.exists(p):
            return p

    # Three-stage fallback: `stage2/step_segments.json` (preferred) or `stage2/step_clips/stepXX_*.mp4`.
    seg_path = os.path.join(item_dir, "stage2", "step_segments.json")
    if os.path.exists(seg_path):
        try:
            seg_json = _read_json(seg_path)
            segments = seg_json.get("segments", [])
            if isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict):
                        continue
                    sid = seg.get("step_id")
                    try:
                        sid_int = int(sid)
                    except Exception:
                        continue
                    if sid_int != int(step_id):
                        continue
                    clip_rel = seg.get("clip_relpath")
                    if isinstance(clip_rel, str) and clip_rel.strip():
                        cand = os.path.join(item_dir, "stage2", clip_rel.strip())
                        if os.path.exists(cand):
                            return cand
        except Exception:
            pass

    clips_dir = os.path.join(item_dir, "stage2", "step_clips")
    if os.path.isdir(clips_dir):
        matches = sorted(glob.glob(os.path.join(clips_dir, f"step{step_id:02d}_*.mp4")))
        if matches:
            return matches[0]
    return None


def _find_keyframe_image(item_dir: str, step_id: int, frame_index: int) -> Optional[str]:
    step_prefix = os.path.join(item_dir, f"{step_id:02d}_*")
    pats: List[str] = []
    for ext in ("jpg", "jpeg", "png"):
        pats.append(os.path.join(step_prefix, f"frame_{frame_index:03d}_ts_*.{ext}"))
        pats.append(os.path.join(step_prefix, f"frame_{frame_index:03d}_*.{ext}"))
    for pat in pats:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[0]
    return None


def _extract_snake_case_objects(plan: Dict[str, Any]) -> List[str]:
    tokens: set[str] = set()
    for step in plan.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        cc = step.get("causal_chain") or {}
        if isinstance(cc, dict):
            for k in ("agent", "patient"):
                v = cc.get(k)
                if isinstance(v, str):
                    tokens |= set(_SNAKE_TOKEN_RE.findall(v))
            for k in (
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ):
                v = cc.get(k)
                if isinstance(v, str):
                    tokens |= set(_SNAKE_TOKEN_RE.findall(v))
        for cf in step.get("critical_frames", []) or []:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction") or {}
            if isinstance(intr, dict):
                for k in ("description", "affordance_type", "mechanism"):
                    v = intr.get(k)
                    if isinstance(v, str):
                        tokens |= set(_SNAKE_TOKEN_RE.findall(v))
            fcc = cf.get("causal_chain") or {}
            if isinstance(fcc, dict):
                for k in (
                    "causal_precondition_on_spatial",
                    "causal_precondition_on_affordance",
                    "causal_effect_on_spatial",
                    "causal_effect_on_affordance",
                ):
                    v = fcc.get(k)
                    if isinstance(v, str):
                        tokens |= set(_SNAKE_TOKEN_RE.findall(v))
    cleaned = [t for t in tokens if t not in _GENERIC_OBJECT_TOKENS]
    cleaned.sort()
    return cleaned


def _extract_key_objects_for_task02(plan: Dict[str, Any]) -> List[str]:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    if not steps:
        return []

    # Detect schema: list-based (SpatialRelation/AffordanceState + interaction.hotspot) vs legacy string-based.
    is_list_schema = False
    for st in steps:
        cc = st.get("causal_chain")
        if isinstance(cc, dict) and isinstance(cc.get("causal_precondition_on_spatial"), list):
            is_list_schema = True
            break
        for cf in st.get("critical_frames", []) or []:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction")
            if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict):
                is_list_schema = True
                break
        if is_list_schema:
            break

    if is_list_schema:
        objs: set[str] = set()

        def _add_obj(x: Any) -> None:
            if not isinstance(x, str):
                return
            t = x.strip()
            if not t:
                return
            if t.lower() in _GENERIC_OBJECT_TOKENS:
                return
            objs.add(t)

        def _add_from_spatial(val: Any) -> None:
            for rel in _parse_spatial_relations(val):
                for o in rel.get("objects", []) if isinstance(rel.get("objects", []), list) else []:
                    _add_obj(str(o))

        def _add_from_affordance(val: Any) -> None:
            for stt in _parse_affordance_states(val):
                _add_obj(stt.get("object_name"))

        for st in steps:
            cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
            if isinstance(cc, dict):
                _add_obj(cc.get("patient"))
                _add_obj(cc.get("agent"))
                _add_from_spatial(cc.get("causal_precondition_on_spatial"))
                _add_from_spatial(cc.get("causal_effect_on_spatial"))
                _add_from_affordance(cc.get("causal_precondition_on_affordance"))
                _add_from_affordance(cc.get("causal_effect_on_affordance"))

            for cf in st.get("critical_frames", []) or []:
                if not isinstance(cf, dict):
                    continue
                fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                if isinstance(fcc, dict):
                    _add_obj(fcc.get("patient"))
                    _add_obj(fcc.get("agent"))
                    _add_from_spatial(fcc.get("causal_precondition_on_spatial"))
                    _add_from_spatial(fcc.get("causal_effect_on_spatial"))
                    _add_from_affordance(fcc.get("causal_precondition_on_affordance"))
                    _add_from_affordance(fcc.get("causal_effect_on_affordance"))

                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                if isinstance(intr, dict):
                    for t in intr.get("tools", []) if isinstance(intr.get("tools", []), list) else []:
                        _add_obj(str(t))
                    for m in intr.get("materials", []) if isinstance(intr.get("materials", []), list) else []:
                        _add_obj(str(m))

        tokens = sorted(objs)
        if len(tokens) > 12:
            tokens = tokens[:12]
        return tokens

    # Final schema: string causal_* fields (numbered statements).
    # Prefer object identifiers from step-level `patient` and additional snake_case objects;
    # avoid affordance/state tokens and generic verbs.
    patient_pool: set[str] = set()
    for st in steps:
        cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        if not isinstance(cc, dict):
            continue
        pat = cc.get("patient")
        if isinstance(pat, str) and pat.strip():
            t = pat.strip()
            if t.lower() not in _GENERIC_OBJECT_TOKENS:
                patient_pool.add(t)

    affordance_types: set[str] = set()
    for st in steps:
        for cf in st.get("critical_frames", []) or []:
            if not isinstance(cf, dict):
                continue
            intr = cf.get("interaction")
            if isinstance(intr, dict):
                v = intr.get("affordance_type")
                if isinstance(v, str) and v.strip():
                    affordance_types.add(v.strip().lower())

    objs = set(patient_pool) | set(_extract_snake_case_objects(plan))

    def _is_object_like(token: str) -> bool:
        t = (token or "").strip()
        if not t:
            return False
        tl = t.lower()
        if tl in _STOPWORDS:
            return False
        if tl in _GENERIC_OBJECT_TOKENS or tl in _VERBISH_TOKENS:
            return False
        if tl in affordance_types:
            return False
        if tl in _RELATION_TOKENS or tl in _GENERIC_PART_TOKENS:
            return False
        if tl.startswith(("ready_to_", "partially_", "more_", "less_", "switched_")):
            return False
        if tl.endswith(("_on", "_off", "_open", "_closed", "_pressed", "_depressed")):
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t):
            return False
        # Heuristic: allow single-word identifiers only when they came from patient fields.
        if "_" not in t and t not in patient_pool:
            return False
        return True

    tokens = [t for t in objs if _is_object_like(t)]
    tokens = sorted(set(tokens))
    if len(tokens) > 12:
        tokens = tokens[:12]
    return tokens


def _stable_int_seed(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _normalize_terms(text: str) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
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
        "at",
        "is",
        "are",
        "be",
        "by",
        "for",
        "with",
        "then",
        "into",
        "from",
        "that",
        "this",
        "it",
        "as",
        "after",
        "before",
        "when",
        "while",
    }
    out = {t for t in tokens if t not in stop and len(t) >= 3}
    out -= _GENERIC_OBJECT_TOKENS
    return out


def _has_dependency(prev_effects: Any, next_preconds: Any) -> bool:
    eff = _terms_from_spatial(prev_effects) | _terms_from_affordance(prev_effects)
    pre = _terms_from_spatial(next_preconds) | _terms_from_affordance(next_preconds)
    if not eff or not pre:
        return False
    if eff & pre:
        return True
    for e in eff:
        for p in pre:
            if e in p or p in e:
                return True
    return False


def _sharegpt_entry(sample: Sample) -> Dict[str, Any]:
    evidence_files: List[str] = list(sample.image)
    if sample.video:
        evidence_files.append(sample.video)
    return {
        "id": str(uuid.uuid4()),
        "image": sample.image,
        **({"video": sample.video} if sample.video else {}),
        "conversations": [
            {"from": "human", "value": sample.question},
            {"from": "gpt", "value": sample.answer},
        ],
        "meta": {
            "task_name": sample.task_name,
            "evidence_type": sample.evidence_type,
            "source_path": sample.source_path,
            "evidence_files": evidence_files,
        },
    }


def _write_jsonl(out_path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _sorted_steps(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    def _to_int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0
    return sorted(steps, key=lambda s: _to_int(s.get("step_id")))


def _require_str(d: Dict[str, Any], key: str) -> str:
    v = d.get(key)
    return v.strip() if isinstance(v, str) else ""


def _frame_index(cf: Dict[str, Any]) -> Optional[int]:
    try:
        v = cf.get("frame_index")
        return int(v)
    except Exception:
        return None


def _make_task01(item_dir: str, plan: Dict[str, Any], input_root: str, *, uniform_k: int) -> Optional[Sample]:
    steps = _sorted_steps(plan)
    if not steps:
        return None
    last_step_id = int(steps[-1].get("step_id", 0) or 0)
    video = _resolve_video_prefix(item_dir, last_step_id)
    hl = _require_str(plan, "high_level_goal")
    if not hl:
        return None

    q = "Based on the full video, what is the most appropriate high-level goal?"
    a = hl

    if video:
        evidence_type = EVIDENCE_PREFIX
        images: List[str] = []
        video_rel = _safe_relpath(video, input_root)
    else:
        sampled = _list_sampled_frames(item_dir)
        if not sampled:
            return None
        evidence_type = EVIDENCE_UNIFORM
        imgs = _pick_uniform(sampled, uniform_k)
        if not imgs:
            return None
        images = [_safe_relpath(p, input_root) for p in imgs]
        video_rel = None
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    return Sample(task_name=TASK_01, evidence_type=evidence_type, image=images, video=video_rel, question=q, answer=a, source_path=source_rel)


def _make_task02(item_dir: str, plan: Dict[str, Any], input_root: str, uniform_k: int, rng: random.Random) -> Optional[Sample]:
    hl = _require_str(plan, "high_level_goal")
    if not hl:
        return None
    sampled = _list_sampled_frames(item_dir)
    if not sampled:
        return None
    imgs = _pick_uniform(sampled, uniform_k)
    if not imgs:
        return None
    label_objs = _extract_key_objects_for_task02(plan)
    if not label_objs:
        return None
    # Candidate list = label + distractors (fill to 10-14 when possible).
    candidates = list(label_objs)
    distractors = list(_DISTRACTOR_OBJECTS)
    rng.shuffle(distractors)
    for d in distractors:
        if d not in candidates:
            candidates.append(d)
        if len(candidates) >= max(10, min(14, len(label_objs) + 6)):
            break
    rng.shuffle(candidates)

    q = (
        f'High-level goal: "{hl}" From the candidate objects {json.dumps(candidates)}, '
        "list the key objects that are directly relevant to the goal and will be used for planning."
    )
    a = json.dumps(label_objs)

    images = [_safe_relpath(p, input_root) for p in imgs]
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    return Sample(task_name=TASK_02, evidence_type=EVIDENCE_UNIFORM, image=images, video=None, question=q, answer=a, source_path=source_rel)


def _make_task03(
    item_dir: str,
    plan: Dict[str, Any],
    input_root: str,
    *,
    require_video: bool,
) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if not hl or len(steps) < 1:
        return []

    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    for st in steps:
        step_id = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if step_id <= 0 or not step_goal:
            continue
        clip = _resolve_video_clip(item_dir, step_id)
        if require_video and not clip:
            continue

        if clip:
            evidence_type = EVIDENCE_CLIP
            q = f'Context: High-level goal: "{hl}" What is the step goal of this clip?'
            video_rel = _safe_relpath(clip, input_root)
            images: List[str] = []
        else:
            cfs = st.get("critical_frames") or []
            thumb = None
            if isinstance(cfs, list) and cfs:
                last_cf = cfs[-1] if isinstance(cfs[-1], dict) else None
                fi = _frame_index(last_cf) if last_cf else None
                if fi is not None:
                    thumb = _find_keyframe_image(item_dir, step_id, fi)
            if not thumb:
                continue
            evidence_type = EVIDENCE_KEYFRAME
            q = f'Context: High-level goal: "{hl}" What is the step goal of this keyframe image?'
            video_rel = None
            images = [_safe_relpath(thumb, input_root)]
        a = step_goal
        out.append(
            Sample(
                task_name=TASK_03,
                evidence_type=evidence_type,
                image=images,
                video=video_rel,
                question=q,
                answer=a,
                source_path=source_rel,
            )
        )
    return out


def _keyframe_for_task(item_dir: str, step: Dict[str, Any], *, prefer_j: int) -> Optional[Tuple[int, Dict[str, Any], str]]:
    step_id = int(step.get("step_id", 0) or 0)
    if step_id <= 0:
        return None
    cfs = step.get("critical_frames") or []
    if not isinstance(cfs, list) or len(cfs) < 1:
        return None
    j = int(prefer_j)
    if j < 0 or j >= len(cfs) or not isinstance(cfs[j], dict):
        j = 0
    cf = cfs[j]
    fi = _frame_index(cf)
    if fi is None:
        return None
    img = _find_keyframe_image(item_dir, step_id, fi)
    if not img:
        return None
    return step_id, cf, img


def _make_task04_to_20(item_dir: str, plan: Dict[str, Any], input_root: str, rng: random.Random) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    hl = _require_str(plan, "high_level_goal")
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    if not steps:
        return []

    out: List[Sample] = []

    # Helper: pick another step with different patient for negative feasibility.
    step_by_id = {int(s.get("step_id", 0) or 0): s for s in steps}
    step_ids = [sid for sid in step_by_id.keys() if sid > 0]
    step_ids.sort()

    def _other_step_with_diff_patient(cur_step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cur_pat = ((cur_step.get("causal_chain") or {}) if isinstance(cur_step.get("causal_chain"), dict) else {}).get("patient")
        for sid in rng.sample(step_ids, k=len(step_ids)):
            cand = step_by_id.get(sid)
            if not cand or cand is cur_step:
                continue
            cand_pat = ((cand.get("causal_chain") or {}) if isinstance(cand.get("causal_chain"), dict) else {}).get("patient")
            if isinstance(cur_pat, str) and isinstance(cand_pat, str) and cur_pat.strip() and cand_pat.strip() and cur_pat.strip() != cand_pat.strip():
                return cand
        return None

    # Pre-collect statements for verification negatives.
    spatial_pre_by_step: Dict[int, List[str]] = {}
    aff_pre_by_step: Dict[int, List[str]] = {}
    spatial_post_by_step: Dict[int, List[str]] = {}
    aff_post_by_step: Dict[int, List[str]] = {}
    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        if sid <= 0:
            continue
        cf0 = None
        cf1 = None
        cfs = st.get("critical_frames") or []
        if isinstance(cfs, list):
            if len(cfs) >= 1 and isinstance(cfs[0], dict):
                cf0 = cfs[0]
            if len(cfs) >= 2 and isinstance(cfs[1], dict):
                cf1 = cfs[1]
        if cf0:
            fcc0 = cf0.get("causal_chain") or {}
            if isinstance(fcc0, dict):
                sp0 = fcc0.get("causal_precondition_on_spatial")
                af0 = fcc0.get("causal_precondition_on_affordance")
                if isinstance(sp0, str):
                    spatial_pre_by_step[sid] = _split_numbered_block(sp0)
                else:
                    spatial_pre_by_step[sid] = [p for p in (_format_spatial_relation(r) for r in _parse_spatial_relations(sp0)) if p]
                if isinstance(af0, str):
                    aff_pre_by_step[sid] = _split_numbered_block(af0)
                else:
                    aff_pre_by_step[sid] = [p for p in (_format_affordance_state_compact(r) for r in _parse_affordance_states(af0)) if p]
        if cf1:
            fcc1 = cf1.get("causal_chain") or {}
            if isinstance(fcc1, dict):
                sp1 = fcc1.get("causal_effect_on_spatial")
                af1 = fcc1.get("causal_effect_on_affordance")
                if isinstance(sp1, str):
                    spatial_post_by_step[sid] = _split_numbered_block(sp1)
                else:
                    spatial_post_by_step[sid] = [p for p in (_format_spatial_relation(r) for r in _parse_spatial_relations(sp1)) if p]
                if isinstance(af1, str):
                    aff_post_by_step[sid] = _split_numbered_block(af1)
                else:
                    aff_post_by_step[sid] = [p for p in (_format_affordance_state_compact(r) for r in _parse_affordance_states(af1)) if p]

    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue

        step_cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        patient = _require_str(step_cc, "patient") if isinstance(step_cc, dict) else ""
        action = _require_str(step_cc, "action") if isinstance(step_cc, dict) else ""

        # --- Task 04 (patient) ---
        k0 = _keyframe_for_task(item_dir, st, prefer_j=0)
        if k0 and patient:
            _, _, img = k0
            q = f'Context: Step goal: "{step_goal}" In this image, what is the primary patient object being acted on?'
            out.append(Sample(TASK_04, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, patient, source_rel))

        # --- Task 05 (action phrase) ---
        if k0 and action:
            _, _, img = k0
            q = f'Context: Step goal: "{step_goal}" What is the action phrase (causal_chain.action) in this keyframe?'
            out.append(Sample(TASK_05, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, action, source_rel))

        # --- Task 06/07/08/09 (hotspot + state evolution) ---
        frames_for_hotspot: List[Tuple[Dict[str, Any], str]] = []
        if k0:
            _, cf0, img0 = k0
            frames_for_hotspot.append((cf0, img0))
        k1_hot = _keyframe_for_task(item_dir, st, prefer_j=1)
        if k1_hot:
            _, cf1, img1 = k1_hot
            if not frames_for_hotspot or frames_for_hotspot[-1][1] != img1:
                frames_for_hotspot.append((cf1, img1))

        for cf, img in frames_for_hotspot:
            intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
            hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
            aff_type = _require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else ""
            mech = _require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else ""
            desc = _require_str(hotspot, "description") if isinstance(hotspot, dict) else ""
            asc = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))

            if aff_type:
                q = f'Context: Step goal: "{step_goal}" What is the affordance_type of the interaction hotspot in this image?'
                out.append(Sample(TASK_06, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, aff_type, source_rel))
            if mech:
                q = f'Context: Step goal: "{step_goal}" Briefly describe the physical mechanism of the interaction hotspot in this image.'
                out.append(Sample(TASK_07, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, mech, source_rel))
            if desc and aff_type and mech:
                q = f'Context: Step goal: "{step_goal}" Locate the interaction hotspot area in the image first, then describe its affordance_type and mechanism.'
                desc_s = desc.strip()
                if desc_s and not desc_s.endswith((".", "!", "?")):
                    desc_s = desc_s + "."
                mech_s = mech.strip()
                if mech_s and not mech_s.endswith((".", "!", "?")):
                    mech_s = mech_s + "."
                a = f"The hotspot is {desc_s} It affords {aff_type.strip()}, and {mech_s}"
                a = _sanitize_space(a)
                out.append(
                    Sample(
                        TASK_08,
                        EVIDENCE_KEYFRAME,
                        [_safe_relpath(img, input_root)],
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "step_goal": step_goal,
                            "hotspot_description": desc.strip(),
                            "affordance_type": aff_type.strip(),
                            "mechanism": mech.strip(),
                        },
                    )
                )
            if asc:
                q = f'Context: Step goal: "{step_goal}" What ongoing action is occurring, and what immediate state change does it cause?'
                out.append(Sample(TASK_09, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, asc, source_rel))

        # --- Task 10 (holistic causal chain) ---
        cfs_all = st.get("critical_frames") or []
        if isinstance(cfs_all, list):
            for j in (0, 1):
                if j >= len(cfs_all) or not isinstance(cfs_all[j], dict):
                    continue
                cf = cfs_all[j]
                fi = _frame_index(cf)
                if fi is None:
                    continue
                img = _find_keyframe_image(item_dir, sid, fi)
                if not img:
                    continue

                fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
                agent = _require_str(step_cc, "agent") if isinstance(step_cc, dict) else ""

                sp_pre_full = fcc.get("causal_precondition_on_spatial") if isinstance(fcc, dict) else ""
                af_pre_full = fcc.get("causal_precondition_on_affordance") if isinstance(fcc, dict) else ""
                sp_eff_full = fcc.get("causal_effect_on_spatial") if isinstance(fcc, dict) else ""
                af_eff_full = fcc.get("causal_effect_on_affordance") if isinstance(fcc, dict) else ""
                sp_pre = _format_spatial(sp_pre_full, max_items=2)
                af_pre = _format_affordance(af_pre_full, max_items=2)
                sp_eff = _format_spatial(sp_eff_full, max_items=2)
                af_eff = _format_affordance(af_eff_full, max_items=2)

                hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
                desc = _require_str(hotspot, "description") if isinstance(hotspot, dict) else ""
                aff_type = _require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else ""
                mech = _require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else ""

                if not (agent and action and patient and (sp_pre or af_pre) and (sp_eff or af_eff) and desc and aff_type and mech):
                    continue

                q = (
                    f'Context: High-level goal: "{hl}" Step goal: "{step_goal}" '
                    "Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects."
                )

                p1 = _sanitize_space(
                    f"{sp_pre} {af_pre} In this setup, {agent} {action} {patient} to carry out the step goal."
                )
                mech_s = mech.strip()
                if mech_s and not mech_s.endswith((".", "!", "?")):
                    mech_s = mech_s + "."
                desc_s = re.sub(r"[.!?]+$", "", desc.strip()).strip()
                p2 = _sanitize_space(
                    f"The interaction hotspot is {desc_s}. It provides the affordance of {aff_type}. {mech_s} "
                    f"As a result, {sp_eff} {af_eff}"
                )
                a = p1 + "\n\n" + p2

                out.append(
                    Sample(
                        TASK_10,
                        EVIDENCE_KEYFRAME,
                        [_safe_relpath(img, input_root)],
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "high_level_goal": hl,
                            "step_goal": step_goal,
                            "agent": agent,
                            "action": action,
                            "patient": patient,
                            "spatial_preconditions": sp_pre_full,
                            "affordance_preconditions": af_pre_full,
                            "hotspot_description": desc,
                            "affordance_type": aff_type,
                            "mechanism": mech,
                            "spatial_effects": sp_eff_full,
                            "affordance_effects": af_eff_full,
                        },
                    )
                )

        # --- Task 11 (rationale) using step-level rationale; evidence j=0 ---
        rationale = _require_str(st, "rationale")
        if k0 and rationale:
            _, _, img = k0
            q = f'High-level goal: "{hl}" Step goal: "{step_goal}" Why is this step necessary for the overall goal?'
            out.append(
                Sample(
                    TASK_11,
                    EVIDENCE_KEYFRAME,
                    [_safe_relpath(img, input_root)],
                    None,
                    q,
                    rationale,
                    source_rel,
                    llm_fields={"high_level_goal": hl, "step_goal": step_goal, "rationale": rationale},
                )
            )

        # --- Preconditions (j=0) ---
        cf0 = st.get("critical_frames")[0] if isinstance(st.get("critical_frames"), list) and st.get("critical_frames") and isinstance(st.get("critical_frames")[0], dict) else None
        if k0 and cf0:
            _, _, img = k0
            fcc0 = cf0.get("causal_chain") if isinstance(cf0.get("causal_chain"), dict) else {}
            sp = _format_spatial(fcc0.get("causal_precondition_on_spatial"), max_items=2) if isinstance(fcc0, dict) else ""
            af = _format_affordance(fcc0.get("causal_precondition_on_affordance"), max_items=2) if isinstance(fcc0, dict) else ""
            if sp:
                q = f'Step goal: "{step_goal}" Describe the spatial preconditions that must hold before executing this step.'
                out.append(Sample(TASK_12, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, sp, source_rel))
            if af:
                q = f'Step goal: "{step_goal}" Describe the affordance preconditions that must hold before executing this step.'
                out.append(Sample(TASK_14, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, af, source_rel))
            # Verification (spatial): build a clean Yes/No pair by flipping truth when list-schema is available.
            sp_raw = fcc0.get("causal_precondition_on_spatial") if isinstance(fcc0, dict) else ""
            if isinstance(sp_raw, list):
                rel = next((r for r in _parse_spatial_relations(sp_raw) if _format_spatial_relation(r)), None)
                if rel:
                    cand_yes = _format_spatial_relation(rel)
                    q = (
                        f'Step goal: "{step_goal}" Candidate spatial precondition: "{cand_yes}" '
                        "Is this spatial precondition correct for executing the step in the current scene? Answer Yes/No/not directly observable."
                    )
                    out.append(Sample(TASK_13, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "Yes", source_rel))

                    rel_no = dict(rel)
                    rel_no["truth"] = not _coerce_truth(rel.get("truth", True))
                    cand_no = _format_spatial_relation(rel_no)
                    if cand_no and cand_no != cand_yes:
                        qn = (
                            f'Step goal: "{step_goal}" Candidate spatial precondition: "{cand_no}" '
                            "Is this spatial precondition correct for executing the step in the current scene? Answer Yes/No/not directly observable."
                        )
                        out.append(Sample(TASK_13, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, qn, "No", source_rel))
            else:
                sp_list = spatial_pre_by_step.get(sid) or []
                if sp_list:
                    cand = sp_list[0]
                    q = (
                        f'Step goal: "{step_goal}" Candidate spatial precondition: "{cand}" '
                        "Is this spatial precondition correct for executing the step in the current scene? Answer Yes/No/not directly observable."
                    )
                    out.append(Sample(TASK_13, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "Yes", source_rel))
            af_list = aff_pre_by_step.get(sid) or []
            if af_list:
                cand = af_list[0]
                q = (
                    f'Step goal: "{step_goal}" Candidate affordance precondition: "{cand}" '
                    "Is this affordance precondition correct for executing the step? Answer Yes/No/not directly observable."
                )
                out.append(Sample(TASK_15, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "Yes", source_rel))
            # Verification (No) using a different step's statement (affordance only; spatial handled above for list-schema).
            other_sid = next((x for x in step_ids if x != sid and aff_pre_by_step.get(x)), None)
            if other_sid is not None and aff_pre_by_step.get(other_sid):
                cand = aff_pre_by_step[other_sid][0]
                q = (
                    f'Step goal: "{step_goal}" Candidate affordance precondition: "{cand}" '
                    "Is this affordance precondition correct for executing the step? Answer Yes/No/not directly observable."
                )
                out.append(Sample(TASK_15, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "No", source_rel))

            # Feasibility
            q = f'Step goal: "{step_goal}" Is this step physically feasible now based on the visible spatial and affordance preconditions?'
            label = "feasible" if (sp or af) else "not directly observable"
            out.append(Sample(TASK_16, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, label, source_rel))

        # --- Postconditions (j=1) ---
        cf1 = None
        cfs = st.get("critical_frames") or []
        if isinstance(cfs, list) and len(cfs) >= 2 and isinstance(cfs[1], dict):
            cf1 = cfs[1]
        k1 = _keyframe_for_task(item_dir, st, prefer_j=1)
        if k1 and cf1:
            _, _, img = k1
            fcc1 = cf1.get("causal_chain") if isinstance(cf1.get("causal_chain"), dict) else {}
            sp = _format_spatial(fcc1.get("causal_effect_on_spatial"), max_items=2) if isinstance(fcc1, dict) else ""
            af = _format_affordance(fcc1.get("causal_effect_on_affordance"), max_items=2) if isinstance(fcc1, dict) else ""
            if sp:
                q = f'Step goal: "{step_goal}" Describe the spatial postconditions that should hold after completing this step.'
                out.append(Sample(TASK_17, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, sp, source_rel))
            if af:
                q = f'Step goal: "{step_goal}" Describe the affordance postconditions that should hold after completing this step.'
                out.append(Sample(TASK_19, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, af, source_rel))

            # Verification (spatial): build a clean Yes/No pair by flipping truth when list-schema is available.
            sp_raw = fcc1.get("causal_effect_on_spatial") if isinstance(fcc1, dict) else ""
            if isinstance(sp_raw, list):
                rel = next((r for r in _parse_spatial_relations(sp_raw) if _format_spatial_relation(r)), None)
                if rel:
                    cand_yes = _format_spatial_relation(rel)
                    q = (
                        f'Step goal: "{step_goal}" Candidate spatial postcondition: "{cand_yes}" '
                        "Is this spatial postcondition correct after completing the step? Answer Yes/No/not directly observable."
                    )
                    out.append(Sample(TASK_18, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "Yes", source_rel))

                    rel_no = dict(rel)
                    rel_no["truth"] = not _coerce_truth(rel.get("truth", True))
                    cand_no = _format_spatial_relation(rel_no)
                    if cand_no and cand_no != cand_yes:
                        qn = (
                            f'Step goal: "{step_goal}" Candidate spatial postcondition: "{cand_no}" '
                            "Is this spatial postcondition correct after completing the step? Answer Yes/No/not directly observable."
                        )
                        out.append(Sample(TASK_18, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, qn, "No", source_rel))
            else:
                sp_list = spatial_post_by_step.get(sid) or []
                if sp_list:
                    cand = sp_list[0]
                    q = (
                        f'Step goal: "{step_goal}" Candidate spatial postcondition: "{cand}" '
                        "Is this spatial postcondition correct after completing the step? Answer Yes/No/not directly observable."
                    )
                    out.append(Sample(TASK_18, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "Yes", source_rel))
            af_list = aff_post_by_step.get(sid) or []
            if af_list:
                cand = af_list[0]
                q = (
                    f'Step goal: "{step_goal}" Candidate affordance postcondition: "{cand}" '
                    "Is this affordance postcondition correct after completing the step? Answer Yes/No/not directly observable."
                )
                out.append(Sample(TASK_20, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "Yes", source_rel))

            # Verification (No) using a different step's statement (affordance only; spatial handled above for list-schema).
            other_sid = next((x for x in step_ids if x != sid and aff_post_by_step.get(x)), None)
            if other_sid is not None and aff_post_by_step.get(other_sid):
                cand = aff_post_by_step[other_sid][0]
                q = (
                    f'Step goal: "{step_goal}" Candidate affordance postcondition: "{cand}" '
                    "Is this affordance postcondition correct after completing the step? Answer Yes/No/not directly observable."
                )
                out.append(Sample(TASK_20, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, "No", source_rel))

    return out


_LATENT_KEYWORDS = (
    "not directly observable",
    "internal",
    "inside",
    "friction",
    "texture",
    "static",
    "circuit",
    "electrical",
    "magnetic",
    "chemical",
    "temperature",
    "pressure",
    "material",
    "viscos",
    "sufficient",
    "enough",
)


def _inline_clause(text: str) -> str:
    s = _sanitize_text_single_line(text)
    s = re.sub(r"[.?!]+\s+", "; ", s)
    s = s.strip().strip(";")
    s = re.sub(r"[;:,]+$", "", s).strip()
    s = re.sub(r"[.?!]+$", "", s).strip()
    return s


def _needs_not_directly_observable(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return False
    if "not directly observable" in s:
        return True
    return any(k in s for k in _LATENT_KEYWORDS[1:])


def _annotate_observability(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    if "not directly observable" in s.lower():
        return s
    if _needs_not_directly_observable(s):
        return f"{s} (not directly observable)"
    return s


def _lowercase_first_alpha(text: str) -> str:
    s = str(text or "")
    if not s:
        return s
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def _with_definite_article(token: str) -> str:
    s = str(token or "").strip()
    if not s:
        return s
    if re.match(r"^(the|a|an)\\b", s, flags=re.IGNORECASE):
        return s
    return f"the {s}"


def _enforce_single_sentence(text: str) -> str:
    s = _sanitize_text_single_line(text)
    if not s:
        return s
    punct = [m.start() for m in re.finditer(r"[.?!]", s)]
    if len(punct) <= 1:
        if s[-1] not in ".?!":
            return s + "."
        return s
    chars = list(s)
    for i in punct[:-1]:
        chars[i] = ";"
    s2 = _sanitize_text_single_line("".join(chars))
    if s2[-1] not in ".?!":
        s2 = s2 + "."
    return s2


def _make_task04_to_16(item_dir: str, plan: Dict[str, Any], input_root: str) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    hl = _require_str(plan, "high_level_goal")
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    if not steps:
        return []

    out: List[Sample] = []
    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue

        step_cc = st.get("causal_chain") if isinstance(st.get("causal_chain"), dict) else {}
        agent = _require_str(step_cc, "agent") if isinstance(step_cc, dict) else ""
        patient = _require_str(step_cc, "patient") if isinstance(step_cc, dict) else ""
        action = _require_str(step_cc, "action") if isinstance(step_cc, dict) else ""

        k0 = _keyframe_for_task(item_dir, st, prefer_j=0)
        k1 = _keyframe_for_task(item_dir, st, prefer_j=1)

        # Task 04/05: early keyframe
        if k0:
            _, _, img0 = k0
            if patient:
                q = f'Context: Step goal: "{step_goal}" In this image, what is the primary patient object being acted on?'
                out.append(Sample(TASK_04, EVIDENCE_KEYFRAME, [_safe_relpath(img0, input_root)], None, q, patient, source_rel))
            if action:
                q = f'Context: Step goal: "{step_goal}" What is the action phrase (causal_chain.action) in this keyframe?'
                out.append(Sample(TASK_05, EVIDENCE_KEYFRAME, [_safe_relpath(img0, input_root)], None, q, action, source_rel))

        # Task 06/07/08/09: use both keyframes when available
        frames_for_hotspot: List[Tuple[Dict[str, Any], str]] = []
        if k0:
            _, cf0, img0 = k0
            frames_for_hotspot.append((cf0, img0))
        if k1:
            _, cf1, img1 = k1
            if not frames_for_hotspot or frames_for_hotspot[-1][1] != img1:
                frames_for_hotspot.append((cf1, img1))

        for cf, img in frames_for_hotspot:
            intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}
            hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
            aff_type = _require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else ""
            mech = _require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else ""
            desc = _require_str(hotspot, "description") if isinstance(hotspot, dict) else ""
            asc = _strip_key_moment_prefix(_require_str(cf, "action_state_change_description"))

            if aff_type:
                q = f'Context: Step goal: "{step_goal}" What is the affordance_type of the interaction hotspot in this image?'
                out.append(Sample(TASK_06, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, aff_type, source_rel))
            if mech:
                q = f'Context: Step goal: "{step_goal}" Briefly describe the physical mechanism of the interaction hotspot in this image.'
                out.append(Sample(TASK_07, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, mech, source_rel))
            if desc and aff_type and mech:
                q = f'Context: Step goal: "{step_goal}" Locate the interaction hotspot area in the image first, then describe its affordance_type and mechanism.'
                mech_clause = _lowercase_first_alpha(_inline_clause(mech))
                desc_clause = _lowercase_first_alpha(desc.strip().rstrip("."))
                a = _sanitize_space(
                    f"The hotspot is {desc_clause}."
                    f" It affords {aff_type.strip()}, and the mechanism is that {mech_clause}."
                )
                out.append(
                    Sample(
                        TASK_08,
                        EVIDENCE_KEYFRAME,
                        [_safe_relpath(img, input_root)],
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "step_goal": step_goal,
                            "hotspot_description": desc.strip(),
                            "affordance_type": aff_type.strip(),
                            "mechanism": mech.strip(),
                        },
                    )
                )
            if asc:
                q = f'Context: Step goal: "{step_goal}" What ongoing action is occurring, and what immediate state change does it cause?'
                out.append(Sample(TASK_09, EVIDENCE_KEYFRAME, [_safe_relpath(img, input_root)], None, q, asc, source_rel))

        # Task 10: one-sentence holistic causal chain (both keyframes when possible)
        cfs_all = st.get("critical_frames") or []
        if isinstance(cfs_all, list):
            for j in (0, 1):
                if j >= len(cfs_all) or not isinstance(cfs_all[j], dict):
                    continue
                cf = cfs_all[j]
                fi = _frame_index(cf)
                if fi is None:
                    continue
                img = _find_keyframe_image(item_dir, sid, fi)
                if not img:
                    continue

                fcc = cf.get("causal_chain") if isinstance(cf.get("causal_chain"), dict) else {}
                intr = cf.get("interaction") if isinstance(cf.get("interaction"), dict) else {}

                sp_pre_full = fcc.get("causal_precondition_on_spatial") if isinstance(fcc, dict) else ""
                af_pre_full = fcc.get("causal_precondition_on_affordance") if isinstance(fcc, dict) else ""
                sp_eff_full = fcc.get("causal_effect_on_spatial") if isinstance(fcc, dict) else ""
                af_eff_full = fcc.get("causal_effect_on_affordance") if isinstance(fcc, dict) else ""

                sp_pre = _inline_clause(_format_spatial(sp_pre_full, max_items=1))
                af_pre = _inline_clause(_format_affordance(af_pre_full, max_items=1))
                sp_eff = _inline_clause(_format_spatial(sp_eff_full, max_items=1))
                af_eff = _inline_clause(_format_affordance(af_eff_full, max_items=1))

                hotspot = intr.get("hotspot") if isinstance(intr, dict) and isinstance(intr.get("hotspot"), dict) else intr
                desc = _inline_clause(_require_str(hotspot, "description") if isinstance(hotspot, dict) else "")
                aff_type = _inline_clause(_require_str(hotspot, "affordance_type") if isinstance(hotspot, dict) else "")
                mech = _inline_clause(_require_str(hotspot, "mechanism") if isinstance(hotspot, dict) else "")

                if not (hl and agent and action and patient and mech and (sp_pre or af_pre) and (sp_eff or af_eff)):
                    continue

                q = (
                    f'Context: High-level goal: "{hl}" Step goal: "{step_goal}" '
                    "Explain the physical causal chain in this keyframe, focusing on spatial setup, affordance mechanism, and immediate effects. "
                    "Answer in one English sentence."
                )

                pre_parts = [_lowercase_first_alpha(_annotate_observability(p)) for p in (sp_pre, af_pre) if p]
                eff_parts = [_lowercase_first_alpha(_annotate_observability(p)) for p in (sp_eff, af_eff) if p]
                pre_clause = " and ".join([p for p in pre_parts if p])
                eff_clause = " and ".join([p for p in eff_parts if p])

                hotspot_clause = ""
                if desc:
                    hotspot_clause = f" at the hotspot ({desc})"
                elif aff_type:
                    hotspot_clause = f" at the hotspot (affordance: {aff_type})"

                agent_phrase = _with_definite_article(agent)
                patient_phrase = _with_definite_article(patient)
                mech2 = _lowercase_first_alpha(mech)
                a = f"When {pre_clause}, {agent_phrase} {action} {patient_phrase}{hotspot_clause}; because {mech2}, {eff_clause}."
                a = _enforce_single_sentence(a)

                out.append(
                    Sample(
                        TASK_10,
                        EVIDENCE_KEYFRAME,
                        [_safe_relpath(img, input_root)],
                        None,
                        q,
                        a,
                        source_rel,
                        llm_fields={
                            "high_level_goal": hl,
                            "step_goal": step_goal,
                            "agent": agent,
                            "action": action,
                            "patient": patient,
                            "spatial_preconditions": sp_pre_full,
                            "affordance_preconditions": af_pre_full,
                            "hotspot_description": desc,
                            "affordance_type": aff_type,
                            "mechanism": mech,
                            "spatial_effects": sp_eff_full,
                            "affordance_effects": af_eff_full,
                        },
                    )
                )

        # Task 11: rationale (early keyframe)
        rationale = _require_str(st, "rationale")
        if k0 and hl and rationale:
            _, _, img0 = k0
            q = f'High-level goal: "{hl}" Step goal: "{step_goal}" Why is this step necessary for the overall goal?'
            out.append(
                Sample(
                    TASK_11,
                    EVIDENCE_KEYFRAME,
                    [_safe_relpath(img0, input_root)],
                    None,
                    q,
                    rationale,
                    source_rel,
                    llm_fields={"high_level_goal": hl, "step_goal": step_goal, "rationale": rationale},
                )
            )

        # Task 12/13/14: preconditions + feasibility (early keyframe)
        if k0:
            _, cf0, img0 = k0
            fcc0 = cf0.get("causal_chain") if isinstance(cf0.get("causal_chain"), dict) else {}
            sp_pre_2 = _format_spatial(fcc0.get("causal_precondition_on_spatial"), max_items=1) if isinstance(fcc0, dict) else ""
            af_pre_2 = _format_affordance(fcc0.get("causal_precondition_on_affordance"), max_items=1) if isinstance(fcc0, dict) else ""
            if sp_pre_2:
                q = f'Step goal: "{step_goal}" Describe the spatial preconditions that must hold before executing this step.'
                out.append(Sample(TASK_12, EVIDENCE_KEYFRAME, [_safe_relpath(img0, input_root)], None, q, sp_pre_2, source_rel))
            if af_pre_2:
                q = f'Step goal: "{step_goal}" Describe the affordance preconditions that must hold before executing this step.'
                out.append(Sample(TASK_13, EVIDENCE_KEYFRAME, [_safe_relpath(img0, input_root)], None, q, af_pre_2, source_rel))

            sp_pre_1 = _format_spatial(fcc0.get("causal_precondition_on_spatial"), max_items=1) if isinstance(fcc0, dict) else ""
            af_pre_1 = _format_affordance(fcc0.get("causal_precondition_on_affordance"), max_items=1) if isinstance(fcc0, dict) else ""
            if sp_pre_1 and af_pre_1:
                sp_clause = _lowercase_first_alpha(_inline_clause(sp_pre_1))
                af_clause = _lowercase_first_alpha(_inline_clause(af_pre_1))
                sp_status = "satisfied"
                if _needs_not_directly_observable(sp_clause):
                    sp_status = "not directly observable"
                af_status = "not directly observable" if _needs_not_directly_observable(af_clause) else "satisfied"
                prefix = "It is feasible now" if af_status == "satisfied" else "It is likely feasible now"
                a = (
                    f"{prefix} because {sp_clause} (spatial precondition {sp_status}) and "
                    f"{af_clause} (affordance precondition {af_status})."
                )
                a = _enforce_single_sentence(a)
                q = (
                    f'Step goal: "{step_goal}" Is this step physically feasible now? Answer in one English sentence, and justify the decision '
                    "by stating one spatial precondition and one affordance precondition, and whether each is satisfied/violated/not directly "
                    "observable in this frame."
                )
                out.append(Sample(TASK_14, EVIDENCE_KEYFRAME, [_safe_relpath(img0, input_root)], None, q, a, source_rel))

        # Task 15/16: postconditions (late keyframe)
        if k1:
            _, cf1, img1 = k1
            fcc1 = cf1.get("causal_chain") if isinstance(cf1.get("causal_chain"), dict) else {}
            sp_post = _format_spatial(fcc1.get("causal_effect_on_spatial"), max_items=1) if isinstance(fcc1, dict) else ""
            af_post = _format_affordance(fcc1.get("causal_effect_on_affordance"), max_items=1) if isinstance(fcc1, dict) else ""
            if sp_post:
                q = f'Step goal: "{step_goal}" Describe the spatial postconditions that should hold after completing this step.'
                out.append(
                    Sample(
                        TASK_15,
                        EVIDENCE_KEYFRAME,
                        [_safe_relpath(img1, input_root)],
                        None,
                        q,
                        _sanitize_space(_annotate_observability(sp_post)),
                        source_rel,
                    )
                )
            if af_post:
                q = f'Step goal: "{step_goal}" Describe the affordance postconditions that should hold after completing this step.'
                out.append(Sample(TASK_16, EVIDENCE_KEYFRAME, [_safe_relpath(img1, input_root)], None, q, _sanitize_space(_annotate_observability(af_post)), source_rel))

    return out


def _make_task17(item_dir: str, plan: Dict[str, Any], input_root: str) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    for i in range(len(steps) - 1):
        s0 = steps[i]
        s1 = steps[i + 1]
        sid0 = int(s0.get("step_id", 0) or 0)
        sg0 = _require_str(s0, "step_goal")
        sg1 = _require_str(s1, "step_goal")
        if sid0 <= 0 or not sg0 or not sg1:
            continue
        cc0 = s0.get("causal_chain") if isinstance(s0.get("causal_chain"), dict) else {}
        cc1 = s1.get("causal_chain") if isinstance(s1.get("causal_chain"), dict) else {}
        eff_sp = cc0.get("causal_effect_on_spatial") if isinstance(cc0, dict) else ""
        eff_af = cc0.get("causal_effect_on_affordance") if isinstance(cc0, dict) else ""
        pre_sp = cc1.get("causal_precondition_on_spatial") if isinstance(cc1, dict) else ""
        pre_af = cc1.get("causal_precondition_on_affordance") if isinstance(cc1, dict) else ""

        eff_terms = _terms_from_spatial(eff_sp) | _terms_from_affordance(eff_af)
        pre_terms = _terms_from_spatial(pre_sp) | _terms_from_affordance(pre_af)
        if not eff_terms or not pre_terms:
            continue
        if not (eff_terms & pre_terms):
            continue
        k1 = _keyframe_for_task(item_dir, s0, prefer_j=1)
        if not k1:
            continue
        _, _, img = k1
        q = (
            f'High-level goal: "{hl}" Previous step goal: "{sg0}" Next step goal: "{sg1}" '
            "How does the outcome of the previous step satisfy the preconditions for the next step?"
        )
        eff_s = _sanitize_space(
            " ".join(
                [
                    _format_spatial(eff_sp, max_items=1),
                    _format_affordance(eff_af, max_items=1),
                ]
            )
        )
        pre_s = _sanitize_space(
            " ".join(
                [
                    _format_spatial(pre_sp, max_items=1),
                    _format_affordance(pre_af, max_items=1),
                ]
            )
        )
        a = (
            f'Completing "{sg0}" establishes {eff_s}, which supports executing "{sg1}" '
            f"by satisfying its required setup, specifically {pre_s}."
        )
        a = _sanitize_space(a)
        out.append(
            Sample(
                TASK_17,
                EVIDENCE_KEYFRAME,
                [_safe_relpath(img, input_root)],
                None,
                q,
                a,
                source_rel,
                llm_fields={
                    "high_level_goal": hl,
                    "prev_step_goal": sg0,
                    "prev_step_effects": {"spatial": eff_sp, "affordance": eff_af},
                    "next_step_goal": sg1,
                    "next_step_preconditions": {"spatial": pre_sp, "affordance": pre_af},
                },
            )
        )
    return out


def _make_task18(item_dir: str, plan: Dict[str, Any], input_root: str) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    for i in range(len(steps) - 1):
        s0 = steps[i]
        s1 = steps[i + 1]
        sid0 = int(s0.get("step_id", 0) or 0)
        sg0 = _require_str(s0, "step_goal")
        sg1 = _require_str(s1, "step_goal")
        if sid0 <= 0 or not sg0 or not sg1:
            continue
        video = _resolve_video_prefix(item_dir, sid0)
        if not video:
            continue
        q = f'Context: High-level goal: "{hl}" Last completed step (in this prefix): "{sg0}" What is the next step goal?'
        evidence_type = EVIDENCE_PREFIX
        images = []
        video_rel = _safe_relpath(video, input_root)
        out.append(
            Sample(
                TASK_18,
                evidence_type,
                images,
                video_rel,
                q,
                sg1,
                source_rel,
            )
        )
    return out


def _make_task19(item_dir: str, plan: Dict[str, Any], input_root: str, head: int, tail: int) -> Optional[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 3 or not hl:
        return None
    sampled = _list_sampled_frames(item_dir)
    imgs = _pick_head_tail(sampled, head=head, tail=tail)
    if len(imgs) < 2:
        return None
    middle = [str(s.get("step_goal", "")).strip() for s in steps[1:-1] if str(s.get("step_goal", "")).strip()]
    if not middle:
        return None
    a = " ".join([f"{i+1}) {sg}" for i, sg in enumerate(middle)])
    q = f'High-level goal: "{hl}" Based on the beginning/end glimpses of the video, infer the missing middle steps in order.'
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    return Sample(
        task_name=TASK_19,
        evidence_type=EVIDENCE_UNIFORM,
        image=[_safe_relpath(p, input_root) for p in imgs],
        video=None,
        question=q,
        answer=_sanitize_space(a),
        source_path=source_rel,
    )


def _make_task20_21(item_dir: str, plan: Dict[str, Any], input_root: str, rng: random.Random) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 2 or not hl:
        return []
    out: List[Sample] = []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)

    # Choose prefix_end_step i and K (3..6) as a single sample per item.
    max_i = min(3, len(steps) - 1)
    i_idx = 0 if max_i <= 1 else rng.randrange(0, max_i)
    prefix_step = steps[i_idx]
    prefix_end_step = int(prefix_step.get("step_id", 0) or 0)
    if prefix_end_step <= 0:
        return []
    remaining = steps[i_idx + 1 :]
    if len(remaining) < 1:
        return []
    k = min(max(3, min(6, len(remaining))), len(remaining))
    gold = [str(s.get("step_goal", "")).strip() for s in remaining[:k] if str(s.get("step_goal", "")).strip()]
    if len(gold) < 1:
        return []
    video = _resolve_video_prefix(item_dir, prefix_end_step)
    if not video:
        return []
    last_completed_goal = str(prefix_step.get("step_goal", "")).strip()
    answer = " ".join([f"{j+1}) {sg}" for j, sg in enumerate(gold)])

    images: List[str] = []
    video_rel = _safe_relpath(video, input_root)
    q20 = (
        f'Context: High-level goal: "{hl}" Last completed step (in this prefix): "{last_completed_goal}" '
        f"Based on this prefix, predict the next K={len(gold)} step goals in order."
    )
    out.append(
        Sample(
            TASK_20,
            EVIDENCE_PREFIX,
            images,
            video_rel,
            q20,
            _sanitize_space(answer),
            source_rel,
        )
    )

    shuffled = list(gold)
    rng.shuffle(shuffled)
    images2: List[str] = []
    q21 = (
        f'Context: High-level goal: "{hl}" Last completed step (in this prefix): "{last_completed_goal}" '
        f"Reorder the shuffled candidate steps {json.dumps(shuffled)} into the most plausible next-step sequence."
    )
    out.append(
        Sample(
            TASK_21,
            EVIDENCE_PREFIX,
            images2,
            video_rel,
            q21,
            _sanitize_space(answer),
            source_rel,
        )
    )
    return out


def _make_task22_23(item_dir: str, plan: Dict[str, Any], input_root: str, rng: random.Random) -> Iterable[Sample]:
    hl = _require_str(plan, "high_level_goal")
    steps = _sorted_steps(plan)
    if len(steps) < 4 or not hl:
        return []
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)

    # Choose a prefix_end_step i near the beginning so that at least K=3 future steps exist.
    max_prefix_idx = min(2, len(steps) - 4)
    if max_prefix_idx < 0:
        return []
    i_idx = rng.randrange(0, max_prefix_idx + 1)
    prefix_step = steps[i_idx]
    prefix_end_step = int(prefix_step.get("step_id", 0) or 0)
    if prefix_end_step <= 0:
        return []

    remaining = steps[i_idx + 1 :]
    if len(remaining) < 3:
        return []
    k = min(3, len(remaining))
    gold = [str(s.get("step_goal", "")).strip() for s in remaining[:k] if str(s.get("step_goal", "")).strip()]
    if len(gold) != k:
        return []

    # Single-error bad plan: replace one step with a later gold step (precondition_missing) when possible.
    bad = list(gold)
    flaw_pos = rng.randrange(0, len(bad))
    later_pool = [str(s.get("step_goal", "")).strip() for s in remaining[k:] if str(s.get("step_goal", "")).strip()]
    if later_pool:
        bad[flaw_pos] = later_pool[0]
        flaw_type = "precondition_missing"
        flawed = str(bad[flaw_pos]).strip().strip('"').rstrip(".").strip()
        missing = str(gold[flaw_pos]).strip().strip('"').rstrip(".").strip()
        reason = (
            f'You cannot "{flawed}" before completing "{missing}" because the prerequisite spatial/affordance setup has not been established yet.'
        )
    else:
        bad[flaw_pos] = "Leave the workspace and stop."
        flaw_type = "goal_mismatch"
        reason = "This step does not contribute to achieving the high-level goal and prematurely terminates the plan."

    video = _resolve_video_prefix(item_dir, prefix_end_step)
    if not video:
        return []
    video_rel = _safe_relpath(video, input_root)
    img_rel: List[str] = []

    bad_steps_inline = " ".join([f'{i+1}) "{s}"' for i, s in enumerate(bad)])
    gold_steps_inline = " ".join([f'{i+1}) "{s}"' for i, s in enumerate(gold)])
    q29 = (
        f'Context: High-level goal: "{hl}" Based on this prefix, the following bad_plan_steps are proposed as the next steps: '
        f"{bad_steps_inline} Identify the flaw in the bad plan."
    )
    a29 = f"FlawStep={flaw_pos+1}; FlawType={flaw_type}; Reason={reason}"

    q30 = (
        f'Context: High-level goal: "{hl}" Based on this prefix, bad_plan_steps are proposed as the next steps: '
        f"{bad_steps_inline} Repair the plan by outputting the corrected {len(gold)}-step sequence."
    )
    a30 = " ".join([f'{i+1}) "{s}"' for i, s in enumerate(gold)])

    return [
        Sample(
            TASK_22,
            EVIDENCE_PREFIX,
            img_rel,
            video_rel,
            q29,
            a29,
            source_rel,
            llm_fields={
                "high_level_goal": hl,
                "bad_plan_steps": bad_steps_inline,
                "gold_plan_steps": gold_steps_inline,
                "flaw_type": flaw_type,
                "flaw_step": flaw_pos + 1,
            },
        ),
        Sample(TASK_23, EVIDENCE_PREFIX, img_rel, video_rel, q30, a30, source_rel),
    ]


def _counterfactual_clause(question: str) -> str:
    s = str(question or "").strip()
    if not s:
        return ""
    s = re.sub(r"^\s*what\s+if\s+", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(" ?!.").strip()
    if not s:
        return ""
    if s[:1].isupper():
        s = s[:1].lower() + s[1:]
    return s


def _make_task24_27(item_dir: str, plan: Dict[str, Any], input_root: str) -> Iterable[Sample]:
    steps = _sorted_steps(plan)
    hl = _require_str(plan, "high_level_goal")
    source_rel = _safe_relpath(os.path.join(item_dir, "causal_plan_with_keyframes.json"), input_root)
    out: List[Sample] = []
    for st in steps:
        sid = int(st.get("step_id", 0) or 0)
        step_goal = _require_str(st, "step_goal")
        if sid <= 0 or not step_goal:
            continue
        k0 = _keyframe_for_task(item_dir, st, prefer_j=0)
        img_rel = [_safe_relpath(k0[2], input_root)] if k0 else []

        q_cf = _require_str(st, "counterfactual_challenge_question")
        a_cf = _require_str(st, "expected_challenge_outcome")
        if q_cf and a_cf and img_rel:
            q24 = (
                f'Context: Step goal: "{step_goal}" Counterfactual: {q_cf} '
                "From a spatial & affordance perspective, what would likely happen? "
                "Only predict the outcome; do not propose any recovery actions."
            )
            out.append(Sample(TASK_24, EVIDENCE_KEYFRAME, img_rel, None, q24, a_cf, source_rel))
            # Shorter outcome for Task_25: first sentence when possible.
            a25 = a_cf.split(".")[0].strip()
            if a25:
                a25 = a25 + "."
            clause = _counterfactual_clause(q_cf) or q_cf.strip().rstrip("?")
            q25 = (
                f'Context: Step goal: "{step_goal}" What is the most likely outcome if {clause}? '
                "Answer with a short outcome prediction grounded in spatial setup and affordance, and do not propose any recovery actions."
            )
            out.append(Sample(TASK_25, EVIDENCE_KEYFRAME, img_rel, None, q25, a25 or a_cf, source_rel))

        fr = st.get("failure_reflecting") if isinstance(st.get("failure_reflecting"), dict) else {}
        reason = _require_str(fr, "reason") if isinstance(fr, dict) else ""
        strat = _require_str(fr, "recovery_strategy") if isinstance(fr, dict) else ""
        if reason and strat:
            if img_rel:
                q26 = (
                    f'Context: Step goal: "{step_goal}" Failure reason: "{reason}" '
                    "What is a plausible recovery strategy? Explain briefly using spatial stability and affordance/mechanism."
                )
                strategy = strat.strip()
                if strategy and not strategy.endswith((".", "!", "?")):
                    strategy = strategy + "."

                markers = ("because", "so that", "to ", "prevent", "avoid", "by ", "thereby", "which ")
                strategy_lower = strategy.lower()
                ignore = set(_STOPWORDS) | set(_GENERIC_OBJECT_TOKENS) | {"step", "goal"}

                def _keywords(text: str) -> set[str]:
                    toks = re.findall(r"[a-zA-Z]+", str(text or "").lower())
                    return {t for t in toks if len(t) >= 4 and t not in ignore}

                needs_expl = False
                if not any(m in strategy_lower for m in markers):
                    overlap = _keywords(reason) & _keywords(strategy)
                    needs_expl = not bool(overlap)
                if needs_expl:
                    reason_inline = reason.strip().rstrip(".!?").strip()
                    a26 = _sanitize_space(
                        f"{strategy} This directly addresses the stated failure ({reason_inline}) by restoring the necessary spatial stability/alignment and enabling the intended affordance/mechanism."
                    )
                else:
                    a26 = strategy
                out.append(
                    Sample(
                        TASK_26,
                        EVIDENCE_KEYFRAME,
                        img_rel,
                        None,
                        q26,
                        a26,
                        source_rel,
                        llm_fields={"step_goal": step_goal, "failure_reason": reason, "recovery_strategy": strat},
                    )
                )

            if sid <= 1:
                continue
            video = _resolve_video_prefix(item_dir, sid - 1)
            if not video:
                continue
            q27 = (
                f'Context: High-level goal: "{hl}" Failure reason: "{reason}" Recovery strategy: "{strat}" '
                "After applying the recovery strategy, what is the most appropriate next step? Answer as a single step_goal."
            )
            # Default label: retry current step_goal
            out.append(Sample(TASK_27, EVIDENCE_PREFIX, [], _safe_relpath(video, input_root), q27, step_goal, source_rel))

    return out


def generate_samples_for_item(
    *,
    item_dir: str,
    input_root: str,
    enabled_tasks: set[str],
    uniform_k: int,
    head: int,
    tail: int,
    require_videos: bool,
    enable_task22: bool,
    strict_schema: bool,
    rng: random.Random,
) -> List[Sample]:
    plan_path = os.path.join(item_dir, "causal_plan_with_keyframes.json")
    plan = _read_json(plan_path)
    if strict_schema:
        _validate_final_plan_schema(plan, source=_safe_relpath(plan_path, input_root), strict=True)

    out: List[Sample] = []

    if TASK_01 in enabled_tasks:
        s = _make_task01(item_dir, plan, input_root, uniform_k=uniform_k)
        if s:
            out.append(s)
    if TASK_02 in enabled_tasks:
        s = _make_task02(item_dir, plan, input_root, uniform_k=uniform_k, rng=rng)
        if s:
            out.append(s)
    if TASK_03 in enabled_tasks:
        out.extend(_make_task03(item_dir, plan, input_root, require_video=require_videos))
    if any(
        t in enabled_tasks
        for t in (
            TASK_04,
            TASK_05,
            TASK_06,
            TASK_07,
            TASK_08,
            TASK_09,
            TASK_10,
            TASK_11,
            TASK_12,
            TASK_13,
            TASK_14,
            TASK_15,
            TASK_16,
        )
    ):
        out.extend(_make_task04_to_16(item_dir, plan, input_root))
        out = [s for s in out if s.task_name in enabled_tasks]
    if TASK_17 in enabled_tasks:
        out.extend(_make_task17(item_dir, plan, input_root))
    if TASK_18 in enabled_tasks:
        out.extend(_make_task18(item_dir, plan, input_root))
    if TASK_19 in enabled_tasks:
        s = _make_task19(item_dir, plan, input_root, head=head, tail=tail)
        if s:
            out.append(s)
    if TASK_20 in enabled_tasks or TASK_21 in enabled_tasks:
        out.extend(_make_task20_21(item_dir, plan, input_root, rng=rng))
    if TASK_22 in enabled_tasks or TASK_23 in enabled_tasks:
        out.extend(_make_task22_23(item_dir, plan, input_root, rng=rng))
    if any(t in enabled_tasks for t in (TASK_24, TASK_25, TASK_26, TASK_27)):
        out.extend(_make_task24_27(item_dir, plan, input_root))

    # Final text leak check.
    final: List[Sample] = []
    for s in out:
        if s.task_name not in enabled_tasks:
            continue
        images = [p for p in s.image if p]
        video = s.video.strip() if isinstance(s.video, str) and s.video.strip() else None
        if not images and not video:
            continue
        if _has_frame_leak(s.question) or _has_frame_leak(s.answer):
            continue
        answer = _sanitize_space(s.answer)
        if s.task_name in SINGLE_SENTENCE_TASKS:
            answer = _enforce_single_sentence(answer)
        final.append(
            Sample(
                task_name=s.task_name,
                evidence_type=s.evidence_type,
                image=images,
                video=video,
                question=_sanitize_space(s.question),
                answer=answer,
                source_path=s.source_path,
                llm_fields=s.llm_fields,
            )
        )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Mani-LongVideo Task_01–Task_27 QA dataset from causal_plan_with_keyframes.json (final schema)."
    )
    parser.add_argument("--input-root", required=True, help="Dataset root containing many item dirs with causal_plan_with_keyframes.json.")
    parser.add_argument("--output-dir", required=True, help="Output root directory (will create one folder per task with data.jsonl).")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N item dirs (0 = no limit).")
    parser.add_argument("--tasks", nargs="*", default=list(ALL_TASKS), help="Subset of task names to generate (default: all Task_01–Task_27).")
    parser.add_argument("--uniform-k", type=int, default=8, help="Number of uniform frames for images_uniform_scene tasks (default: 8).")
    parser.add_argument("--head", type=int, default=4, help="Head frames for Task_19 (default: 4).")
    parser.add_argument("--tail", type=int, default=4, help="Tail frames for Task_19 (default: 4).")
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="If set, require video_clip assets for Task_03 (skip keyframe fallback); video_prefix tasks always require mp4.",
    )
    parser.add_argument(
        "--enable-task22",
        action="store_true",
        help="Deprecated (ignored). Task_22 is controlled via --tasks.",
    )
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--strict-schema",
        dest="strict_schema",
        action="store_true",
        help="Enable strict final-schema validation for causal_plan_with_keyframes.json (default).",
    )
    schema_group.add_argument(
        "--no-strict-schema",
        dest="strict_schema",
        action="store_false",
        help="Disable strict schema validation (legacy/unsafe).",
    )
    parser.set_defaults(strict_schema=True)
    parser.add_argument("--no-api", action="store_true", help="Disable OpenAI-compatible API two-stage rewriting; keep deterministic answers.")
    parser.add_argument("--llm-tasks", nargs="*", default=list(DEFAULT_LLM_TASKS), help="Tasks to rewrite/polish via API (default: a small subset).")
    parser.add_argument("--llm-max-tokens", type=int, default=0, help="Override MAX_TOKENS for API calls (0 uses env/default).")
    parser.add_argument("--llm-temperature", type=float, default=0.3, help="Override TEMPERATURE for API calls (default: 0.3).")
    parser.add_argument("--llm-single-pass", action="store_true", help="Use a single API pass (no second polishing pass).")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir)
    enabled_tasks = set(args.tasks or [])
    llm_tasks = set(args.llm_tasks or [])

    unknown = sorted([t for t in enabled_tasks if t not in set(ALL_TASKS)])
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    unknown_llm = sorted([t for t in llm_tasks if t not in set(ALL_TASKS)])
    if unknown_llm:
        raise ValueError(f"Unknown llm task names: {unknown_llm}")

    item_dirs = _list_item_dirs(input_root)
    if args.limit and int(args.limit) > 0:
        item_dirs = item_dirs[: int(args.limit)]
    if not item_dirs:
        raise FileNotFoundError(f"No item dirs found under {input_root} (expecting causal_plan_with_keyframes.json).")

    logger.info(f"Found {len(item_dirs)} item dirs under: {input_root}")
    logger.info(f"Enabled tasks: {sorted(enabled_tasks)}")
    if llm_tasks:
        logger.info(f"LLM rewrite tasks: {sorted(llm_tasks)}")

    llm: Optional[TwoStageLlm] = None
    if not bool(args.no_api):
        cfg = ApiConfig()
        if int(args.llm_max_tokens) > 0:
            cfg.max_tokens = int(args.llm_max_tokens)
        cfg.temperature = float(args.llm_temperature)
        llm = TwoStageLlm(cfg)
        if not llm.enabled():
            llm = None
            logger.info("LLM disabled (missing API_KEY or client init failure); proceeding without API rewriting.")

    total = 0
    for idx, item_dir in enumerate(item_dirs, start=1):
        rel_item = _safe_relpath(item_dir, input_root)
        logger.info(f"[{idx}/{len(item_dirs)}] Processing item: {rel_item}")
        item_rng = random.Random(_stable_int_seed(rel_item))
        samples = generate_samples_for_item(
            item_dir=item_dir,
            input_root=input_root,
            enabled_tasks=enabled_tasks,
            uniform_k=int(args.uniform_k),
            head=int(args.head),
            tail=int(args.tail),
            require_videos=bool(args.require_videos),
            enable_task22=bool(args.enable_task22),
            strict_schema=bool(args.strict_schema),
            rng=item_rng,
        )
        if llm is not None and llm_tasks:
            samples = _apply_llm(samples, llm, llm_tasks, two_pass=not bool(args.llm_single_pass))
        for s in samples:
            entry = _sharegpt_entry(s)
            out_path = os.path.join(output_dir, s.task_name, "data.jsonl")
            _write_jsonl(out_path, entry)
            total += 1
        if idx % 10 == 0:
            logger.info(f"Progress: items={idx}, samples_written={total}")
    logger.info(f"Done. Total samples_written={total}. Output_dir={output_dir}")


if __name__ == "__main__":
    main()
