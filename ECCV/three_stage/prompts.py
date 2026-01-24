from __future__ import annotations

from typing import Tuple


SYSTEM_PROMPT_ANALYST = """
You are a highly advanced AI acting as a Physical Interaction Analyst and Causal Planner. Your primary mission is to deconstruct observed actions in video frames into their fundamental causal, spatial, and affordance-based physical principles.
You must analyze key moments from a continuous action sequence to produce structured annotations grounded strictly in visual evidence.
Your output MUST be a single, syntactically flawless JSON object. JSON validity is a critical, non-negotiable requirement.
Return JSON only: no markdown, no comments, no extra text.
""".strip()


def build_stage1_user_prompt(num_frames: int, image_dimensions: Tuple[int, int]) -> str:
    w, h = image_dimensions
    return f"""
Analyze the provided {num_frames} frames (uniformly sampled from one continuous video, chronological order). Treat the frames as the ONLY source of truth.

Goal: Generate a draft, step-by-step causal plan and step-level annotations for the entire video.

CRITICAL CONSTRAINT (Stage 1):
- Do NOT generate ANY keyframe-level fields.
- In particular, do NOT output `critical_frames`, `frame_index`, `interaction`, or `keyframe_image_path` anywhere.
- Do NOT reference frame/image indices or timestamps in any field (e.g., "Frame 12", "Image 12", "t=3.2s").

Language & grounding:
- Use objective, professional English.
- Use conservative naming (e.g., "container", "tool", "vegetable") when unsure.
- Do not hallucinate hidden states or off-screen objects.
- Keep the plan at a granularity that is realistic to localize with {num_frames} sampled frames (prefer 4-7 steps; keep within 3-8).
- Use consistent object naming across all steps (do not rename the same object with different synonyms).
- Prefer `snake_case` for object identifiers in lists/fields (e.g., `objects`, `object_name`, `agent`, `patient`).
- Avoid placeholders like "unknown", "N/A", "..." — fill every field with grounded, specific content.

Output format (strict JSON only; no extra text):
{{
  "high_level_goal": "One comprehensive English sentence describing the overall goal and intended final outcome of the entire video (focus on the final world state; do NOT list steps).",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "A specific, action-oriented sub-goal for this step (unique across steps; chronological; no frame/time references).",
      "rationale": "Causal justification: why this step is necessary and what it enables for later steps (mechanistic, not narration).",
      "causal_chain": {{
        "agent": "Primary force/controller for the WHOLE step (prefer body part like 'hands'/'right_hand'; use a tool part only if it is clearly the direct force applicator).",
        "action": "Concise verb phrase summarizing the core physical action for the WHOLE step (include the physical mechanism when helpful; e.g., 'apply torque to loosen', 'tilt to pour').",
        "patient": "Primary entity being acted upon in this step (`snake_case`; reuse the same identifier across all steps and lists).",
        "causal_precondition_on_spatial": [
          {{
            "relation": "Short, mechanistic, visually verifiable spatial/physical relation token that MUST hold immediately before and throughout this step (prefer contact/support/grasp/containment/alignment/open/closed over vague text). `objects` must list the involved entities (`snake_case`), and `truth` indicates whether the relation holds. Examples: 'holding', 'contacting', 'on_top_of', 'inside', 'inserted_into', 'aligned_with', 'open', 'closed'.",
            "objects": ["object_a", "object_b"],
            "truth": true
          }}
        ],
        "causal_precondition_on_affordance": [
          {{
            "object_name": "Object whose functional affordance/state MUST already be true to execute this step (`snake_case`; grounded in visible evidence). `affordance_types` must be a non-empty list of short `snake_case` tokens, and `reasons` must justify them.",
            "affordance_types": ["affordance_a"],
            "reasons": "Grounded justification citing visible cues and why this affordance/state is required (no speculation)."
          }}
        ],
        "causal_effect_on_spatial": [
          {{
            "relation": "Short, concrete spatial/physical relation token that becomes true or false as a RESULT of this step (visually verifiable). Set `truth` to the post-step truth value (true = established, false = broken).",
            "objects": ["object_a", "object_b"],
            "truth": true
          }}
        ],
        "causal_effect_on_affordance": [
          {{
            "object_name": "Object whose functional affordance/state changes as a RESULT of this step (`snake_case`; grounded in visible evidence). `affordance_types` must be a non-empty list of short `snake_case` tokens, and `reasons` must justify them.",
            "affordance_types": ["affordance_a"],
            "reasons": "Grounded justification citing visible cues and the causal mechanism for the state change (no speculation)."
          }}
        ]
      }},
      "counterfactual_challenge_question": "A realistic what-if question that perturbs a physical precondition of this step (do NOT mention frame numbers).",
      "expected_challenge_outcome": "Predicted physical outcome (and why) for the challenge question.",
      "failure_reflecting": {{
        "reason": "A plausible failure mode for this step (physical/procedural; grounded).",
        "recovery_strategy": "A concrete, actionable recovery strategy to still accomplish the step."
      }}
    }}
  ]
}}

Additional constraints:
- Step ordering MUST follow the chronological order implied by the frames.
- `step_id` MUST start at 1 and increase by 1.
- Each `step_goal` must be non-empty, specific, and not duplicated across steps.
- Keep each `step_goal` concise and focused on a single sub-goal (prefer <= 12 words).
- `causal_chain.agent/action/patient` MUST be non-empty strings.
- `causal_chain.causal_precondition_on_spatial`, `causal_chain.causal_precondition_on_affordance`, `causal_chain.causal_effect_on_spatial`, `causal_chain.causal_effect_on_affordance` MUST be non-empty lists.
- `causal_chain.causal_precondition_on_affordance[*].reasons` and `causal_chain.causal_effect_on_affordance[*].reasons` MUST be non-empty grounded explanations.
- `counterfactual_challenge_question` and `expected_challenge_outcome` MUST be non-empty strings.
- `failure_reflecting.reason` and `failure_reflecting.recovery_strategy` MUST be non-empty strings.
- Ensure cross-step causal consistency: Step i `causal_effect_on_*` should make Step i+1 `causal_precondition_on_*` plausible (avoid contradictions).
- Do NOT add any extra keys beyond the schema above.
- Each step should be anchorable to visual evidence, but you are NOT selecting frames in this stage.

Silent self-check before you output:
- Strict valid JSON only (double quotes, no trailing commas, no markdown fences).
- No frame/image index references anywhere.
- All `truth` fields are JSON booleans (`true`/`false`), not strings.
- All list-typed fields (`steps`, `objects`, `affordance_types`, etc.) are JSON arrays (not single strings).
- Step count within 3-8 (preferred 4-7).
- No empty lists/strings for required fields; no placeholder values.
- No forbidden keys (`critical_frames`, `frame_index`, `interaction`, `keyframe_image_path`) anywhere.

The input frames are resized to approximately ({w}x{h}) pixels. Now output the final strict JSON object only.
""".strip()


def build_stage2_user_prompt(high_level_goal: str, draft_plan_outline: str, num_frames: int) -> str:
    return f"""
You are an expert video step temporal localization assistant.
You are given:
1) {num_frames} uniformly sampled frames from the FULL original video (chronological order).
2) A draft step list extracted from a plan (read-only; do NOT edit it).

High-level goal (context): {high_level_goal}

Draft steps (read-only):
{draft_plan_outline}

Note on indices:
- Some frames may look identical due to uniform sampling/padding; avoid choosing a segment whose boundaries fall on visually identical frames with no time progress.

Task (Stage 2):
For EACH step, predict the corresponding time interval in the original video by selecting:
- `start_frame_index`: the 1-based index of the boundary timestamp where this step starts (inclusive).
- `end_frame_index`: the 1-based index of the boundary timestamp where this step ends (exclusive; the first frame AFTER the step ends).

Interpretation:
- Let `t(i)` be the timestamp of sampled frame `i`.
- The step clip is cut as the half-open interval `[t(start_frame_index), t(end_frame_index))`.
- Because boundaries are on a shared grid, `end_i` may equal `start_(i+1)` (contiguous, no overlap).

Example:
- If Step 2 ends and Step 3 begins at the same boundary, you may set `end_2 == start_3`.

Procedure (recommended):
1) Read the draft step_goal texts to know what to look for.
2) Scan the sampled frames in order to find where each step begins/ends.
3) Choose boundaries that fully cover the step. When uncertain, expand outward by 1 frame rather than risking cutting out essential context for Stage 3.

IMPORTANT:
- Indices refer ONLY to the provided {num_frames} frames (1..{num_frames}), not the original video frame numbers.
- You MAY output `{num_frames + 1}` ONLY for `end_frame_index` to indicate the exclusive boundary AFTER the last provided frame (typically for the last step to cover the video end).
- Do NOT output seconds/timestamps; output indices only.
- Do NOT add/remove/reorder steps. Output must cover exactly the draft step_ids.
- Output must contain exactly one entry per draft `step_id` and MUST NOT include any extra step_ids.
- Do NOT change `step_goal` (it is fixed by the draft).
- Enforce monotonic, non-overlapping segments: for consecutive steps, `end_i <= start_(i+1)`.
- Enforce positive duration in the sampled timeline: `start_frame_index < end_frame_index` for every step.
- Prefer near-contiguous coverage across steps (often `end_i == start_(i+1)`), unless there is clear idle time or a real gap between actions.
- Prefer full coverage of the video: typically `start_1 == 1` and `end_last == {num_frames + 1}` unless the video clearly begins/ends with irrelevant idle content.
- Do NOT add any semantic annotations or extra fields beyond the required indices.
- Each entry in `steps` MUST contain exactly these keys: `step_id`, `start_frame_index`, `end_frame_index`.
- Output MUST be exactly one JSON object with a single top-level key `steps` (no other top-level keys).
- Output steps in ascending `step_id` order.
- The example below is illustrative only; your output MUST include exactly one entry per draft `step_id`.

Silent self-check before you output:
- All step_ids included exactly once; no extra ids.
- All indices are integers; `start_frame_index` within [1, {num_frames}], `end_frame_index` within [2, {num_frames + 1}].
- For every step: start < end.
- For every consecutive pair: end_i <= start_(i+1).

Output format (strict JSON only):

Field definitions (read carefully; output JSON must contain ONLY the keys in the template):
- `steps` (list): Exactly one entry per draft `step_id` (no extra/missing ids), in ascending `step_id` order.
- `steps[*].step_id` (int): Draft step identifier (must match exactly; do not renumber/reorder).
- `steps[*].start_frame_index` (int): Inclusive start boundary (1-based, within [1, {num_frames}]). Choose the boundary where the step begins; when uncertain, bias slightly earlier to preserve context for Stage 3.
- `steps[*].end_frame_index` (int): Exclusive end boundary (1-based, within [2, {num_frames + 1}]). Choose the first boundary AFTER the step ends; can equal the next step's `start_frame_index`; must satisfy `start_frame_index < end_frame_index`. Use `{num_frames + 1}` to indicate "after the last provided frame" (typically for the last step).

Output JSON template (replace the numbers with your chosen indices; keep keys exactly):
{{
  "steps": [
    {{
      "step_id": 1,
      "start_frame_index": 1,
      "end_frame_index": 2
    }},
    {{
      "step_id": 2,
      "start_frame_index": 2,
      "end_frame_index": 5
    }}
  ]
}}
""".strip()


def build_stage3_user_prompt(high_level_goal: str, draft_plan_outline: str, draft_step_json: str, num_frames: int) -> str:
    return f"""
You are an expert Physical Interaction Analyst and Causal Planner.
You are given {num_frames} uniformly sampled frames from a SINGLE STEP CLIP (chronological order), and the draft step definition (read-only).

Task (Stage 3):
Using the step-clip frames as the PRIMARY evidence, refine and complete the annotation for this step and generate 2 keyframe annotations.

Keyframe selection procedure (recommended; follow silently):
1) Scan all frames quickly to understand the step progression and physical state changes.
2) Pick exactly 2 DISTINCT frames that are the two most causally important and visually anchorable key moments within this step (NOT limited to initiation/completion).
3) Treat each keyframe as a conjunction of constraints: the selected `frame_index` MUST be consistent with its own
   `action_state_change_description`, `causal_chain` (frame-level), and `interaction` simultaneously (avoid partial matches).
4) Ensure the 2 selected frames are in chronological order (`frame_index` strictly increases). If multiple frames match similarly well, prefer the EARLIER index.

Strict requirements:
- You MUST NOT change `step_id` or `step_goal` from the draft.
- You MUST output exactly 2 `critical_frames`.
- Each `critical_frames[*].frame_index` MUST be an integer in [1, {num_frames}] and refers to the step-clip frame pool provided here.
- Do NOT reference frame/image numbers (e.g., "Frame 12", "Image 12") in any text fields; use only the numeric `frame_index` field to specify frames.
- Do NOT reference timestamps/durations/timecodes in any text fields (e.g., "3 seconds", "00:03", "t=3.2s").
- Choose 2 DISTINCT frames that show meaningful temporal progression within the step; do not pick duplicates.
- The indices within `critical_frames` must be in strictly increasing time order.
- Do NOT output `keyframe_image_path` (keyframe JPEGs are resolved from the filesystem by the script).
- Output strict JSON only; no explanations.
- Do NOT add any extra keys beyond the schema below.
- `counterfactual_challenge_question` MUST start with 'What if ...?'.
- In each `critical_frames[*]`, `causal_chain` MUST contain ONLY the 4 textual fields (`causal_precondition_on_spatial`, `causal_precondition_on_affordance`, `causal_effect_on_spatial`, `causal_effect_on_affordance`) and MUST NOT include `agent`/`action`/`patient`.
- In each `critical_frames[*]`, `interaction` MUST contain ONLY `description`, `affordance_type`, and `mechanism` (do NOT output tools/materials and do NOT nest a `hotspot` object).
- For every `causal_*` field (both step-level and keyframe-level), output a single JSON string listing numbered points using lines starting with `1. `, `2. `, ...; do NOT put raw newlines inside a JSON string and instead separate points using the escaped sequence `\\n` inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each numbered line must be a complete, objective statement tightly tied to the current step or key moment (avoid short, generic fragments).

Quality and grounding constraints:
- Treat the frames as the ONLY source of truth. Do not hallucinate objects, contacts, or states not supported by the images.
- Step-level `causal_chain.causal_precondition_on_*` and `causal_chain.causal_effect_on_*` MUST be MACRO summaries that integrate the entire step (not a single instant).
- In each `critical_frames[*]`, `causal_chain.causal_precondition_on_spatial` and `causal_chain.causal_precondition_on_affordance` MUST describe the state of the world TRUE/REQUIRED AT that key moment, and MUST be visually consistent with the chosen image.
- In each `critical_frames[*]`, `causal_chain.causal_effect_on_spatial` and `causal_chain.causal_effect_on_affordance` MUST describe the PREDICTED immediate, local post-action effects right after the micro-action implied by `action_state_change_description` completes (short-term/local; not necessarily currently visible).
- `interaction.description/affordance_type/mechanism` must refer to a specific functional region that is visibly involved (edge, handle, rim, hinge, etc.) and explain a plausible physical mechanism (force/torque transfer, friction, leverage, flow, etc.).
- Avoid placeholders like "N/A", "unknown", or empty strings; fill all required fields with grounded, specific content.
- Use consistent object naming across all fields (causal_chain, interaction); do not rename the same object with different synonyms within the step.
- Prefer concrete, mechanistic relations and state terms (e.g., contacting, holding, inside, aligned_with, open/closed) rather than vague language.
- Use `snake_case` for object identifiers where possible.
- Keep `agent` as a person/body part (e.g., "hand", "hands", "finger", "person"); keep `action` as a verb phrase; keep `patient` as the acted-on object.

Output schema (strict):
(Do NOT add any extra keys beyond this schema. Your output must be valid JSON with real values.)

Top-level fields (one step JSON object):
- `step_id` (int): Must equal the draft `step_id` exactly (read-only).
- `step_goal` (string): Must exactly equal the draft `step_goal` (read-only; do not rephrase or paraphrase).
- `rationale` (string): 1–2 grounded sentences explaining why this step is necessary and what macro preconditions/effects it connects (mechanistic, not narration; do NOT mention frames/images/timestamps).
- `causal_chain` (object): Step-level MACRO physical causal analysis for the ENTIRE step:
  - `agent` (string): Primary force/controller for the whole step (prefer body part like 'hands'/'left_hand'/'right_hand'; use a tool part only if it is clearly the direct force applicator). Use one stable identifier.
  - `action` (string): Concise physical verb phrase for the whole step (include mechanism when possible: push/pull/rotate/tilt/insert/press). Avoid vague verbs like 'do'/'move'.
  - `patient` (string): Primary acted-on object identifier in `snake_case`. Keep naming consistent across all fields.
  - `causal_precondition_on_spatial` (string): A single JSON string listing MACRO spatial preconditions for the ENTIRE step as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (avoid short, generic fragments).
  - `causal_precondition_on_affordance` (string): A single JSON string listing MACRO affordance/state preconditions for the ENTIRE step as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (avoid short, generic fragments).
  - `causal_effect_on_spatial` (string): A single JSON string listing MACRO spatial effects AFTER the ENTIRE step completes as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (resulting contacts/containment/support/alignment/orientation/open-closed changes; avoid short, generic fragments).
  - `causal_effect_on_affordance` (string): A single JSON string listing MACRO affordance/state effects AFTER the ENTIRE step completes as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (resulting functional/state changes; avoid short, generic fragments).
- `counterfactual_challenge_question` (string): One realistic counterfactual what-if question that could disrupt this step due to physics/constraints, grounded in the scene. MUST start with 'What if ...?'. This field is ONLY about a counterfactual disruption; do NOT mix in non-counterfactual failure analysis. Do NOT mention frames/images/timestamps.
- `expected_challenge_outcome` (string): Predicted physical outcome if that counterfactual challenge occurs (specific failure/deviation), in one concise English sentence (no frame/time references).
- `failure_reflecting` (object): Real (non-counterfactual) failure analysis and recovery:
  - `reason` (string): Most plausible real failure mode for this step (physical/interaction reason), grounded in what is visible and the mechanism (avoid invisible/unknown causes).
  - `recovery_strategy` (string): A concrete, physically plausible recovery action that would still achieve the step_goal (do not introduce new unseen tools/objects).
- `critical_frames` (list): MUST contain exactly 2 objects, in strictly increasing time order. These are the two most causally important, visually anchorable key moments within the step (NOT limited to initiation/completion).
  Each `critical_frames[*]` object contains:
  - `frame_index` (int): 1-based index into THIS step-clip frame pool (1..{num_frames}); the 2 indices must be distinct and strictly increasing.
  - `action_state_change_description` (string): Describe the micro-action at this key moment and the key state that begins changing, with discriminative contacts/spatial relations/orientation/open-closed cues; objective and grounded in visual evidence; no frame/time references.
  - `causal_chain` (object): Keyframe-level causal analysis with EXACTLY these 4 fields (no agent/action/patient):
    - `causal_precondition_on_spatial` (string): A single JSON string listing DETAILED spatial preconditions TRUE AT this key moment as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this key moment (avoid short, generic fragments).
    - `causal_precondition_on_affordance` (string): A single JSON string listing DETAILED affordance/state preconditions REQUIRED AT this key moment as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this key moment (avoid short, generic fragments).
    - `causal_effect_on_spatial` (string): A single JSON string listing PREDICTED immediate, local spatial effects right AFTER the micro-action implied by action_state_change_description completes (short-term/local post-action prediction; not necessarily currently visible) as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this key moment (avoid short, generic fragments).
    - `causal_effect_on_affordance` (string): A single JSON string listing PREDICTED immediate, local affordance/state effects right AFTER that micro-action completes (short-term/local post-action prediction; not necessarily currently visible) as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this key moment (avoid short, generic fragments).
  - `interaction` (object): MUST contain ONLY these 3 keys (do NOT output tools/materials and do NOT nest a `hotspot` object):
    - `description` (string): Specific functional region involved (e.g., handle, rim, edge, hinge); keep it concrete and visually grounded.
    - `affordance_type` (string): One `snake_case` token describing this region's functional role (e.g., grasp_point, pressing_surface, contact_surface).
    - `mechanism` (string): Brief physical mechanism describing how interaction at this region achieves the micro-action (force/torque transfer, friction, leverage, flow, etc.), grounded in what is visible.

Output JSON template (fill with real values; keep keys exactly; `critical_frames` MUST have exactly 2 entries):
{{
  "step_id": 1,
  "step_goal": "Exactly equal to the draft step_goal (do not change).",
  "rationale": "1–2 grounded sentences explaining why this step is necessary and what it enables.",
  "causal_chain": {{
    "agent": "hands",
    "action": "rotate to loosen",
    "patient": "snake_case_patient_object",
    "causal_precondition_on_spatial": "1. ...\\n2. ...",
    "causal_precondition_on_affordance": "1. ...\\n2. ...",
    "causal_effect_on_spatial": "1. ...\\n2. ...",
    "causal_effect_on_affordance": "1. ...\\n2. ..."
  }},
  "counterfactual_challenge_question": "What if ...?",
  "expected_challenge_outcome": "One concise sentence describing the predicted outcome.",
  "failure_reflecting": {{
    "reason": "A plausible failure mode for this step.",
    "recovery_strategy": "A concise recovery strategy."
  }},
  "critical_frames": [
    {{
      "frame_index": 1,
      "action_state_change_description": "Key moment 1 (earlier than Key moment 2): describe the micro-action and key state change onset.",
      "causal_chain": {{
        "causal_precondition_on_spatial": "1. ...\\n2. ...",
        "causal_precondition_on_affordance": "1. ...\\n2. ...",
        "causal_effect_on_spatial": "1. ...\\n2. ...",
        "causal_effect_on_affordance": "1. ...\\n2. ..."
      }},
      "interaction": {{
        "description": "Specific functional region involved (e.g., handle, rim, edge, hinge).",
        "affordance_type": "grasp_point",
        "mechanism": "Explain the physical mechanism grounded in what is visible."
      }}
    }},
    {{
      "frame_index": 2,
      "action_state_change_description": "Key moment 2 (later than Key moment 1): describe the micro-action and key state change onset.",
      "causal_chain": {{
        "causal_precondition_on_spatial": "1. ...\\n2. ...",
        "causal_precondition_on_affordance": "1. ...\\n2. ...",
        "causal_effect_on_spatial": "1. ...\\n2. ...",
        "causal_effect_on_affordance": "1. ...\\n2. ..."
      }},
      "interaction": {{
        "description": "Specific functional region involved (edge, handle, rim, hinge, etc.).",
        "affordance_type": "contact_surface",
        "mechanism": "Explain the physical mechanism grounded in what is visible."
      }}
    }}
  ]
}}

High-level goal (context): {high_level_goal}

Draft plan outline (read-only; for coherence across steps; do not modify step_goals):
{draft_plan_outline}

Reference draft step JSON (read-only; do not echo it in output):
```json
{draft_step_json}
```
""".strip()
