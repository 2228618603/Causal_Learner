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
- In particular, do NOT output `critical_frames`, `frame_index`, or `keyframe_image_path` anywhere.
- Do NOT reference frame/image indices or timestamps in any field (e.g., "Frame 12", "Image 12", "t=3.2s").

Language & grounding:
- Use objective, professional English.
- Use conservative naming (e.g., "container", "tool", "vegetable") when unsure.
- Do not hallucinate hidden states or off-screen objects.
- Keep the plan at a granularity that is realistic to localize with {num_frames} sampled frames (prefer 5–8 steps; keep within 4–9).
- Use consistent object naming across all steps (do not rename the same object with different synonyms).
- Prefer `snake_case` for object identifiers in lists (e.g., `objects`, `object_name`, tools/material names).
- Avoid placeholders like "unknown", "N/A", "..." — fill every field with grounded, specific content.

Output format (strict JSON only; no extra text):
{{
  "high_level_goal": "One comprehensive English sentence describing the overall goal of the entire video.",
  "steps": [
    {{
      "step_id": 1,
      "step_goal": "A specific, action-oriented description of the sub-goal for this step.",
      "rationale": "Why this step is necessary for the overall plan.",
      "preconditions": ["Macro-level states that must be true before this step."],
      "expected_effects": ["Macro-level states that will be true after this step (include spatial + state/affordance outcomes)."],
      "spatial_postconditions_detail": [{{"relation": "A concrete spatial/state relation after the step completes.", "objects": ["object_a", "object_b"], "truth": true}}],
      "affordance_postconditions_detail": [{{"object_name": "object", "affordance_types": ["affordance_a"], "reasons": "Why the affordance/state holds after completion."}}],
      "predicted_next_actions": ["2–4 short, plausible next micro-actions that would immediately follow this step (consistent with the overall plan)."],
      "tool_and_material_usage": {{
        "tools": ["Tools actively used in this step."],
        "materials": ["Materials being acted on/consumed in this step."]
      }},
      "causal_challenge_question": "A realistic what-if question challenging the physical understanding of this step.",
      "expected_challenge_outcome": "The predicted physical outcome for the challenge question.",
      "failure_handling": {{
        "reason": "A plausible failure mode for this step.",
        "recovery_strategy": "A concise recovery strategy."
      }}
    }}
  ]
}}

Additional constraints:
- Step ordering MUST follow the chronological order implied by the frames.
- `step_id` MUST start at 1 and increase by 1.
- Each `step_goal` must be non-empty, specific, and not duplicated across steps.
- Keep each `step_goal` concise and focused on a single sub-goal (prefer <= 12 words).
- `preconditions` and `expected_effects` MUST be non-empty lists.
- `spatial_postconditions_detail`, `affordance_postconditions_detail`, and `predicted_next_actions` MUST be non-empty lists.
- `predicted_next_actions` MUST contain 2–4 items.
- Items in list fields should be DISTINCT (no duplicates), especially in `predicted_next_actions` and tool/material lists.
- `affordance_postconditions_detail[*].reasons` MUST be a non-empty, grounded explanation (no placeholders or tautologies).
- `failure_handling.reason` and `failure_handling.recovery_strategy` MUST be non-empty.
- `tool_and_material_usage` MUST include at least one tool or material (use "hands" as a tool if no external tool is used).
- Ensure cross-step causal consistency: Step i `expected_effects` should make Step i+1 `preconditions` plausible (avoid contradictions).
- Do NOT add any extra keys beyond the schema above.
- Each step should be anchorable to visual evidence, but you are NOT selecting frames in this stage.

Silent self-check before you output:
- Strict valid JSON only (double quotes, no trailing commas, no markdown fences).
- No frame/image index references anywhere.
- No empty lists/strings for required fields; no placeholder values.

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
- The 1-based frame index may also be overlaid on each image as text like "Frame 07". Treat that as the same index you must output.
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
- Do NOT output `{num_frames + 1}` (end_frame_index must be within 1..{num_frames}).
- Do NOT output seconds/timestamps; output indices only.
- Do NOT add/remove/reorder steps. Output must cover exactly the draft step_ids.
- Output must contain exactly one entry per draft `step_id` and MUST NOT include any extra step_ids.
- Do NOT change `step_goal` (it is fixed by the draft).
- Enforce monotonic, non-overlapping segments: for consecutive steps, `end_i <= start_(i+1)`.
- Enforce positive duration in the sampled timeline: `start_frame_index < end_frame_index` for every step.
- Prefer near-contiguous coverage across steps (often `end_i == start_(i+1)`), unless there is clear idle time or a real gap between actions.
- Prefer full coverage of the video: typically `start_1 == 1` and `end_last == {num_frames}` unless the video clearly begins/ends with irrelevant idle content.
- Do NOT add any semantic annotations or extra fields beyond the required indices.
- Each entry in `steps` MUST contain exactly these keys: `step_id`, `start_frame_index`, `end_frame_index` (no `step_goal`, no notes, no confidence).
- Output MUST be exactly one JSON object with a single top-level key `steps` (no other top-level keys).
- Output steps in ascending `step_id` order.
- The example below is illustrative only; your output MUST include exactly one entry per draft `step_id`.

Silent self-check before you output:
- All step_ids included exactly once; no extra ids.
- All indices are integers within [1, {num_frames}].
- For every step: start < end.
- For every consecutive pair: end_i <= start_(i+1).

Output format (strict JSON only):
{{
  "steps": [
    {{"step_id": 1, "start_frame_index": 1, "end_frame_index": 2}},
    {{"step_id": 2, "start_frame_index": 2, "end_frame_index": 5}}
  ]
}}
""".strip()


def build_stage3_user_prompt(high_level_goal: str, draft_plan_outline: str, draft_step_json: str, num_frames: int) -> str:
    return f"""
You are an expert Physical Interaction Analyst and Causal Planner.
You are given {num_frames} uniformly sampled frames from a SINGLE STEP CLIP (chronological order), and the draft step definition (read-only).

Task (Stage 3):
Using the step-clip frames as the PRIMARY evidence, refine and complete the annotation for this step and generate 1–2 keyframe annotations.

Keyframe selection procedure (recommended; follow silently):
1) Scan all frames quickly to understand the step progression.
2) Pick 1–2 frames that best represent (a) initiation and/or (b) completion of the step, with clear visual evidence.
3) Treat each keyframe as a conjunction of constraints: the selected `frame_index` MUST be consistent with its own
   `action_description`, `state_change_description`, `spatial_preconditions`, `affordance_preconditions`, `causal_chain`,
   and `affordance_hotspot` simultaneously (avoid partial matches).
4) If multiple frames match similarly well, prefer the EARLIER index. If the clip is highly duplicated/too short, output only 1 keyframe.

Strict requirements:
- You MUST NOT change `step_id` or `step_goal` from the draft.
- You MUST output exactly 1 or 2 `critical_frames`.
- Each `critical_frames[*].frame_index` MUST be an integer in [1, {num_frames}] and refers to the step-clip frame pool provided here.
- Do NOT reference frame/image numbers (e.g., "Frame 12", "Image 12") in any text fields; use only the numeric `frame_index` field to specify frames.
- If you output 2 critical frames, choose 2 DISTINCT frames that show meaningful temporal progression (initiation → completion); do not pick duplicates.
- If the clip is too short / many sampled frames are duplicates, output only 1 `critical_frame` rather than forcing 2 near-identical moments.
- The indices within `critical_frames` must be in increasing time order.
- Do NOT output `keyframe_image_path` (it will be added by the script).
- Output strict JSON only; no explanations.
- Do NOT add any extra keys beyond the schema below.
- `preconditions` and `expected_effects` MUST be non-empty lists.
- `spatial_postconditions_detail`, `affordance_postconditions_detail`, and `predicted_next_actions` MUST be non-empty.
- `predicted_next_actions` MUST contain 2–4 short items.
- Items in `predicted_next_actions` MUST be DISTINCT (no duplicates).
- Items in other list fields should also be DISTINCT where applicable (e.g., tools/materials, object/affordance identifier lists).
- Each `critical_frames[*].spatial_preconditions` and `critical_frames[*].affordance_preconditions` MUST be non-empty lists.
- `affordance_postconditions_detail[*].reasons` and `critical_frames[*].affordance_preconditions[*].reasons` MUST be non-empty grounded strings.
- `causal_chain.causal_effect_on_patient` and `causal_chain.causal_effect_on_environment` MUST be non-empty strings.
- `affordance_hotspot.mechanism` MUST be a non-empty, grounded string (how the hotspot transmits force/motion/constraint to cause the effect).
- `affordance_hotspot.affordance_type` SHOULD be a short `snake_case` identifier (no spaces).
- `tool_and_material_usage` MUST include at least one tool or material (use "hands" as a tool if no external tool is used).

Quality and grounding constraints:
- Treat the frames as the ONLY source of truth. Do not hallucinate objects, contacts, or states not supported by the images.
- `spatial_preconditions` must be visually verifiable in the chosen frame (contacts, containment, support, relative pose, reachability).
- `affordance_preconditions` must be grounded in visible functional state (e.g., graspable handle exposed; blade contacting object; container open).
- `causal_chain` should describe the ongoing physical interaction at that instant using precise language.
- `affordance_hotspot` must refer to a specific functional region that is visibly involved (edge, handle, rim, hinge, etc.).
- `affordance_hotspot.mechanism` should explain the physical mechanism (force/torque transfer, friction, leverage, fluid flow, heat transfer, stress concentration, etc.).
- Avoid placeholders like "N/A", "unknown", or empty strings; fill all required fields with grounded, specific content.
- Use consistent object naming across all fields (spatial/affordance preconditions, causal_chain, hotspot); do not rename the same object with different synonyms within the step.
- Prefer concrete relation verbs in `spatial_preconditions` (examples: "contacting", "holding", "on_top_of", "inside", "inserted_into", "aligned_with", "open", "closed").
- Use `snake_case` for object identifiers where possible (e.g., "cutting_board", "light_switch").
- Keep `agent` as a person/body part (e.g., "hand", "hands", "finger", "person"); do not use camera-centric agents; keep `action` as a verb phrase; keep `patient` as the acted-on object.

Output schema (strict):
(`<>` markers below are placeholders for types; do NOT output them literally. Your output must be valid JSON with real values.)
{{
  "step_id": <int>,
  "step_goal": <string exactly equal to the draft>,
  "rationale": <string>,
  "preconditions": [<string>, ...],
  "expected_effects": [<string>, ...],
  "spatial_postconditions_detail": [{{"relation": <string>, "objects": [<string>, ...], "truth": <bool>}}, ...],
  "affordance_postconditions_detail": [{{"object_name": <string>, "affordance_types": [<string>, ...], "reasons": <string>}}, ...],
  "predicted_next_actions": [<string>, ...],
  "tool_and_material_usage": {{"tools": [<string>, ...], "materials": [<string>, ...]}},
  "causal_challenge_question": <string>,
  "expected_challenge_outcome": <string>,
  "failure_handling": {{"reason": <string>, "recovery_strategy": <string>}},
  "critical_frames": [
    {{
      "frame_index": <int>,
      "action_description": <string>,
      "state_change_description": <string>,
      "spatial_preconditions": [{{"relation": <string>, "objects": [<string>, ...], "truth": <bool>}}, ...],
      "affordance_preconditions": [{{"object_name": <string>, "affordance_types": [<string>, ...], "reasons": <string>}}, ...],
      "causal_chain": {{
        "agent": <string>,
        "action": <string>,
        "patient": <string>,
        "causal_effect_on_patient": <string>,
        "causal_effect_on_environment": <string>
      }},
      "affordance_hotspot": {{
        "description": <string>,
        "affordance_type": <string>,
        "mechanism": <string>
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
