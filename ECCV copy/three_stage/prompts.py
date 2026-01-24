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
1) Scan all frames quickly to understand the step progression.
2) Pick 2 frames that best represent (a) initiation and (b) completion of the step, with clear visual evidence.
3) Treat each keyframe as a conjunction of constraints: the selected `frame_index` MUST be consistent with its own
   `action_state_change_description`, `causal_chain`, and `interaction` simultaneously (avoid partial matches).
4) If multiple frames match similarly well, prefer the EARLIER index.

Strict requirements:
- You MUST NOT change `step_id` or `step_goal` from the draft.
- You MUST output exactly 2 `critical_frames`.
- Each `critical_frames[*].frame_index` MUST be an integer in [1, {num_frames}] and refers to the step-clip frame pool provided here.
- Do NOT reference frame/image numbers (e.g., "Frame 12", "Image 12") in any text fields; use only the numeric `frame_index` field to specify frames.
- Do NOT reference timestamps/durations/timecodes in any text fields (e.g., "3 seconds", "00:03", "t=3.2s").
- Choose 2 DISTINCT frames that show meaningful temporal progression (initiation → completion); do not pick duplicates.
- The indices within `critical_frames` must be in increasing time order.
- Do NOT output `keyframe_image_path` (keyframe JPEGs are resolved from the filesystem by the script).
- Output strict JSON only; no explanations.
- Do NOT add any extra keys beyond the schema below.
- The step-level `causal_chain.agent`, `causal_chain.action`, `causal_chain.patient` MUST be identical to the corresponding fields in EACH `critical_frames[*].causal_chain` (copy them verbatim).
- `causal_chain.agent/action/patient` MUST be non-empty strings.
- `causal_chain.causal_precondition_on_spatial`, `causal_chain.causal_precondition_on_affordance`, `causal_chain.causal_effect_on_spatial`, `causal_chain.causal_effect_on_affordance` MUST be non-empty lists.
- `causal_chain.causal_precondition_on_affordance[*].reasons` and `causal_chain.causal_effect_on_affordance[*].reasons` MUST be non-empty grounded strings.
- `interaction.hotspot.description`, `interaction.hotspot.affordance_type`, `interaction.hotspot.mechanism` MUST be non-empty grounded strings.
- `interaction.tools` and `interaction.materials` MUST be lists; BOTH must be non-empty (use "hands" as a tool if no external tool is used).
- In each `critical_frames[*].interaction`, `tools` MUST include the `causal_chain.agent`, and `materials` MUST include the `causal_chain.patient` (keep identifiers consistent).
- All `truth` fields are JSON booleans (`true`/`false`), not strings.

Quality and grounding constraints:
- Treat the frames as the ONLY source of truth. Do not hallucinate objects, contacts, or states not supported by the images.
- `causal_chain.causal_precondition_on_spatial` must be visually verifiable in the chosen frame (contacts, containment, support, relative pose, reachability).
- `causal_chain.causal_precondition_on_affordance` must be grounded in visible functional state (e.g., graspable handle exposed; blade contacting object; container open).
- `causal_chain` should describe the physical constraints and the causal effects for the whole step (consistent with the chosen keyframes).
- `interaction.hotspot` must refer to a specific functional region that is visibly involved (edge, handle, rim, hinge, etc.).
- `interaction.hotspot.mechanism` should explain the physical mechanism (force/torque transfer, friction, leverage, fluid flow, heat transfer, stress concentration, etc.).
- Avoid placeholders like "N/A", "unknown", or empty strings; fill all required fields with grounded, specific content.
- Use consistent object naming across all fields (causal_chain, interaction); do not rename the same object with different synonyms within the step.
- Prefer concrete relation verbs in `causal_precondition_on_spatial` and `causal_effect_on_spatial` (examples: "contacting", "holding", "on_top_of", "inside", "inserted_into", "aligned_with", "open", "closed").
- Use `snake_case` for object identifiers where possible.
- Keep `agent` as a person/body part (e.g., "hand", "hands", "finger", "person"); keep `action` as a verb phrase; keep `patient` as the acted-on object.

Output schema (strict):
(Do NOT add any extra keys beyond this schema. Your output must be valid JSON with real values.)

Top-level fields (one step JSON object):
- `step_id` (int): Must equal the draft `step_id` exactly (read-only).
- `step_goal` (string): Must exactly equal the draft `step_goal` (read-only; do not rephrase or paraphrase).
- `rationale` (string): 1–3 sentences explaining why this step is necessary and what it causally enables/changes (mechanistic and grounded; do NOT mention frame numbers).
- `causal_chain` (object): Step-level physical causal analysis for the WHOLE step; must be consistent with both keyframes (see `CausalChain` below).
- `counterfactual_challenge_question` (string): A realistic what-if that changes a physical precondition (e.g., access, friction, alignment, openness), not a question about frames.
- `expected_challenge_outcome` (string): Predicted physical outcome for the what-if, with brief physical reasoning.
- `failure_reflecting` (object):
  - `reason` (string): Plausible failure mode for this step (what goes wrong physically/procedurally).
  - `recovery_strategy` (string): Concrete, actionable recovery strategy that would plausibly fix the failure.
- `critical_frames` (list; MUST contain exactly 2 objects, in strictly increasing time order):
  Each `critical_frames[*]` object contains:
  - `frame_index` (int): 1-based index into THIS step-clip frame pool (1..{num_frames}); the 2 indices must be distinct.
  - `action_state_change_description` (string): Observable action/state change at this frame (describe objects + state; do NOT write "Frame X" in text).
  - `causal_chain` (object): Keyframe-level causal analysis; SAME `CausalChain` schema as the step-level one (preconditions should be true at/just before this frame).
  - `interaction` (object):
    - `tools` (list[string]): Force applicators (use "hands" if no external tool); prefer `snake_case` identifiers; MUST be non-empty and MUST include the step's `causal_chain.agent` (same identifier).
    - `materials` (list[string]): Manipulated objects/substances; prefer `snake_case`; keep naming consistent with `causal_chain` entities; MUST be non-empty and MUST include the step's `causal_chain.patient` (same identifier).
    - `hotspot` (object):
      - `description` (string): Specific functional region involved (e.g., handle, rim, edge, hinge); keep it concrete and visually grounded.
      - `affordance_type` (string): One token describing the hotspot's functional role (prefer `snake_case`, e.g., "grasp_point", "cutting_edge", "pour_spout").
      - `mechanism` (string): Brief physical mechanism explaining how interaction at the hotspot achieves the action (force/torque transfer, friction, leverage, flow, etc.).

`CausalChain` (used in BOTH `causal_chain` and `critical_frames[*].causal_chain`):
- `agent` (string): Primary force/controller for the step (prefer body part like "hands"/"right_hand"; use a tool part only if it is clearly the direct force applicator).
- `action` (string): Concise verb phrase describing the core physical action (ideally includes the physical mechanism: push/pull/rotate/tilt/insert/press, etc.).
- `patient` (string): Primary entity being acted upon (`snake_case` identifier; keep it consistent across all fields in the step).
- `causal_precondition_on_spatial` (list; non-empty): Spatial/physical relations that MUST hold immediately before and throughout the step (concrete, visually verifiable).
- `causal_precondition_on_affordance` (list; non-empty): Functional affordances/states that MUST already be true to execute the action (grounded in visible cues).
- `causal_effect_on_spatial` (list; non-empty): Spatial/physical relations that become true/false as a RESULT of completing the step.
- `causal_effect_on_affordance` (list; non-empty): Functional affordances/states that change as a RESULT of completing the step.

`SpatialRelation` (elements of `*_on_spatial` lists):
- `relation` (string): Short, concrete, visually verifiable token (prefer mechanistic relations like "holding", "contacting", "inside", "aligned_with", "open", "closed"; avoid full sentences).
- `objects` (list[string]): Involved entities (`snake_case`; non-empty list; typically two entities).
- `truth` (bool): Whether the relation holds. For preconditions this is usually true; for effects, set it to the post-step truth value (true = established, false = broken).

`AffordanceState` (elements of `*_on_affordance` lists):
- `object_name` (string): The object whose functional affordance/state is asserted (`snake_case`; grounded in the frames).
- `affordance_types` (list[string]): One or more affordance/state tokens (`snake_case`; keep tokens short and functional; non-empty list).
- `reasons` (string): Grounded justification referencing visible cues and physical constraints/mechanism (no speculation).

Output JSON template (fill with real values; keep keys exactly; `critical_frames` MUST have exactly 2 entries):
{{
  "step_id": 1,
  "step_goal": "Exactly equal to the draft step_goal (do not change).",
  "rationale": "Why this step is necessary for the overall plan, explained causally (how it enables later steps).",
  "causal_chain": {{
    "agent": "hands",
    "action": "A concise verb phrase summarizing the core physical action for the WHOLE step.",
    "patient": "snake_case_patient_object",
    "causal_precondition_on_spatial": [
      {{
        "relation": "holding",
        "objects": ["hands", "snake_case_patient_object"],
        "truth": true
      }}
    ],
    "causal_precondition_on_affordance": [
      {{
        "object_name": "snake_case_patient_object",
        "affordance_types": ["graspable"],
        "reasons": "Grounded justification referencing visible cues and why this affordance/state is required."
      }}
    ],
    "causal_effect_on_spatial": [
      {{
        "relation": "aligned_with",
        "objects": ["snake_case_patient_object", "snake_case_target_object"],
        "truth": true
      }}
    ],
    "causal_effect_on_affordance": [
      {{
        "object_name": "snake_case_patient_object",
        "affordance_types": ["position_changed"],
        "reasons": "Grounded justification of how the action causes this affordance/state change."
      }}
    ]
  }},
  "counterfactual_challenge_question": "A realistic what-if question challenging the physical understanding of this step.",
  "expected_challenge_outcome": "The predicted physical outcome for the challenge question.",
  "failure_reflecting": {{
    "reason": "A plausible failure mode for this step.",
    "recovery_strategy": "A concise recovery strategy."
  }},
  "critical_frames": [
    {{
      "frame_index": 1,
      "action_state_change_description": "Initiation: what observable action/state change begins at this frame.",
      "causal_chain": {{
        "agent": "hands",
        "action": "A concise verb phrase summarizing the core physical action for the WHOLE step.",
        "patient": "snake_case_patient_object",
        "causal_precondition_on_spatial": [
          {{
            "relation": "contacting",
            "objects": ["hands", "snake_case_patient_object"],
            "truth": true
          }}
        ],
        "causal_precondition_on_affordance": [
          {{
            "object_name": "snake_case_patient_object",
            "affordance_types": ["reachable"],
            "reasons": "Grounded justification referencing visible cues and why this affordance/state is required."
          }}
        ],
        "causal_effect_on_spatial": [
          {{
            "relation": "moving_toward",
            "objects": ["snake_case_patient_object", "snake_case_target_object"],
            "truth": true
          }}
        ],
        "causal_effect_on_affordance": [
          {{
            "object_name": "snake_case_patient_object",
            "affordance_types": ["in_motion"],
            "reasons": "Grounded justification of how the action causes this affordance/state change."
          }}
        ]
      }},
      "interaction": {{
        "tools": ["hands"],
        "materials": ["snake_case_patient_object"],
        "hotspot": {{
          "description": "Specific functional region involved (e.g., handle, rim, edge, hinge).",
          "affordance_type": "grasp_point",
          "mechanism": "Explain the physical mechanism grounded in what is visible."
        }}
      }}
    }},
    {{
      "frame_index": 2,
      "action_state_change_description": "Completion: what observable action/state change is achieved at this frame.",
      "causal_chain": {{
        "agent": "hands",
        "action": "A concise verb phrase summarizing the core physical action for the WHOLE step.",
        "patient": "snake_case_patient_object",
        "causal_precondition_on_spatial": [
          {{
            "relation": "aligned_with",
            "objects": ["snake_case_patient_object", "snake_case_target_object"],
            "truth": true
          }}
        ],
        "causal_precondition_on_affordance": [
          {{
            "object_name": "snake_case_target_object",
            "affordance_types": ["available"],
            "reasons": "Grounded justification referencing visible cues and why this affordance/state is required."
          }}
        ],
        "causal_effect_on_spatial": [
          {{
            "relation": "inside",
            "objects": ["snake_case_patient_object", "snake_case_target_object"],
            "truth": true
          }}
        ],
        "causal_effect_on_affordance": [
          {{
            "object_name": "snake_case_patient_object",
            "affordance_types": ["placed"],
            "reasons": "Grounded justification of how the action causes this affordance/state change."
          }}
        ]
      }},
      "interaction": {{
        "tools": ["hands"],
        "materials": ["snake_case_patient_object"],
        "hotspot": {{
          "description": "Specific functional region involved (edge, handle, rim, hinge, etc.).",
          "affordance_type": "contact_surface",
          "mechanism": "Explain the physical mechanism grounded in what is visible."
        }}
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
