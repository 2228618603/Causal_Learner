# -*- coding: utf-8 -*-
"""
Causal-SPaFA-Plan Dataset Generation Script
Version: 6.9 (Definitive Fix for API Response Indexing)

This script processes a long video, uniformly samples frames, sends them to an
LMM, and parses the response. It follows the same structure and formatting as
the medium/short variants, and ensures correct API response handling by
indexing `choices[0]` before reading `message.content`.
"""

import base64
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
import json
import os
import re
import time
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

# Suppress OpenAI's internal httpx logging to keep the console clean
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==============================================================================
# === 1. CONFIGURATION PARAMETERS ==============================================
# ==============================================================================

# *** NEW CONFIGURATION: ABSOLUTE PATH TO VIDEO FOLDER ***
# Please modify this path to the absolute path of the folder containing your videos.

@dataclass
class ScriptConfig:
    """Centralized configuration for the script."""
    # --- API Credentials ---
    API_KEY: str = os.environ.get("API_KEY", "EMPTY")
    API_BASE_URL: str = "http://model.mify.ai.srv/v1"
    MODEL_PROVIDER_ID: str = "vertex_ai"
    # MODEL_PROVIDER_ID: str = "volcengine_maas"
    MODEL_NAME: str = "gemini-3-pro-preview"
    # MODEL_NAME: str = "doubao-1-5-thinking-vision-pro-250428"

    # --- Input/Output Paths ---
    VIDEO_PATH: str = "example.mp4" # This will be updated dynamically in the loop
    OUTPUT_BASE_FOLDER: str = "causal_spafa_plan_dataset_long"
    # OUTPUT_BASE_FOLDER: str = "causal_spafa_plan_dataset_seed"

    # --- Video Processing ---
    MAX_FRAMES_TO_SAMPLE: int = 50
    RESIZE_DIMENSION: Tuple[int, int] = None  # e.g., (1280, 720) or None
    # JPEG_QUALITY: int = 100
    JPEG_QUALITY: int = 100
    # --- Planning Constraints ---
    # Desired step count for Stage 1 plan (inclusive). Controls granularity and avoids fragmented numbering.
    PLAN_MIN_STEPS: int = int(os.environ.get("PLAN_MIN_STEPS", "4"))
    PLAN_MAX_STEPS: int = int(os.environ.get("PLAN_MAX_STEPS", "7"))
    # --- Script Behavior ---
    VERBOSE_LOGGING: bool = True
    # Overlay frame index/timestamp onto images sent to the API to avoid off-by-one confusion
    EMBED_INDEX_ON_API_IMAGES: bool = True

# Instantiate the configuration
# The VIDEO_PATH will be overwritten for each file in the folder.
PLANNING_CONFIG = ScriptConfig(
    VIDEO_PATH="placeholder.mp4", 
    OUTPUT_BASE_FOLDER="causal_plan_dataset_gemini",
    API_KEY=os.environ.get("API_KEY", "EMPTY"),
    API_BASE_URL="http://model.mify.ai.srv/v1",
    MODEL_PROVIDER_ID="vertex_ai",
    MODEL_NAME="gemini-3-pro-preview",
    VERBOSE_LOGGING=True,
)

SELECTION_CONFIG = ScriptConfig(
    VIDEO_PATH="placeholder.mp4",
    OUTPUT_BASE_FOLDER="causal_spafa_plan_dataset_long",
    API_KEY=os.environ.get("API_KEY", "EMPTY"),
    API_BASE_URL="http://model.mify.ai.srv/v1",
    MODEL_PROVIDER_ID="vertex_ai",
    MODEL_NAME="gemini-3-pro-preview",
    VERBOSE_LOGGING=True,
)

# ==============================================================================
# === 2. DATA STRUCTURE DEFINITIONS (SCHEMA) ===================================
# ==============================================================================

@dataclass
class StepCausalChain:
    """Macro physical causal reasoning structure for an entire step."""

    agent: str
    action: str
    patient: str
    causal_precondition_on_spatial: str
    causal_precondition_on_affordance: str
    causal_effect_on_spatial: str
    causal_effect_on_affordance: str


@dataclass
class FrameCausalChain:
    """Frame-level causal reasoning for a key moment within a step.

    NOTE: This frame-level causal_chain MUST NOT include agent/action/patient.
    """

    causal_precondition_on_spatial: str
    causal_precondition_on_affordance: str
    causal_effect_on_spatial: str
    causal_effect_on_affordance: str


@dataclass
class Interaction:
    """Keyframe interaction details (flattened; no tools/materials/hotspot nesting)."""

    description: str
    affordance_type: str
    mechanism: str

@dataclass
class CriticalFrameAnnotation:
    """Complete annotation for a single pivotal moment."""

    # 1-based in the sampled frame pool; omitted in Stage 1 and added in Stage 2.
    frame_index: Optional[int]
    action_state_change_description: str
    causal_chain: FrameCausalChain
    interaction: Interaction

@dataclass
class FailureReflecting:
    """Describes a plausible failure and recovery strategy."""

    reason: str
    recovery_strategy: str

@dataclass
class PlanningStep:
    """A single step in the hierarchical plan."""

    step_id: int
    step_goal: str
    rationale: str
    causal_chain: StepCausalChain
    counterfactual_challenge_question: str
    expected_challenge_outcome: str
    failure_reflecting: FailureReflecting
    critical_frames: List[CriticalFrameAnnotation]

def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _parse_step_causal_chain(data: Any) -> StepCausalChain:
    """Parse step-level `causal_chain` while tolerating missing fields."""
    if not isinstance(data, dict):
        data = {}
    return StepCausalChain(
        agent=str(data.get("agent", "")),
        action=str(data.get("action", "")),
        patient=str(data.get("patient", "")),
        causal_precondition_on_spatial=_coerce_str(data.get("causal_precondition_on_spatial", "")),
        causal_precondition_on_affordance=_coerce_str(data.get("causal_precondition_on_affordance", "")),
        causal_effect_on_spatial=_coerce_str(data.get("causal_effect_on_spatial", "")),
        causal_effect_on_affordance=_coerce_str(data.get("causal_effect_on_affordance", "")),
    )


def _parse_frame_causal_chain(data: Any) -> FrameCausalChain:
    """Parse frame-level `causal_chain` while tolerating missing fields."""
    if not isinstance(data, dict):
        data = {}
    return FrameCausalChain(
        causal_precondition_on_spatial=_coerce_str(data.get("causal_precondition_on_spatial", "")),
        causal_precondition_on_affordance=_coerce_str(data.get("causal_precondition_on_affordance", "")),
        causal_effect_on_spatial=_coerce_str(data.get("causal_effect_on_spatial", "")),
        causal_effect_on_affordance=_coerce_str(data.get("causal_effect_on_affordance", "")),
    )


def _parse_interaction(data: Any) -> Interaction:
    if not isinstance(data, dict):
        data = {}
    hotspot = data.get("hotspot", {}) if isinstance(data.get("hotspot"), dict) else {}
    description = data.get("description") or hotspot.get("description") or ""
    affordance_type = data.get("affordance_type") or hotspot.get("affordance_type") or ""
    mechanism = data.get("mechanism") or hotspot.get("mechanism") or ""
    return Interaction(
        description=str(description),
        affordance_type=str(affordance_type),
        mechanism=str(mechanism),
    )


def _parse_failure_reflecting(data: Any) -> FailureReflecting:
    if not isinstance(data, dict):
        data = {}
    return FailureReflecting(
        reason=str(data.get("reason", "")),
        recovery_strategy=str(data.get("recovery_strategy", "")),
    )

# ==============================================================================
# === 3. CORE UTILITY FUNCTIONS ================================================
# ==============================================================================

def initialize_api_client(config: ScriptConfig) -> Any:
    """Initializes and returns the OpenAI API client."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
            default_headers={"X-Model-Provider-Id": config.MODEL_PROVIDER_ID}
        )
        print(">>> [SUCCESS] OpenAI client initialized successfully.")
        return client
    except ImportError:
        print("!!! [FATAL] 'openai' library not found. Please run 'pip install openai'.")
        return None
    except Exception as e:
        print(f"!!! [FATAL] Failed to initialize OpenAI client: {e}")
        return None

def process_video_to_frames(config: ScriptConfig) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """Extracts, resizes, and base64-encodes frames uniformly from a video."""
    if cv2 is None:
        print("!!! [FATAL] 'opencv-python' (cv2) is not installed. Please run 'pip install opencv-python'.")
        return [], None
    print(f"\n>>> [INFO] Starting video processing for: {config.VIDEO_PATH}")
    if not os.path.exists(config.VIDEO_PATH):
        print(f"!!! [ERROR] Video file not found: {config.VIDEO_PATH}")
        return [], None
    video_capture = cv2.VideoCapture(config.VIDEO_PATH)
    if not video_capture.isOpened():
        print(f"!!! [ERROR] Cannot open video file: {config.VIDEO_PATH}")
        return [], None
    frame_data_list, original_dimensions = [], None
    try:
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_dimensions = (width, height)
        if total_frames == 0 or fps == 0:
            print("!!! [ERROR] Video has 0 frames or 0 FPS.")
            return [], original_dimensions
        print(f">>> [INFO] Video Details: {total_frames} frames, {fps:.2f} FPS, ({width}x{height})")
        frame_indices = [int(i * total_frames / config.MAX_FRAMES_TO_SAMPLE) for i in range(config.MAX_FRAMES_TO_SAMPLE)]
        for i, frame_idx in enumerate(frame_indices):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if not success: continue
            if config.RESIZE_DIMENSION:
                frame = cv2.resize(frame, config.RESIZE_DIMENSION)
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
            frame_data_list.append({
                "base64": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                "timestamp_sec": frame_idx / fps,
                "original_frame_index": frame_idx
            })
    finally:
        video_capture.release()
    print(f">>> [SUCCESS] Video processing complete. Extracted {len(frame_data_list)} frames.")
    return frame_data_list, original_dimensions

def save_keyframe_images(config: ScriptConfig, annotations: List[CriticalFrameAnnotation], step_output_path: str, all_frame_data: List[Dict[str, Any]]):
    """Extracts and saves the original keyframe images.

    Note: `annotation.frame_index` is treated as 1-based. Internally converted
    to 0-based when indexing `all_frame_data`.
    """
    print(f"  -> Saving {len(annotations)} keyframe images...")
    if cv2 is None:
        print("    !!! [FATAL] 'opencv-python' (cv2) is not installed. Cannot save keyframe images.")
        return
    video_capture = cv2.VideoCapture(config.VIDEO_PATH)
    if not video_capture.isOpened():
        print(f"    !!! [ERROR] Cannot re-open video to save keyframes: {config.VIDEO_PATH}")
        return
    try:
        for anno in annotations:
            try:
                # Convert 1-based index to 0-based for internal lookup
                if anno.frame_index is None:
                    print("    !!! [WARNING] Missing frame_index. Skipping.")
                    continue
                idx0 = int(anno.frame_index) - 1
                if idx0 < 0 or idx0 >= len(all_frame_data):
                    print(f"    !!! [WARNING] Invalid 1-based frame_index {anno.frame_index}. Skipping.")
                    continue
                frame_info = all_frame_data[idx0]
                original_frame_idx = frame_info["original_frame_index"]
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
                success, frame = video_capture.read()
                if not success:
                    print(f"    !!! [WARNING] Failed to capture frame at index {original_frame_idx}.")
                    continue
                if config.RESIZE_DIMENSION:
                    frame = cv2.resize(frame, config.RESIZE_DIMENSION)
                filename = f"frame_{anno.frame_index:03d}_ts_{frame_info['timestamp_sec']:.2f}s.jpg"
                filepath = os.path.join(step_output_path, filename)
                cv2.imwrite(filepath, frame)
            except Exception as e:
                print(f"    !!! [ERROR] Error saving keyframe for index {anno.frame_index}: {e}")
    finally:
        video_capture.release()

def save_sampled_frames_jpegs(sampled_frames: List[Dict[str, Any]], output_dir: str):
    """Save all uniformly sampled frames (JPEG) to `output_dir`.

    Naming starts at 1 (not 0): `sample_001_ts_YY.YYs.jpg`.
    Frames are written directly from their base64-encoded JPEG buffers.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"!!! [ERROR] Failed to create frames output dir '{output_dir}': {e}")
        return

    count = 0
    for i, frame in enumerate(sampled_frames):
        try:
            ts = float(frame.get("timestamp_sec", 0.0))
            idx1 = i + 1  # 1-based numbering for saved filenames
            name = f"sample_{idx1:03d}_ts_{ts:.2f}s.jpg"
            path = os.path.join(output_dir, name)
            data = base64.b64decode(frame["base64"]) if isinstance(frame.get("base64"), str) else None
            if not data:
                print(f"    !!! [WARNING] Missing base64 for frame {i}. Skipping.")
                continue
            with open(path, "wb") as f:
                f.write(data)
            count += 1
        except Exception as e:
            print(f"    !!! [WARNING] Failed to save sampled frame {i}: {e}")
    print(f"  -> Saved {count}/{len(sampled_frames)} sampled frames to: {output_dir}")

def build_index_manifest(sampled_frames: List[Dict[str, Any]]) -> str:
    """Return a textual manifest mapping 1-based index to timestamp seconds."""
    lines = ["Frame Index Manifest (1-based):"]
    for i, frame in enumerate(sampled_frames, start=1):
        ts = float(frame.get("timestamp_sec", 0.0))
        lines.append(f"- Frame {i}: t={ts:.2f}s")
    return "\n".join(lines)

def _overlay_index_on_base64_image(b64_img: str, index_1based: int, timestamp_sec: float) -> str:
    """Overlay index and timestamp onto a base64 JPEG image and return new base64."""
    try:
        import numpy as np
        data = base64.b64decode(b64_img)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return b64_img
        text = f"Frame {index_1based:02d}  t={timestamp_sec:.2f}s"
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return b64_img
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return b64_img

def build_api_content(
    sampled_frames: List[Dict[str, Any]],
    embed_index: bool,
    *,
    include_manifest: bool = True,
    include_frame_labels: bool = True,
) -> List[Dict[str, Any]]:
    """Build message content list for the multimodal API in 1-based order.

    - When include_manifest is True, adds an index+timestamp manifest text block.
    - When include_frame_labels is True, adds per-image "Frame N" text items.
    - When embed_index is True, overlays the 1-based frame index + timestamp onto each image.
    """
    content: List[Dict[str, Any]] = []
    if include_manifest:
        manifest = build_index_manifest(sampled_frames)
        content.append({"type": "text", "text": manifest})
    for i, frame in enumerate(sampled_frames, start=1):
        ts = float(frame.get("timestamp_sec", 0.0))
        b64 = frame.get("base64")
        if embed_index and isinstance(b64, str):
            b64 = _overlay_index_on_base64_image(b64, i, ts)
        if include_frame_labels:
            content.append({"type": "text", "text": f"Frame {i}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content

def sanitize_filename(text: str) -> str:
    """Cleans a string to be a valid folder/file name."""
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[-\s]+', '_', text)
    return text

_FRAME_REF_RE = re.compile(r"\b(frame|image|img|picture)\s*\d+\b", re.IGNORECASE)
_FRAME_REF_ORDINAL_RE = re.compile(
    r"\b(initial|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|beginning|ending)\s+"
    r"(frame|image|img|picture)\b",
    re.IGNORECASE,
)


def _contains_frame_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_FRAME_REF_RE.search(s) or _FRAME_REF_ORDINAL_RE.search(s))


_TIME_REF_RE = re.compile(
    r"(?:"
    r"\bt\s*=\s*\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"
    r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"
    r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b"
    r")",
    re.IGNORECASE,
)


def _contains_time_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_TIME_REF_RE.search(s))


def _assert_no_disallowed_refs(text: Any, *, label: str) -> None:
    if _contains_frame_ref(text):
        raise ValueError(f"{label} must not reference frame/image indices (e.g., 'Frame 12').")
    if _contains_time_ref(text):
        raise ValueError(f"{label} must not reference timestamps/durations/timecodes (e.g., 't=3.2s', '00:03').")

def extract_json_from_response(response_text: str) -> str:
    """Extracts a JSON string from the model's response using multiple strategies."""
    if not isinstance(response_text, str):
        raise ValueError("Input to extract_json_from_response was not a string.")

    # Strategy 1: Look for a markdown ```json ... ``` block
    match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
    if match:
        print(">>> [INFO] Strategy 1: Found JSON within a markdown block.")
        return match.group(1).strip()

    # Strategy 2: If no markdown, find the first '{' and last '}'
    print(">>> [INFO] Strategy 1 failed. Trying Strategy 2: Find first '{' and last '}' characters.")
    start_brace = response_text.find('{')
    end_brace = response_text.rfind('}')
    
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        return response_text[start_brace : end_brace + 1].strip()

    raise ValueError("Could not find a valid JSON structure in the model's response.")

# ==============================================================================
# === 4. PROMPT ENGINEERING ====================================================
# ==============================================================================

# *** SYSTEM PROMPT ***
system_prompt = """
You are a highly advanced AI acting as a Physical Interaction Analyst and Causal Planner. Your primary mission is to deconstruct observed actions in video frames into their fundamental causal, spatial, and affordance-based physical principles.
You must analyze key moments from a continuous action sequence to produce a hierarchical task plan. This plan must explain not just *what* happened, but precisely *how* and *why* it is happening from a physical standpoint by inferring the dynamics implied within each key moment.
Your output MUST be a single, syntactically flawless JSON object. The level of detail and adherence to the causal schema is paramount.
CRITICAL: Your plan MUST cover the entire video timeline from the first provided frame to the last provided frame; do not omit late-stage events.
"""

def create_planning_user_prompt(
    num_frames: int,
    image_dimensions: Tuple[int, int],
    *,
    min_steps: int,
    max_steps: int,
) -> str:
    """Generates the full, detailed user prompt for the LMM API call."""
    return f"""
You are a world-class AI, a doctorate-level expert in physics, robotics, and cognitive science, acting as a **Physical Interaction Analyst and Causal Planner**. Your primary mission is to deconstruct observed human actions from video frames into their most fundamental causal, kinetic, and physical principles. You must think step-by-step with extreme precision, logical rigor, and unwavering adherence to the specified JSON schema. Your output is not just a description; it is a scientific annotation.

Analyze the provided {num_frames} frames, which are uniformly sampled from a continuous video of a task. Your task is to reverse-engineer the high-level goal and generate a deeply detailed, **hierarchical causal plan**. This plan MUST be broken down into a moderate number of medium-grained, logical steps (**between {min_steps} and {max_steps} steps inclusive**).

Treat the frames as the ONLY source of truth. Use conservative language when uncertain and prefer generic object naming (e.g., "container", "bottle", "tool") over guessing brands or invisible states.

Your response MUST be a single, syntactically flawless JSON object. No extra text, no apologies, no explanations outside of the JSON structure. The JSON validity is a critical, non-negotiable part of the task.

**Detailed JSON Schema to Follow (output strict JSON; keep keys exactly; MUST omit `critical_frames[*].frame_index`:**
{{
    "high_level_goal": "One comprehensive English sentence describing the overall goal and intended final outcome of the entire video (final world state; do NOT list steps; do NOT mention frames/images/timestamps).",
    "steps": [
        {{
            "step_id": 1,
            "step_goal": "One English sentence describing the intended intermediate world-state outcome of this step as a single coherent phase (avoid listing multiple independent actions; no frame/time references).",
            "rationale": "Grounded sentences explaining WHY this step is necessary: (a) what macro physical/spatial/affordance preconditions it assumes across the entire step, and (b) what macro effects it establishes across the entire step that enable later steps. Do NOT just restate step_goal.",
            "causal_chain": {{
                "agent": "Primary force/controller for the whole step (prefer body part like 'hands'/'left_hand'/'right_hand'; use tool part only if it is clearly the direct force applicator). Use one stable identifier and keep it consistent within the step.",
                "action": "Physical verb phrase for the whole step (include mechanism when possible: push/pull/rotate/tilt/insert/press). Avoid vague verbs like 'do'/'move'.",
                "patient": "Primary acted-on object identifier in snake_case. Keep naming consistent across all fields (do not rename the same object).",
                "causal_precondition_on_spatial": "A single JSON string listing MACRO spatial preconditions for the ENTIRE step as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (avoid short, generic fragments).",
                "causal_precondition_on_affordance": "A single JSON string listing MACRO affordance/state preconditions for the ENTIRE step as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (avoid short, generic fragments).",
                "causal_effect_on_spatial": "A single JSON string listing MACRO spatial effects AFTER the ENTIRE step completes as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (resulting contacts/containment/support/alignment/orientation/open-closed changes; avoid short, generic fragments).",
                "causal_effect_on_affordance": "A single JSON string listing MACRO affordance/state effects AFTER the ENTIRE step completes as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this step (resulting functional/state changes; avoid short, generic fragments)."
            }},
            "counterfactual_challenge_question": "One realistic counterfactual what-if question that could disrupt this step due to physics/constraints, grounded in the scene. Start with 'What if ...?'. This field is ONLY about a counterfactual disruption; do NOT mix in non-counterfactual failure analysis. Do NOT mention frames/images/timestamps.",
            "expected_challenge_outcome": "Predicted physical outcome if that counterfactual challenge occurs (specific failure/deviation), in one English sentence (no frame/time references).",
            "failure_reflecting": {{
                "reason": "Most plausible real (non-counterfactual) failure mode for this step (physical/interaction reason), grounded in what is visible and the mechanism (avoid invisible/unknown causes).",
                "recovery_strategy": "A concrete, physically plausible recovery action that would still achieve the step_goal (do not introduce new unseen tools/objects)."
            }},
            "critical_frames": [
                {{
                    "action_state_change_description": "Key moment 1 (earlier in time than Key moment 2; NOT limited to initiation/completion): describe the micro-action at this moment and the key state that begins changing, with discriminative contacts/spatial relations/orientation/open-closed cues; objective and grounded in visual evidence; no frame/time references.",
                    "causal_chain": {{
                        "causal_precondition_on_spatial": "A single JSON string listing DETAILED spatial preconditions TRUE AT this key moment as numbered points. Formatting rules: numbered '1. ', '2. ', ...; separate points using escaped '\\n' inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each line must be complete, objective, and tightly tied to this key moment.",
                        "causal_precondition_on_affordance": "A single JSON string listing DETAILED affordance/state preconditions REQUIRED AT this key moment as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this key moment (avoid short, generic fragments).",
                        "causal_effect_on_spatial": "A single JSON string listing PREDICTED immediate, local spatial effects right AFTER the micro-action implied by action_state_change_description completes (short-term/local post-action prediction; not necessarily currently visible). Formatting rules: numbered lines; separated by escaped '\\n' inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each line must be physically grounded and tightly tied to this micro-action.",
                        "causal_effect_on_affordance": "A single JSON string listing PREDICTED immediate, local affordance/state effects right AFTER that micro-action completes (short-term/local post-action prediction; not necessarily currently visible). Formatting rules: numbered lines; separated by escaped '\\n' inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each line must be physically grounded and tightly tied to this micro-action."
                    }},
                    "interaction": {{
                        "description": "Specific functional region involved (e.g., handle, rim, edge, hinge); keep it concrete and visually grounded. NOTE: Do NOT output tools/materials/hotspot; interaction MUST contain ONLY description/affordance_type/mechanism.",
                        "affordance_type": "One snake_case token describing this region's functional role (e.g., grasp_point, pressing_surface, contact_surface).",
                        "mechanism": "Physical mechanism: how interaction at this region achieves the micro-action (force/torque transfer, friction, leverage, flow, etc.), grounded in what is visible."
                    }}
                }},
                {{
                    "action_state_change_description": "Key moment 2 (later in time than Key moment 1; NOT limited to initiation/completion): describe the micro-action at this moment and the key state that begins changing, with discriminative contacts/spatial relations/orientation/open-closed cues; objective and grounded in visual evidence; no frame/time references.",
                    "causal_chain": {{
                        "causal_precondition_on_spatial": "A single JSON string listing DETAILED spatial preconditions TRUE AT this key moment as numbered points. Formatting rules: numbered '1. ', '2. ', ...; separate points using escaped '\\n' inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each line must be complete, objective, and tightly tied to this key moment.",
                        "causal_precondition_on_affordance": "A single JSON string listing DETAILED affordance/state preconditions REQUIRED AT this key moment as numbered points. Formatting rules: (1) Use lines numbered '1. ', '2. ', ...; (2) do NOT put raw newlines inside a JSON string; instead separate points using the escaped sequence '\\n' inside the string; (3) each numbered line may contain multiple sentences, but the last character of EACH numbered line MUST be '.' (mandatory); (4) each numbered line MUST be a complete, objective English statement tightly tied to this key moment (avoid short, generic fragments).",
                        "causal_effect_on_spatial": "A single JSON string listing PREDICTED immediate, local spatial effects right AFTER the micro-action implied by action_state_change_description completes (short-term/local post-action prediction; not necessarily currently visible). Formatting rules: numbered lines; separated by escaped '\\n' inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each line must be physically grounded and tightly tied to this micro-action.",
                        "causal_effect_on_affordance": "A single JSON string listing PREDICTED immediate, local affordance/state effects right AFTER that micro-action completes (short-term/local post-action prediction; not necessarily currently visible). Formatting rules: numbered lines; separated by escaped '\\n' inside the string; each numbered line may contain multiple sentences but MUST end with '.' (mandatory); each line must be physically grounded and tightly tied to this micro-action."
                    }},
                    "interaction": {{
                        "description": "Specific functional region involved (e.g., handle, rim, edge, hinge); keep it concrete and visually grounded. NOTE: Do NOT output tools/materials/hotspot; interaction MUST contain ONLY description/affordance_type/mechanism.",
                        "affordance_type": "One snake_case token describing this region's functional role (e.g., grasp_point, pressing_surface, contact_surface).",
                        "mechanism": "Physical mechanism: how interaction at this region achieves the micro-action (force/torque transfer, friction, leverage, flow, etc.), grounded in what is visible."
                    }}
                }}
            ]
        }}
    ]
}}

**CRITICAL INSTRUCTIONS TO FOLLOW AT ALL COSTS:**

1.  **Extreme Detail and Objectivity:** Every description must be highly detailed, objective, and grounded in visual evidence.
2.  **Scientific Causal Reasoning:** All fields within `causal_chain` MUST be plausible and consistent with the principles of physics and dynamics.
3.  **Focus on the Key Moment:** In each `critical_frames[*]`, `causal_chain.causal_precondition_on_spatial` and `causal_chain.causal_precondition_on_affordance` MUST comprehensively describe the state of the world TRUE AT that specific key moment. In each `critical_frames[*]`, `causal_chain.causal_effect_on_spatial` and `causal_chain.causal_effect_on_affordance` MUST describe the PREDICTED immediate, local post-action effects right after the micro-action implied by `action_state_change_description` completes (short-term/local; not necessarily currently visible).
4.  **Key Moments & Exactly Two Critical Frames:** Each `step` MUST contain exactly 2 `critical_frames`, ordered in time (Key moment 1 earlier than Key moment 2). These are the two most causally important, visually anchorable, representative moments within the step (NOT limited to initiation/completion).
5.  **Infer Dynamics from a Snapshot:** Your descriptions must infer motion, force, and consequence from a single, static key frame.
6.  **Complete All Fields:** Ensure every single textual field in the schema is filled with a meaningful and accurate value.
7.  **Critical Frames :** In this Stage 1 planning phase, DO NOT include any image references or frame indices in the JSON. In particular, OMIT `critical_frames[*].frame_index` and do NOT output `keyframe_image_path` anywhere. Only provide the textual fields for each `critical_frames[*]` entry (`action_state_change_description`, `causal_chain`, `interaction`). Frame selection happens later in Stage 2, where `frame_index` is 1-based in [1, {num_frames}].
8.  **Grounding & Non‑Hallucination:** Base all facts strictly on what is visible in the provided frames. Do not invent objects, brands, or states that are not visually supported. If uncertain, use generic terms.
9.  **Consistency Requirement:** Use consistent object names across steps and frames. Prefer a stable canonical name.
10. **JSON Structure is Paramount:** Adhere strictly to the schema.
11. **Causal Text Formatting is Mandatory:** For every `causal_*` field (both step-level and frame-level), output a single JSON string with numbered points using `1. ...`, `2. ...`, etc. Separate points inside the string using the escaped sequence `\\n` (no raw newlines inside a JSON string). Avoid overly short or generic fragments; each line must be a complete, objective statement tightly tied to the current step or key moment.
12. **Step count & numbering (STRICT):** You MUST output between {min_steps} and {max_steps} steps (inclusive). `step_id` MUST be consecutive integers starting at 1 with NO gaps (1..N). Before finalizing, self-check that the number of steps is within range and the numbering is contiguous.
13. **Balanced step granularity:** Avoid over-splitting into tiny micro-steps. If the video implies many micro-actions, MERGE adjacent micro-actions into a single step when they belong to the same phase and do not create a new stable intermediate world state. Each step should represent a coherent, contiguous phase (early → middle → late) and the steps should be roughly balanced in temporal coverage (avoid one step being extremely short while another spans most of the sequence).
14. **Hard full-video coverage constraint (NON-NEGOTIABLE):** The ordered `steps` MUST collectively cover the ENTIRE timeline of all {num_frames} images (early → middle → late), including what happens in the FINAL portion of the video. Do NOT compress the whole plan into only the early/middle frames; later steps MUST reflect later-video events. The plan MUST NOT end early: the LAST step MUST include and reflect the last frames. Do NOT invent an "achieved final state" if the video ends mid-action; describe the last observed state and any visible ongoing action.
15. **Coverage audit (MANDATORY):** Before finalizing your JSON, internally verify that every meaningful action/state change visible across the {num_frames} frames is assigned to exactly one step. If any late-frame events or final outcomes are missing, REVISE step boundaries and merge/summarize earlier content as needed — but still keep the step count within {min_steps}..{max_steps}.
16. **Key moments must be meaningful:** Ensure every described `critical_frames[*]` corresponds to a visually distinct, high-value moment within its step (a decisive micro-action or discriminative state transition). Do NOT choose key moments just to "cover" early/late frames. For the LAST step, Key moment 2 should represent the most informative late-stage moment within that step (the clearest decisive micro-action, transition, or stable outcome state that best characterizes the step). If the outcome persists across many frames, it may be satisfied by the earliest frame where the state first becomes true and stable (not necessarily the last frame of the video). Crucially, write each key moment so it can be matched to a SINGLE sampled frame: include discriminative, visibly checkable cues (contact/containment/open-closed/alignment/posture intent) and avoid hidden-state claims or vague timing language.
17. **Principle of Visual Anchorability:** While you are not selecting frames now, every critical_frame you describe must correspond to a visually distinct and unambiguous moment. Your descriptions should be "anchor-able" to a plausible visual snapshot (a single image) with concrete, visible evidence. Do not describe events that are inherently invisible or highly ambiguous from a third-person perspective.
---

TEMPORAL ALIGNMENT REQUIREMENTS (DO NOT IGNORE):
1) The ordering of your `steps` MUST strictly follow the chronological order of the {num_frames} frames as provided (earliest frames first, latest frames last).
2) Do NOT reorder events out of time; ensure that earlier events described in earlier frames appear in earlier steps.
3) Within each step, maintain descriptions consistent with the earlier-to-later progression implied by the frames.
4) Do NOT end the plan early: the final step(s) MUST explicitly account for what happens in the last frames and conclude at the true final state of the video.

Now, based on the uniformly sampled frames I have provided from a continuous video, and adhering strictly to all constraints (including temporal alignment) and the highest standards of quality, generate the complete and detailed JSON output for this video.
"""

# ==============================================================================
# === 5. MAIN EXECUTION LOGIC ==================================================
# ==============================================================================

def _coerce_json_bool(value: Any) -> bool:
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
    return False


_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\\u2022)\s+")
_LEADING_NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)、]\s*")


def _normalize_numbered_statement_block(text: str) -> str:
    """Normalize a multi-line text block into '1. ...' numbered statements.

    - Each statement occupies one line.
    - Lines are re-numbered sequentially starting from 1.
    - Each numbered line MUST end with '.'.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    # If a model output included literal "\\n" sequences, convert them into real newlines.
    raw = raw.replace("\\n", "\n")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        lines = [raw]

    normalized: List[str] = []
    for i, line in enumerate(lines, start=1):
        line = _LEADING_LIST_MARKER_RE.sub("", line)
        line = _LEADING_NUMBER_RE.sub("", line)
        line = line.strip()
        if not line:
            continue
        if not line.endswith("."):
            line = f"{line}."
        normalized.append(f"{i}. {line}")
    return "\n".join(normalized)


def _spatial_item_to_statement(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        relation = str(item.get("relation", "")).strip()
        objects = item.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        objects = [str(o).strip() for o in objects if str(o).strip()]
        truth = _coerce_json_bool(item.get("truth", True))
        rel = relation or "unspecified_relation"
        if objects:
            objs = ", ".join(objects)
            return f"Relation '{rel}' {'holds' if truth else 'does not hold'} between {objs}."
        return f"Relation '{rel}' {'holds' if truth else 'does not hold'}."
    return _coerce_str(item).strip()


def _affordance_item_to_statement(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        object_name = str(item.get("object_name", "")).strip()
        affordance_types = item.get("affordance_types", [])
        if not isinstance(affordance_types, list):
            affordance_types = []
        affordance_types = [str(a).strip() for a in affordance_types if str(a).strip()]
        reasons = str(item.get("reasons", "")).strip()

        obj = object_name or "unspecified_object"
        aff = ", ".join(affordance_types) if affordance_types else "unspecified_affordance"
        if reasons:
            return f"The object {obj} has affordance/state {aff}. {reasons}"
        return f"The object {obj} has affordance/state {aff}."
    return _coerce_str(item).strip()


def _normalize_causal_text(value: Any, *, kind: str) -> str:
    """Normalize a causal_* field into a numbered statement block string.

    Supports:
    - New schema: a single string with numbered points.
    - Legacy schema: a list of structured dicts for spatial/affordance.
    """
    if isinstance(value, list):
        if kind == "spatial":
            statements = [_spatial_item_to_statement(v) for v in value]
        else:
            statements = [_affordance_item_to_statement(v) for v in value]
        # Join as lines, then normalize numbering/periods.
        return _normalize_numbered_statement_block("\n".join(s for s in statements if s))

    if isinstance(value, dict):
        if kind == "spatial":
            stmt = _spatial_item_to_statement(value)
        else:
            stmt = _affordance_item_to_statement(value)
        return _normalize_numbered_statement_block(stmt)

    return _normalize_numbered_statement_block(_coerce_str(value))


def _normalize_interaction(interaction: Any) -> Dict[str, str]:
    """Normalize keyframe interaction into flattened schema with 3 keys."""
    if not isinstance(interaction, dict):
        interaction = {}
    hotspot = interaction.get("hotspot", {}) if isinstance(interaction.get("hotspot"), dict) else {}
    description = interaction.get("description") or hotspot.get("description") or ""
    affordance_type = interaction.get("affordance_type") or hotspot.get("affordance_type") or ""
    mechanism = interaction.get("mechanism") or hotspot.get("mechanism") or ""
    return {
        "description": str(description),
        "affordance_type": str(affordance_type),
        "mechanism": str(mechanism),
    }


def _normalize_step_causal_chain(causal_chain: Any) -> Dict[str, Any]:
    if not isinstance(causal_chain, dict):
        causal_chain = {}
    return {
        "agent": str(causal_chain.get("agent", "")),
        "action": str(causal_chain.get("action", "")),
        "patient": str(causal_chain.get("patient", "")),
        "causal_precondition_on_spatial": _normalize_causal_text(
            causal_chain.get("causal_precondition_on_spatial"), kind="spatial"
        ),
        "causal_precondition_on_affordance": _normalize_causal_text(
            causal_chain.get("causal_precondition_on_affordance"), kind="affordance"
        ),
        "causal_effect_on_spatial": _normalize_causal_text(causal_chain.get("causal_effect_on_spatial"), kind="spatial"),
        "causal_effect_on_affordance": _normalize_causal_text(
            causal_chain.get("causal_effect_on_affordance"), kind="affordance"
        ),
    }


def _normalize_frame_causal_chain(causal_chain: Any) -> Dict[str, Any]:
    if not isinstance(causal_chain, dict):
        causal_chain = {}
    # Remove prohibited keys if present (legacy outputs).
    causal_chain = {k: v for k, v in causal_chain.items() if k not in ("agent", "action", "patient")}
    return {
        "causal_precondition_on_spatial": _normalize_causal_text(
            causal_chain.get("causal_precondition_on_spatial"), kind="spatial"
        ),
        "causal_precondition_on_affordance": _normalize_causal_text(
            causal_chain.get("causal_precondition_on_affordance"), kind="affordance"
        ),
        "causal_effect_on_spatial": _normalize_causal_text(causal_chain.get("causal_effect_on_spatial"), kind="spatial"),
        "causal_effect_on_affordance": _normalize_causal_text(
            causal_chain.get("causal_effect_on_affordance"), kind="affordance"
        ),
    }


def _normalize_plan_schema(plan: Any, *, min_steps: int = 4, max_steps: int = 7) -> Dict[str, Any]:
    """Normalize a plan dict into the latest schema while preserving semantics.

    This makes Stage 2 robust to legacy/partially-noncompliant Stage 1 outputs and
    enforces key schema invariants to protect final dataset quality.
    """
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")
    if min_steps < 1:
        raise ValueError(f"min_steps must be >= 1, got {min_steps}.")
    if max_steps < min_steps:
        raise ValueError(f"max_steps must be >= min_steps, got min_steps={min_steps}, max_steps={max_steps}.")

    high_level_goal = str(plan.get("high_level_goal", "")).strip()
    if not high_level_goal:
        raise ValueError("Missing or empty 'high_level_goal'.")
    _assert_no_disallowed_refs(high_level_goal, label="high_level_goal")

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("'steps' must be a list.")

    normalized_steps: List[Dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue

        # Canonicalize step_id to be contiguous (1..N) in list order. This avoids
        # fragmented numbering like 1,2,3,7 and stabilizes downstream Stage 2.
        step_id = len(normalized_steps) + 1

        step_goal = str(raw_step.get("step_goal", "")).strip()
        rationale = str(raw_step.get("rationale", "")).strip()
        counterfactual_q = str(raw_step.get("counterfactual_challenge_question", "")).strip()
        expected_outcome = str(raw_step.get("expected_challenge_outcome", "")).strip()

        failure_reflecting = raw_step.get("failure_reflecting", {})
        if not isinstance(failure_reflecting, dict):
            failure_reflecting = {}
        failure_reason = str(failure_reflecting.get("reason", "")).strip()
        recovery_strategy = str(failure_reflecting.get("recovery_strategy", "")).strip()

        cfs = raw_step.get("critical_frames", [])
        if not isinstance(cfs, list):
            raise ValueError(f"step_id={step_id} critical_frames must be a list.")
        if len(cfs) != 2:
            raise ValueError(f"step_id={step_id} must contain exactly 2 critical_frames, got {len(cfs)}.")

        normalized_cfs: List[Dict[str, Any]] = []
        for cf in cfs:
            if not isinstance(cf, dict):
                cf = {}
            normalized_cfs.append(
                {
                    "action_state_change_description": str(cf.get("action_state_change_description", "")).strip(),
                    "causal_chain": _normalize_frame_causal_chain(cf.get("causal_chain", {})),
                    "interaction": _normalize_interaction(cf.get("interaction", {})),
                }
            )

        normalized_step = {
            "step_id": step_id,
            "step_goal": step_goal,
            "rationale": rationale,
            "causal_chain": _normalize_step_causal_chain(raw_step.get("causal_chain", {})),
            "counterfactual_challenge_question": counterfactual_q,
            "expected_challenge_outcome": expected_outcome,
            "failure_reflecting": {
                "reason": failure_reason,
                "recovery_strategy": recovery_strategy,
            },
            "critical_frames": normalized_cfs,
        }

        # Enforce required textual fields are present (quality gate).
        required_str_fields = [
            ("step_goal", normalized_step["step_goal"]),
            ("rationale", normalized_step["rationale"]),
            ("counterfactual_challenge_question", normalized_step["counterfactual_challenge_question"]),
            ("expected_challenge_outcome", normalized_step["expected_challenge_outcome"]),
            ("failure_reflecting.reason", normalized_step["failure_reflecting"]["reason"]),
            ("failure_reflecting.recovery_strategy", normalized_step["failure_reflecting"]["recovery_strategy"]),
            ("causal_chain.agent", normalized_step["causal_chain"]["agent"]),
            ("causal_chain.action", normalized_step["causal_chain"]["action"]),
            ("causal_chain.patient", normalized_step["causal_chain"]["patient"]),
            ("causal_chain.causal_precondition_on_spatial", normalized_step["causal_chain"]["causal_precondition_on_spatial"]),
            (
                "causal_chain.causal_precondition_on_affordance",
                normalized_step["causal_chain"]["causal_precondition_on_affordance"],
            ),
            ("causal_chain.causal_effect_on_spatial", normalized_step["causal_chain"]["causal_effect_on_spatial"]),
            ("causal_chain.causal_effect_on_affordance", normalized_step["causal_chain"]["causal_effect_on_affordance"]),
        ]
        for name, value in required_str_fields:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"step_id={step_id} missing or empty required field: {name}")
        if not normalized_step["counterfactual_challenge_question"].lstrip().lower().startswith("what if"):
            raise ValueError(
                f"step_id={step_id} counterfactual_challenge_question must start with 'What if ...?'."
            )

        _assert_no_disallowed_refs(normalized_step["step_goal"], label=f"step_id={step_id} step_goal")
        _assert_no_disallowed_refs(normalized_step["rationale"], label=f"step_id={step_id} rationale")
        _assert_no_disallowed_refs(
            normalized_step["counterfactual_challenge_question"],
            label=f"step_id={step_id} counterfactual_challenge_question",
        )
        _assert_no_disallowed_refs(
            normalized_step["expected_challenge_outcome"],
            label=f"step_id={step_id} expected_challenge_outcome",
        )
        _assert_no_disallowed_refs(
            normalized_step["failure_reflecting"]["reason"],
            label=f"step_id={step_id} failure_reflecting.reason",
        )
        _assert_no_disallowed_refs(
            normalized_step["failure_reflecting"]["recovery_strategy"],
            label=f"step_id={step_id} failure_reflecting.recovery_strategy",
        )

        step_cc = normalized_step["causal_chain"]
        _assert_no_disallowed_refs(step_cc["agent"], label=f"step_id={step_id} causal_chain.agent")
        _assert_no_disallowed_refs(step_cc["action"], label=f"step_id={step_id} causal_chain.action")
        _assert_no_disallowed_refs(step_cc["patient"], label=f"step_id={step_id} causal_chain.patient")
        _assert_no_disallowed_refs(
            step_cc["causal_precondition_on_spatial"],
            label=f"step_id={step_id} causal_chain.causal_precondition_on_spatial",
        )
        _assert_no_disallowed_refs(
            step_cc["causal_precondition_on_affordance"],
            label=f"step_id={step_id} causal_chain.causal_precondition_on_affordance",
        )
        _assert_no_disallowed_refs(
            step_cc["causal_effect_on_spatial"],
            label=f"step_id={step_id} causal_chain.causal_effect_on_spatial",
        )
        _assert_no_disallowed_refs(
            step_cc["causal_effect_on_affordance"],
            label=f"step_id={step_id} causal_chain.causal_effect_on_affordance",
        )

        for i, cf in enumerate(normalized_step["critical_frames"], start=1):
            if not cf.get("action_state_change_description"):
                raise ValueError(f"step_id={step_id} critical_frames[{i}] missing action_state_change_description.")
            _assert_no_disallowed_refs(
                cf.get("action_state_change_description", ""),
                label=f"step_id={step_id} critical_frames[{i}].action_state_change_description",
            )
            intr = cf.get("interaction", {})
            if not isinstance(intr, dict):
                raise ValueError(f"step_id={step_id} critical_frames[{i}] interaction must be an object.")
            for k in ("description", "affordance_type", "mechanism"):
                if not str(intr.get(k, "")).strip():
                    raise ValueError(f"step_id={step_id} critical_frames[{i}] interaction.{k} is empty.")
                _assert_no_disallowed_refs(
                    intr.get(k, ""),
                    label=f"step_id={step_id} critical_frames[{i}].interaction.{k}",
                )
            chain = cf.get("causal_chain", {})
            for k in (
                "causal_precondition_on_spatial",
                "causal_precondition_on_affordance",
                "causal_effect_on_spatial",
                "causal_effect_on_affordance",
            ):
                if not str(chain.get(k, "")).strip():
                    raise ValueError(f"step_id={step_id} critical_frames[{i}] causal_chain.{k} is empty.")
                _assert_no_disallowed_refs(
                    chain.get(k, ""),
                    label=f"step_id={step_id} critical_frames[{i}].causal_chain.{k}",
                )

        normalized_steps.append(normalized_step)

    if not normalized_steps:
        raise ValueError("No valid steps found in plan.")

    if not (min_steps <= len(normalized_steps) <= max_steps):
        raise ValueError(
            f"Plan must contain between {min_steps} and {max_steps} steps (inclusive), got {len(normalized_steps)}."
        )

    return {"high_level_goal": high_level_goal, "steps": normalized_steps}


def _filter_plan_remove_keyframe_fields(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied plan with keyframe-specific fields removed.

    Removes 'keyframe_image_path' and 'frame_index' from each item in 'critical_frames'.
    """
    def _clean_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
        new_frame = {k: v for k, v in frame.items() if k not in ("keyframe_image_path", "frame_index")}
        return new_frame

    cleaned = {
        "high_level_goal": plan.get("high_level_goal"),
        "steps": []
    }
    for step in plan.get("steps", []):
        step_copy = {k: v for k, v in step.items() if k != "critical_frames"}
        cfs = step.get("critical_frames", [])
        step_copy["critical_frames"] = [_clean_frame(cf) for cf in cfs]
        cleaned["steps"].append(step_copy)
    return cleaned

def _create_frame_selection_prompt(plan_json_str: str, num_frames: int) -> str:
    """Build a second-stage prompt to select frame indices for each critical frame.

    The model must pick indices in [1, num_frames] for each critical frame per step (1-based),
    based on the provided textual annotations. Output must be strict JSON.
    """
    head_window = max(3, (num_frames + 9) // 10)  # ~first 10% (min 3 frames)
    head_end = min(num_frames, head_window)
    tail_window = max(3, (num_frames + 9) // 10)  # ~last 10% (min 3 frames)
    tail_start = max(1, num_frames - tail_window + 1)
    return f"""
You are an expert vision-time alignment assistant. Your task is to select, with maximal accuracy, the single best-matching frame index for each `critical_frames` entry in a provided plan. You are given: (1) a set of {num_frames} uniformly sampled images from a single video in chronological order, and (2) a detailed JSON plan with rich annotations per step and per critical frame.

Treat the frames as the ONLY source of truth. Your PRIMARY goal is exact visual alignment: the selected `frame_index` for each `critical_frames[*]` MUST make that critical frame's textual description TRUE at that moment. Do NOT "reinterpret" the text; pick the frame that matches it. If the plan text conflicts with visual evidence, select the closest plausible frame, but never select indices that contradict the image (missing required objects/contact) when an alternative exists.

Match Criteria (apply all rigorously):
- Treat each critical frame's fields as conjunctive constraints:
  - `action_state_change_description` must match the visible micro-action and discriminative contacts/orientation/open-closed cues in the image.
  - `critical_frames[*].causal_chain.causal_precondition_on_spatial` and `critical_frames[*].causal_chain.causal_precondition_on_affordance` contain numbered natural-language points; each point is a required precondition that MUST be visually consistent with the chosen image.
  - `critical_frames[*].interaction` contains ONLY `description` / `affordance_type` / `mechanism`; the chosen image should match the described functional region and the implied physical mechanism.
- `critical_frames[*].causal_chain.causal_effect_on_spatial` and `critical_frames[*].causal_chain.causal_effect_on_affordance` are predicted immediate/local post-action effects after the micro-action completes; they may NOT be visible yet. Use them only as a plausibility and temporal-consistency check, not as a hard visual constraint.
- Prefer frames that best satisfy the preconditions AND most clearly depict the micro-action itself (clear contact, pose, alignment, and intent).
- If multiple frames plausibly match, choose the one with the strongest overall alignment. If ties remain, break ties by **key-moment fidelity** (not by being early/late): pick the frame where the described micro-action/state-change is most visually evident and discriminative (clear contact, posture intent, open/closed cue, insertion depth, deformation, flow onset, etc.), and avoid idle/paused frames if a more informative transition frame exists.
- Within a step containing multiple critical frames, enforce STRICTLY increasing indices (Key moment 1 frame_index < Key moment 2 frame_index). The two selected frames MUST be different.
- Across steps, enforce macro-temporal order: earlier steps MUST correspond to earlier indices, and later steps MUST NOT be assigned earlier indices than earlier steps unless absolutely unavoidable due to near-identical frames.
- Do NOT optimize indices for global "coverage". The step partition must cover the full video, but keyframes are for the most causally important, visually anchorable moments within each step.

Alignment self-test (do silently):
- If you showed ONLY the selected image to a human and gave them the `critical_frames[*]` text, would they agree it matches (objects, contact, action, and state) without hand-waving? If not, reject that candidate and choose another.
- Never choose a frame that matches only the step broadly but does NOT match the specific critical frame description (avoid generic "close enough" picks).

Required Procedure (do not skip):
1) Systematically scan all {num_frames} images to identify candidate indices per critical frame.
2) For each candidate, check every listed precondition point for visual plausibility, and verify the micro-action cues and interaction region are consistent.
3) Use the predicted post-action effects only as a plausibility check to break ties (e.g., whether the next immediate state change is physically consistent with the scene).
4) Select the index with maximal constraint satisfaction; break ties by key-moment fidelity (most discriminative action/state-change evidence) while preserving step temporal order.

Internal QA (proofreading) — perform silently before output:
- Pre-filter: Disqualify frames missing required objects or blatant contradictions (e.g., door must be open/closed, contact required but absent).
- Constraint checklist: For each remaining candidate, tick off each numbered precondition point (spatial + affordance/state) as satisfied/unsatisfied, and check interaction region/mechanism plausibility.
- Causal alignment: Verify dynamic cues consistent with the micro-action described in `action_state_change_description` (e.g., grasp vs. lift vs. insert) using posture, relative motion hints, and context.
- Hard alignment gate: Reject any candidate that contradicts the `action_state_change_description`, ANY required precondition point, or the `interaction` description/mechanism when a better-aligned alternative exists. Do NOT accept "close enough" matches.
- Consistency pass: Ensure indices within a step are strictly increasing (Key moment 1 < Key moment 2); across steps, prefer earlier indices for earlier steps. If a violation is detected, re-evaluate candidates to restore temporal consistency.
- Final sanity pass: If two candidates are near-equal, prefer the one satisfying more constraints; if still tied, choose the one with clearer action evidence and less ambiguity (avoid idle/paused frames).

Strictness:
- Do not rewrite or alter any text from the plan.
- Think through QA internally; do NOT include any rationale in the final output.
- Output only the indices you select; no comments, no explanations, no extra keys.
- For each `step_id`, output exactly one `frame_index` per `critical_frames` entry in the reference plan (same order).
- Before outputting, ensure your JSON is syntactically valid (no missing commas, no trailing commas, no unescaped quotes).

OUTPUT FORMAT (strict JSON only):
{{
  "steps": [
    {{
      "step_id": <int>,
      "critical_frames": [
        {{
          "frame_index": <int>
        }},
        ...
      ]
    }},
    ...
  ]
}}

Reference plan JSON (read-only; do not echo or modify in output):
```json
{plan_json_str}
```
"""


def _parse_stage2_selections(sel_data: Any, steps_data: List[Dict[str, Any]], num_frames: int) -> Dict[int, List[int]]:
    """Parse and validate Stage 2 selection output."""
    if not isinstance(sel_data, dict):
        raise ValueError("Stage 2 output must be a JSON object.")
    raw_steps = sel_data.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("Stage 2 output must contain 'steps' as a list.")

    by_id: Dict[int, Dict[str, Any]] = {}
    for st in raw_steps:
        if not isinstance(st, dict):
            continue
        try:
            sid = int(st.get("step_id"))
        except Exception:
            continue
        by_id[sid] = st

    selections: Dict[int, List[int]] = {}
    for step_json in steps_data:
        try:
            expected_sid = int(step_json.get("step_id", -1))
        except Exception:
            continue
        if expected_sid < 0:
            continue

        expected_cfs = step_json.get("critical_frames", [])
        expected_n = len(expected_cfs) if isinstance(expected_cfs, list) else 0

        st_out = by_id.get(expected_sid)
        if st_out is None:
            raise ValueError(f"Stage 2 output missing step_id={expected_sid}.")

        out_cfs = st_out.get("critical_frames", [])
        if not isinstance(out_cfs, list):
            raise ValueError(f"Stage 2 step_id={expected_sid} critical_frames must be a list.")

        idxs: List[int] = []
        for cf in out_cfs:
            if isinstance(cf, dict):
                raw_idx = cf.get("frame_index")
            else:
                raw_idx = cf
            try:
                idx = int(raw_idx)
            except Exception:
                idx = -1
            idxs.append(idx)

        if expected_n and len(idxs) != expected_n:
            raise ValueError(f"Stage 2 step_id={expected_sid} returned {len(idxs)} indices, expected {expected_n}.")

        for idx in idxs:
            if idx < 1 or idx > num_frames:
                raise ValueError(
                    f"Stage 2 step_id={expected_sid} frame_index out of range: {idx} (expected 1..{num_frames})."
                )
        for i in range(1, len(idxs)):
            if idxs[i] <= idxs[i - 1]:
                raise ValueError(f"Stage 2 step_id={expected_sid} indices must be strictly increasing: {idxs}.")

        selections[expected_sid] = idxs

    return selections


def process_single_video(video_file_path: str):
    """Processes a single video file using the defined pipeline with resume support.

    Resume behavior:
    - If final output `causal_plan_with_keyframes.json` exists, skip this video.
    - If only `causal_plan.json` exists, skip Stage 1 and run Stage 2 only.
    - Otherwise, run Stage 1 then Stage 2.
    """
    # Update global configs for the current video
    PLANNING_CONFIG.VIDEO_PATH = video_file_path
    SELECTION_CONFIG.VIDEO_PATH = video_file_path

    print(f"\n==============================================================================")
    print(f"=== PROCESSING VIDEO: {video_file_path}")
    print(f"==============================================================================")

    # Determine output paths and resume flags early
    video_filename_base, _ = os.path.splitext(os.path.basename(PLANNING_CONFIG.VIDEO_PATH))
    video_output_folder = os.path.join(PLANNING_CONFIG.OUTPUT_BASE_FOLDER, video_filename_base)
    sampled_frames_dir = os.path.join(video_output_folder, "sampled_frames")
    try:
        os.makedirs(video_output_folder, exist_ok=True)
    except Exception as e:
        print(f"!!! [ERROR] Failed to create video output folder '{video_output_folder}': {e}")
        return

    stage1_path = os.path.join(video_output_folder, "causal_plan.json")
    stage2_path = os.path.join(video_output_folder, "causal_plan_with_keyframes.json")

    # If Stage 2 already completed, skip entirely
    if os.path.exists(stage2_path):
        print(f">>> [INFO] Final plan already exists. Skipping: {video_output_folder}")
        return

    # Decide whether to resume at Stage 2
    resume_stage2_only = os.path.exists(stage1_path)

    print(">>> [INFO] Script started for single video.")

    # Initialize planning client only if running Stage 1
    planning_client = None
    if not resume_stage2_only:
        planning_client = initialize_api_client(PLANNING_CONFIG)
        if not planning_client:
            return  # Skip this video if client fails

    # 0) Extract frames (always re-extract for reproducibility)
    sampled_frames, original_dims = process_video_to_frames(PLANNING_CONFIG)
    if not sampled_frames:
        print(f"!!! [ERROR] No frames extracted for {video_file_path}. Skipping.")
        return

    # Persist the sampled frames alongside outputs
    save_sampled_frames_jpegs(sampled_frames, sampled_frames_dir)

    response_content = None
    filtered_plan = None
    high_level_goal = None
    steps_data = None

    if not resume_stage2_only:
        # 1) First call: generate plan JSON only (no keyframe images/paths and no frame_index)
        print("\n>>> [INFO] Building API request payload (Stage 1: plan only)...")
        try:
            stage1_min_steps = int(getattr(PLANNING_CONFIG, "PLAN_MIN_STEPS", 4))
            stage1_max_steps = int(getattr(PLANNING_CONFIG, "PLAN_MAX_STEPS", 7))

            # Stage 1 should avoid any frame index/timestamp artifacts in inputs to reduce the chance
            # of the model echoing them in text fields (which is forbidden by schema).
            stage1_frames_content = build_api_content(
                sampled_frames,
                embed_index=False,
                include_manifest=False,
                include_frame_labels=False,
            )
            print(f">>> [SUCCESS] API payload built with {len(sampled_frames)} frames.")
        except Exception as e:
            print(f"!!! [FATAL] Failed to build API request: {e}")
            return

        stage1_retries = int(os.environ.get("STAGE1_RETRIES", "3"))
        stage1_base_delay = float(os.environ.get("STAGE1_RETRY_DELAY_SEC", "3"))
        stage1_max_tokens = int(os.environ.get("STAGE1_MAX_TOKENS", "30000"))
        raw_stage1_path = os.path.join(video_output_folder, "stage1_raw_response.txt")
        last_err: Optional[Exception] = None
        last_content: str = ""
        retry_exact_steps: Optional[int] = None

        for attempt in range(1, stage1_retries + 1):
            try:
                retry_prefix = ""
                if attempt > 1:
                    if retry_exact_steps is not None:
                        retry_prefix = (
                            f"RETRY NOTICE (attempt {attempt}/{stage1_retries}): "
                            f"Your previous output violated the step-count constraint. "
                            f"On this retry, output EXACTLY {retry_exact_steps} steps, and ensure step_id is "
                            f"contiguous 1..{retry_exact_steps} with NO gaps.\n\n"
                        )
                    else:
                        retry_prefix = (
                            f"RETRY NOTICE (attempt {attempt}/{stage1_retries}): "
                            "Your previous output failed schema validation. "
                            "Re-check all constraints (strict JSON only; no frame/time references; each step has "
                            "exactly 2 critical_frames; step count and numbering constraints).\n\n"
                        )

                user_prompt = retry_prefix + create_planning_user_prompt(
                    len(sampled_frames),
                    original_dims,
                    min_steps=stage1_min_steps,
                    max_steps=stage1_max_steps,
                )
                user_content = [{"type": "text", "text": user_prompt}] + stage1_frames_content
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]

                print(
                    f"\n>>> [INFO] Sending request to model '{PLANNING_CONFIG.MODEL_NAME}' (Stage 1) "
                    f"attempt {attempt}/{stage1_retries}..."
                )
                start_time = time.time()
                response = planning_client.chat.completions.create(
                    model=PLANNING_CONFIG.MODEL_NAME,
                    messages=messages,
                    max_tokens=stage1_max_tokens,
                )
                end_time = time.time()

                if not (response and response.choices and len(response.choices) > 0):
                    raise RuntimeError("API response is invalid or does not contain any 'choices'.")
                first_choice = response.choices[0]
                if not hasattr(first_choice, 'message') or not hasattr(first_choice.message, 'content'):
                    raise RuntimeError("The first choice object is missing 'message' or 'content' attributes.")

                response_content = first_choice.message.content
                last_content = str(response_content or "")
                try:
                    with open(raw_stage1_path, "w", encoding="utf-8") as f:
                        f.write(last_content)
                except Exception:
                    pass

                if PLANNING_CONFIG.VERBOSE_LOGGING and last_content:
                    print("\n>>> [DEBUG] Raw API Response Content (Stage 1):")
                    print(last_content)
                print(f">>> [SUCCESS] Stage 1 API call in {end_time - start_time:.2f}s.")

                if not last_content:
                    raise RuntimeError("No content extracted from API response (Stage 1).")

                print("\n>>> [INFO] Parsing JSON response (Stage 1)...")
                clean_json_string = extract_json_from_response(last_content)
                plan_data = json.loads(clean_json_string)

                normalized_plan = _normalize_plan_schema(
                    {"high_level_goal": plan_data.get("high_level_goal"), "steps": plan_data.get("steps")},
                    min_steps=stage1_min_steps,
                    max_steps=stage1_max_steps,
                )
                high_level_goal = normalized_plan.get("high_level_goal", "No Goal Provided")
                steps_data = normalized_plan.get("steps", [])
                print(f">>> [SUCCESS] JSON parsed & normalized. High-Level Goal: {high_level_goal}")

                print(f">>> [IO] Output folder: {video_output_folder}")

                filtered_plan = _filter_plan_remove_keyframe_fields(normalized_plan)
                plan_json_path = os.path.join(video_output_folder, "causal_plan.json")
                with open(plan_json_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_plan, f, indent=4, ensure_ascii=False)
                print(f"\n>>> [SUCCESS] Stage 1 plan saved (no keyframe images/paths/indices) to: {plan_json_path}")

                run_summary = {
                    "source_video": os.path.basename(PLANNING_CONFIG.VIDEO_PATH),
                    "processing_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "models_used": {
                        "planning": PLANNING_CONFIG.MODEL_NAME,
                        "selection": SELECTION_CONFIG.MODEL_NAME
                    },
                    "config_planning": asdict(PLANNING_CONFIG),
                    "config_selection": asdict(SELECTION_CONFIG),
                    "stages": ["plan_only", "frame_selection"]
                }
                summary_json_path = os.path.join(video_output_folder, "run_summary.json")
                with open(summary_json_path, 'w', encoding='utf-8') as f:
                    json.dump(run_summary, f, indent=4, ensure_ascii=False)
                print(f">>> [SUCCESS] Run summary saved to: {summary_json_path}")
                last_err = None
                break
            except (json.JSONDecodeError, ValueError, RuntimeError) as e:
                last_err = e
                retry_exact_steps = None
                if isinstance(e, ValueError) and "Plan must contain between" in str(e):
                    raw_steps = plan_data.get("steps") if isinstance(locals().get("plan_data"), dict) else None
                    raw_step_count = len(raw_steps) if isinstance(raw_steps, list) else None
                    if raw_step_count is not None:
                        if raw_step_count > stage1_max_steps:
                            retry_exact_steps = stage1_max_steps
                        elif raw_step_count < stage1_min_steps:
                            retry_exact_steps = stage1_min_steps
                if attempt >= stage1_retries:
                    break
                delay = stage1_base_delay * (2 ** (attempt - 1))
                print(f"!!! [WARNING] Stage 1 failed (attempt {attempt}/{stage1_retries}): {e}")
                print(f">>> [INFO] Retrying Stage 1 in {delay:.1f}s ...")
                time.sleep(delay)
            except Exception as e:
                last_err = e
                if attempt >= stage1_retries:
                    break
                delay = stage1_base_delay * (2 ** (attempt - 1))
                print(f"!!! [WARNING] Stage 1 unexpected error (attempt {attempt}/{stage1_retries}): {e}")
                print(f">>> [INFO] Retrying Stage 1 in {delay:.1f}s ...")
                time.sleep(delay)

        if last_err is not None or not filtered_plan:
            print(f"\n!!! [FATAL] Stage 1 failed after {stage1_retries} attempts: {last_err}")
            error_log_path = os.path.join(
                PLANNING_CONFIG.OUTPUT_BASE_FOLDER, f"error_response_{video_filename_base}.txt"
            )
            os.makedirs(PLANNING_CONFIG.OUTPUT_BASE_FOLDER, exist_ok=True)
            try:
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(last_content if last_content else str(last_err))
                print(f">>> [INFO] Problematic response saved to: {error_log_path}")
            except Exception as e:
                print(f"!!! [WARNING] Failed to write error log: {e}")
            return
    else:
        # Resume Stage 2 using existing causal_plan.json
        try:
            with open(stage1_path, 'r', encoding='utf-8') as f:
                filtered_plan = json.load(f)
            filtered_plan = _normalize_plan_schema(
                filtered_plan,
                min_steps=int(getattr(PLANNING_CONFIG, "PLAN_MIN_STEPS", 4)),
                max_steps=int(getattr(PLANNING_CONFIG, "PLAN_MAX_STEPS", 7)),
            )
            high_level_goal = filtered_plan.get("high_level_goal", "No Goal Provided")
            steps_data = filtered_plan.get("steps", [])
            print(f"\n>>> [INFO] Resume mode: Loaded existing Stage 1 plan from: {stage1_path}")
        except Exception as e:
            print(f"\n!!! [FATAL] Failed to load existing plan for resume: {e}")
            return

    # 2) Second call: provide plan JSON + images, ask for frame indices
    try:
        print(">>> [INFO] Stage 2: Selecting frames based on saved plan...")
        plan_json_text = json.dumps(filtered_plan, ensure_ascii=False)
        sel_user_prompt = _create_frame_selection_prompt(plan_json_text, len(sampled_frames))
        embed_index = bool(getattr(SELECTION_CONFIG, "EMBED_INDEX_ON_API_IMAGES", True))
        stage2_frames_content = build_api_content(
            sampled_frames,
            embed_index=embed_index,
            include_manifest=True,
            include_frame_labels=not embed_index,
        )

        selection_client = initialize_api_client(SELECTION_CONFIG)
        if not selection_client:
            return

        stage2_max_tokens = int(os.environ.get("STAGE2_MAX_TOKENS", "8000"))
        stage2_temperature = float(os.environ.get("STAGE2_TEMPERATURE", os.environ.get("TEMPERATURE", "0.0")))
        stage2_retries = max(1, int(os.environ.get("STAGE2_MAX_RETRIES", os.environ.get("MAX_RETRIES", "3"))))
        stage2_base_delay = max(
            0.0, float(os.environ.get("STAGE2_RETRY_DELAY_SEC", os.environ.get("RETRY_DELAY_SEC", "3")))
        )

        stage2_raw_path = os.path.join(video_output_folder, "stage2_raw_response.txt")
        selections: Dict[int, List[int]] = {}
        last_err = None

        for attempt in range(1, stage2_retries + 1):
            try:
                if attempt == 1:
                    prompt_text = sel_user_prompt
                else:
                    prompt_text = (
                        "Your previous output was invalid JSON or failed schema validation.\n"
                        f"Error: {last_err}\n"
                        "Regenerate the output as STRICT JSON ONLY using the exact required schema.\n\n"
                        + sel_user_prompt
                    )

                sel_user_content = [{"type": "text", "text": prompt_text}] + stage2_frames_content
                sel_messages = [
                    {
                        "role": "system",
                        "content": "You select frame indices that best match given annotations. Output strict JSON only.",
                    },
                    {"role": "user", "content": sel_user_content},
                ]

                print(f">>> [INFO] Sending request (Stage 2: frame selection) attempt {attempt}/{stage2_retries}...")
                sel_start = time.time()
                sel_resp = selection_client.chat.completions.create(
                    model=SELECTION_CONFIG.MODEL_NAME,
                    messages=sel_messages,
                    max_tokens=stage2_max_tokens,
                    temperature=stage2_temperature,
                )
                sel_end = time.time()
                if not (sel_resp and sel_resp.choices and len(sel_resp.choices) > 0):
                    raise RuntimeError("Stage 2 response invalid or missing choices.")
                sel_choice = sel_resp.choices[0]
                if not hasattr(sel_choice, "message") or not hasattr(sel_choice.message, "content"):
                    raise RuntimeError("Stage 2 missing message.content.")
                sel_content = sel_choice.message.content or ""

                try:
                    with open(stage2_raw_path, "w", encoding="utf-8") as f:
                        f.write(sel_content)
                except Exception:
                    pass

                if SELECTION_CONFIG.VERBOSE_LOGGING:
                    print("\n>>> [DEBUG] Stage 2 raw content:")
                    print(sel_content)
                print(f">>> [SUCCESS] Stage 2 API call in {sel_end - sel_start:.2f}s.")

                print(">>> [INFO] Parsing Stage 2 selection JSON ...")
                sel_json_str = extract_json_from_response(sel_content)
                sel_data = json.loads(sel_json_str)
                selections = _parse_stage2_selections(sel_data, steps_data, len(sampled_frames))
                break
            except Exception as e:
                last_err = e
                if attempt >= stage2_retries:
                    raise
                delay = stage2_base_delay * (2 ** (attempt - 1))
                print(f"!!! [WARNING] Stage 2 failed (attempt {attempt}/{stage2_retries}): {e}")
                print(f">>> [INFO] Retrying Stage 2 in {delay:.1f}s ...")
                time.sleep(delay)

        # Reconstruct directories and save selected keyframe images
        # using original annotations for text, but with selected indices.
        per_step_annotations: Dict[int, List[CriticalFrameAnnotation]] = {}
        for step_json in steps_data:
            step_id = step_json.get('step_id', 0)
            step_goal = step_json.get('step_goal', 'unnamed_step')
            step_folder_name = f"{step_id:02d}_{sanitize_filename(step_goal)}"
            step_output_path = os.path.join(video_output_folder, step_folder_name)
            os.makedirs(step_output_path, exist_ok=True)
            print(f"\n  -> Stage 2 Saving Step {step_id}: '{step_goal}'")

            picked_indices1 = selections.get(int(step_id), [])
            cf_src_list = step_json.get('critical_frames', [])
            critical_frame_annotations: List[CriticalFrameAnnotation] = []
            for i, frame in enumerate(cf_src_list):
                # If selection length mismatches, fallback: skip saving for this frame
                if i >= len(picked_indices1):
                    print(f"    !!! [WARNING] No selected index for critical frame #{i} in step {step_id}. Skipping.")
                    continue
                chosen_idx1 = int(picked_indices1[i])
                # Convert 1-based to 0-based for data access
                chosen_idx0 = chosen_idx1 - 1
                if chosen_idx0 < 0 or chosen_idx0 >= len(sampled_frames):
                    print(f"    !!! [WARNING] Selected 1-based index {chosen_idx1} out of range. Skipping.")
                    continue
                critical_frame_annotations.append(
                    CriticalFrameAnnotation(
                        frame_index=chosen_idx1,
                        causal_chain=_parse_frame_causal_chain(frame.get('causal_chain', {})),
                        interaction=_parse_interaction(frame.get('interaction', {})),
                        action_state_change_description=frame.get('action_state_change_description', '')
                    )
                )

            # Save images for selected frames
            if critical_frame_annotations:
                save_keyframe_images(SELECTION_CONFIG, critical_frame_annotations, step_output_path, sampled_frames)
            per_step_annotations[int(step_id)] = critical_frame_annotations

        print("\n>>> [SUCCESS] Stage 2: Keyframe images saved according to selected indices.")

        # Build and save augmented plan with selected indices
        processed_planning_steps_2: List[PlanningStep] = []
        for step_json in steps_data:
            step_id = step_json.get('step_id', 0)
            failure_reflecting_data = step_json.get('failure_reflecting')
            failure_reflecting_obj = _parse_failure_reflecting(failure_reflecting_data)

            reconstructed = PlanningStep(
                step_id=step_id,
                step_goal=step_json.get('step_goal', ''),
                rationale=step_json.get('rationale', ''),
                causal_chain=_parse_step_causal_chain(step_json.get('causal_chain', {})),
                counterfactual_challenge_question=step_json.get('counterfactual_challenge_question', ''),
                expected_challenge_outcome=step_json.get('expected_challenge_outcome', ''),
                failure_reflecting=failure_reflecting_obj,
                critical_frames=per_step_annotations.get(int(step_id), []),
            )
            processed_planning_steps_2.append(reconstructed)

        final_plan_with_keyframes = {
            "high_level_goal": high_level_goal,
            "steps": [asdict(step) for step in processed_planning_steps_2]
        }
        with open(stage2_path, 'w', encoding='utf-8') as f:
            json.dump(final_plan_with_keyframes, f, indent=4, ensure_ascii=False)
        print(f">>> [SUCCESS] Stage 2 augmented plan saved to: {stage2_path}")

        # Keep sampled frames by default for traceability/debugging.
        # Set CLEANUP_SAMPLED_FRAMES=1 to remove after successful Stage 2 (saves disk space).
        cleanup_sampled_frames = os.environ.get("CLEANUP_SAMPLED_FRAMES", "0").strip().lower() in ("1", "true", "yes", "y")
        if cleanup_sampled_frames:
            try:
                import shutil

                if os.path.isdir(sampled_frames_dir):
                    shutil.rmtree(sampled_frames_dir)
                    print(f">>> [INFO] Removed sampled_frames directory: {sampled_frames_dir}")
            except Exception as e:
                print(f"!!! [WARNING] Unable to remove sampled_frames directory: {e}")
        else:
            print(f">>> [INFO] Keeping sampled_frames directory: {sampled_frames_dir}")

    except Exception as e:
        print(f"\n!!! [FATAL] Error during processing: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n>>> [INFO] Finished processing video: {video_file_path}")

def main():
    """Iterates through all videos in the INPUT_VIDEO_DIRECTORY and processes them sequentially."""
    print(">>> [INFO] Batch Processing Script started.")
    # Allow overriding via environment variables for portability
    input_dir = os.environ.get("INPUT_VIDEO_DIRECTORY", "/e2e-data/embodied-research-data/luzheng/kitchen/long")
    output_base = os.environ.get("OUTPUT_BASE_FOLDER", PLANNING_CONFIG.OUTPUT_BASE_FOLDER)
    PLANNING_CONFIG.OUTPUT_BASE_FOLDER = output_base
    SELECTION_CONFIG.OUTPUT_BASE_FOLDER = output_base
    print(f">>> [INFO] Input Directory: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"!!! [FATAL] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Allowed video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    # Get list of video files
    video_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(video_extensions)]
    
    # Sort files alphabetically to ensure sequential processing "from front to back"
    video_files.sort()

    if not video_files:
        print("!!! [WARNING] No video files found in the directory.")
        sys.exit(0)

    print(f">>> [INFO] Found {len(video_files)} videos to process.")

    for i, filename in enumerate(video_files):
        full_path = os.path.join(input_dir, filename)
        print(f"\n\n##############################################################################")
        print(f"### BATCH PROGRESS: Video {i+1} of {len(video_files)}")
        print(f"### Filename: {filename}")
        print(f"##############################################################################")
        
        try:
            process_single_video(full_path)
        except Exception as e:
            print(f"!!! [ERROR] Unhandled exception while processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            print(">>> [INFO] Continuing to next video...")

    print("\n>>> [INFO] All videos in the folder have been processed.")

if __name__ == "__main__":
    main()
