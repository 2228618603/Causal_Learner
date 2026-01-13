from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


logger = logging.getLogger("three_stage")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Reduce noise from OpenAI-compatible clients
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


DEFAULT_MAX_FRAMES = 50
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class ApiConfig:
    api_key: str = os.environ.get("API_KEY", "EMPTY")
    api_base_url: str = os.environ.get("API_BASE_URL", "http://model.mify.ai.srv/v1")
    model_provider_id: str = os.environ.get("MODEL_PROVIDER_ID", "vertex_ai")
    model_name: str = os.environ.get("MODEL_NAME", "gemini-3-pro-preview")
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "30000"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.2"))
    embed_index_on_api_images: bool = os.environ.get("EMBED_INDEX_ON_API_IMAGES", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    verbose: bool = os.environ.get("VERBOSE_LOGGING", "0").strip().lower() in {"1", "true", "yes", "y"}
    api_call_retries: int = int(os.environ.get("API_CALL_RETRIES", "3"))
    api_call_retry_backoff_sec: float = float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0"))


@dataclass
class SamplingConfig:
    max_frames: int = DEFAULT_MAX_FRAMES
    resize_dimension: Optional[Tuple[int, int]] = None
    jpeg_quality: int = 95


def default_output_root() -> str:
    return os.path.join(os.path.dirname(__file__), "causal_spafa_plan_dataset_long")


def video_id_from_path(video_path: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    return name


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "")).strip().lower()
    text = re.sub(r"[-\s]+", "_", text)
    text = text or "unnamed"
    # Avoid extremely long filenames while keeping readability.
    if len(text) > 80:
        text = text[:80].rstrip("_")
    return text or "unnamed"


_PLACEHOLDER_STRINGS = {
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unspecified",
    "tbd",
    "todo",
    "-",
    "...",
}


def _is_placeholder_str(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = re.sub(r"\s+", " ", s.strip().lower())
    return t in _PLACEHOLDER_STRINGS


_FRAME_REF_RE = re.compile(r"\b(frame|image|img|picture)\s*\d+\b", re.IGNORECASE)
_FRAME_REF_ORDINAL_RE = re.compile(
    r"\b(initial|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|beginning|ending)\s+"
    r"(frame|image|img|picture)\b",
    re.IGNORECASE,
)


def _contains_frame_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_FRAME_REF_RE.search(s) or _FRAME_REF_ORDINAL_RE.search(s))


def _text_dedupe_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _dedupe_keep_order(items: List[str], *, key_fn: Optional[Callable[[str], str]] = None) -> List[str]:
    if not items:
        return []
    if key_fn is None:
        key_fn = lambda x: x  # noqa: E731
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = key_fn(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def initialize_api_client(cfg: ApiConfig) -> Any:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.api_base_url,
            default_headers={"X-Model-Provider-Id": cfg.model_provider_id},
        )
        return client
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI-compatible client: {e}")
        return None


def extract_json_from_response(response_text: str) -> str:
    if not isinstance(response_text, str):
        raise ValueError("Response was not a string.")

    match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
    if match:
        return match.group(1).strip()

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response_text[start : end + 1].strip()

    raise ValueError("Could not find a valid JSON object in the response.")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


def build_retry_prefix(errors: List[str], prev_output: str) -> str:
    err_text = "\n".join(f"- {e}" for e in (errors or [])[:50])
    prev = (prev_output or "")[:12000]
    return (
        "Your previous output was invalid and failed strict validation.\n"
        "Fix ALL errors and return ONLY the corrected strict JSON.\n\n"
        f"Validation errors:\n{err_text}\n\n"
        "Previous output (for reference; correct it):\n"
        f"{prev}\n\n"
    )


def collect_videos(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    paths: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p) and name.lower().endswith(exts):
            paths.append(p)
    return paths


def sample_video_to_frames(
    video_path: str,
    sampling: SamplingConfig,
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required to sample frames.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_dimensions = (width, height)
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError(f"Video has invalid metadata (frames={total_frames}, fps={fps}).")

        if sampling.max_frames <= 0:
            raise ValueError(f"max_frames must be positive (got {sampling.max_frames})")
        if sampling.max_frames == 1:
            indices = [0]
        else:
            # Inclusive uniform sampling over [0, total_frames - 1].
            denom = float(sampling.max_frames - 1)
            indices = [int(round(i * (total_frames - 1) / denom)) for i in range(sampling.max_frames)]
        indices = [min(max(0, idx), total_frames - 1) for idx in indices]
        frames: List[Dict[str, Any]] = []
        last_good: Optional[Dict[str, Any]] = None
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_good is None:
                    continue
                # Keep length stable (50 images) by reusing previous bytes if a read fails.
                # Duplicate the full previous entry (including timestamp/index) to avoid
                # mismatching image bytes with a different timestamp.
                frames.append({**last_good})
                continue

            if sampling.resize_dimension:
                frame = cv2.resize(frame, sampling.resize_dimension)
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(sampling.jpeg_quality)])
            if not ok2:
                if last_good is None:
                    continue
                frames.append({**last_good})
                continue

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            last_good = {"base64": b64, "timestamp_sec": float(frame_idx) / fps, "original_frame_index": int(frame_idx)}
            frames.append({**last_good})

        if len(frames) != sampling.max_frames and frames:
            # Pad to requested size by repeating the last frame.
            while len(frames) < sampling.max_frames:
                frames.append({**frames[-1]})
            frames = frames[: sampling.max_frames]

        if len(frames) != sampling.max_frames:
            raise RuntimeError(f"Failed to sample {sampling.max_frames} frames from {video_path} (got {len(frames)}).")
        return frames, original_dimensions
    finally:
        cap.release()


def save_sampled_frames_jpegs(frames: List[Dict[str, Any]], output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    rel_paths: List[str] = []
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
        path = os.path.join(output_dir, name)
        data = base64.b64decode(fr["base64"]) if isinstance(fr.get("base64"), str) else None
        if not data:
            raise RuntimeError(f"Missing base64 for sampled frame {i}.")
        with open(path, "wb") as f:
            f.write(data)
        rel_paths.append(name)
    return rel_paths


def write_frame_manifest(
    frames: List[Dict[str, Any]],
    sampled_frames_dir: str,
    manifest_path: str,
) -> Dict[str, Any]:
    manifest_dir = os.path.dirname(manifest_path)
    os.makedirs(manifest_dir, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        name = f"sample_{i:03d}_ts_{ts:.2f}s.jpg"
        abs_img = os.path.join(sampled_frames_dir, name)
        rel_img = os.path.relpath(abs_img, manifest_dir)
        entries.append(
            {
                "frame_index_1based": i,
                "timestamp_sec": ts,
                "original_frame_index": int(fr.get("original_frame_index", -1)),
                "image_relpath": rel_img,
            }
        )

    manifest = {
        "num_frames": len(entries),
        "note": "frame_index_1based is the 1-based index used in prompts and model outputs for this frame pool.",
        "frames": entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def load_frames_from_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    manifest = read_json(manifest_path)
    base_dir = os.path.dirname(manifest_path)
    frames: List[Dict[str, Any]] = []
    for entry in manifest.get("frames", []):
        rel = entry.get("image_relpath")
        if not rel:
            continue
        img_path = os.path.join(base_dir, rel)
        with open(img_path, "rb") as f:
            b = f.read()
        frames.append(
            {
                "base64": base64.b64encode(b).decode("utf-8"),
                "timestamp_sec": float(entry.get("timestamp_sec", 0.0)),
                "original_frame_index": int(entry.get("original_frame_index", -1)),
            }
        )
    if len(frames) != int(manifest.get("num_frames", len(frames))):
        logger.warning("Manifest frame count mismatch; continuing with loaded frames.")
    return frames


def build_index_manifest_text(frames: List[Dict[str, Any]]) -> str:
    lines = ["Frame Index Manifest (1-based):"]
    for i, fr in enumerate(frames, start=1):
        ts = float(fr.get("timestamp_sec", 0.0))
        lines.append(f"- Frame {i}: t={ts:.2f}s")
    return "\n".join(lines)


def _overlay_index_on_base64_image(b64_img: str, index_1based: int) -> str:
    if cv2 is None:
        return b64_img
    try:
        import numpy as np

        data = base64.b64decode(b64_img)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return b64_img
        # Only overlay the 1-based index to avoid tempting the model to output timestamps.
        # (Timestamps remain available in the on-disk frame_manifest.json for script-side conversion.)
        text = f"Frame {index_1based:02d}"
        cv2.putText(img, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return b64_img
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return b64_img


def build_api_content(
    frames: List[Dict[str, Any]],
    embed_index: bool,
    *,
    include_manifest: bool = True,
    include_frame_labels: bool = True,
    label_prefix: str = "Frame",
    label_numbers: bool = True,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if include_manifest:
        content.append({"type": "text", "text": build_index_manifest_text(frames)})
    for i, fr in enumerate(frames, start=1):
        b64 = fr.get("base64")
        if embed_index and isinstance(b64, str):
            b64 = _overlay_index_on_base64_image(b64, i)
        if include_frame_labels:
            if label_numbers:
                content.append({"type": "text", "text": f"{label_prefix} {i}"})
            else:
                content.append({"type": "text", "text": str(label_prefix)})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def cut_video_segment_ffmpeg(
    ffmpeg_bin: str,
    src_video: str,
    start_sec: float,
    end_sec: float,
    dst_video: str,
    overwrite: bool,
    *,
    mode: str = "reencode",
    seek_slop_sec: float = 1.0,
    crf: int = 18,
    preset: str = "veryfast",
    keep_audio: bool = False,
) -> None:
    duration = float(end_sec) - float(start_sec)
    if duration <= 0:
        raise ValueError(f"Non-positive clip duration: start={start_sec}, end={end_sec}")

    os.makedirs(os.path.dirname(dst_video), exist_ok=True)

    # For high-quality alignment, default to re-encoding with a hybrid seek:
    # - A fast pre-seek near the target time (start - seek_slop_sec)
    # - An accurate post-seek within the decoded window
    # This avoids the common stream-copy behavior of snapping to previous keyframes.
    mode = (mode or "").strip().lower()
    if mode not in {"copy", "reencode"}:
        raise ValueError(f"Unknown cut mode: {mode} (expected 'copy' or 'reencode')")

    if mode == "copy":
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            src_video,
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            dst_video,
        ]
    else:
        pre = max(0.0, float(start_sec) - float(seek_slop_sec))
        post = float(start_sec) - pre
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{pre:.3f}",
            "-i",
            src_video,
            "-ss",
            f"{post:.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if keep_audio:
            # `0:a?` makes audio optional (some videos have no audio track).
            cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd.append(dst_video)
    cmd.insert(1, "-y" if overwrite else "-n")
    logger.info("[ffmpeg] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"ffmpeg binary not found: '{ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit={e.returncode}). Command: " + " ".join(cmd)) from e


def update_run_summary(path: str, updates: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            data = read_json(path)
        except Exception:
            data = {}
    data.update(updates)
    write_json(path, data)


def normalize_draft_plan(plan: Any) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not isinstance(plan, dict):
        warnings.append(f"Top-level JSON must be an object; got {type(plan).__name__}.")
        plan = {}

    def _norm_str(v: Any) -> str:
        s = str(v).strip() if v is not None else ""
        return "" if _is_placeholder_str(s) else s

    def _norm_identifier(v: Any) -> str:
        s = _norm_str(v)
        if not s:
            return ""
        ident = sanitize_filename(s)
        return "" if ident == "unnamed" else ident

    def _norm_str_list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        out_list: List[str] = []
        for x in v:
            s = _norm_str(x)
            if s:
                out_list.append(s)
        return _dedupe_keep_order(out_list, key_fn=_text_dedupe_key)

    def _norm_identifier_list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        out_list: List[str] = []
        for x in v:
            s = _norm_identifier(x)
            if s:
                out_list.append(s)
        return _dedupe_keep_order(out_list)

    def _parse_bool(v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"true", "t", "yes", "y", "1"}:
                return True
            if t in {"false", "f", "no", "n", "0"}:
                return False
        if isinstance(v, (int, float)) and v in {0, 1}:
            return bool(v)
        return None

    def _norm_spatial_relations(v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            return []
        out_list: List[Dict[str, Any]] = []
        for sp in v:
            if not isinstance(sp, dict):
                continue
            rel = _norm_str(sp.get("relation", ""))
            objs = sp.get("objects")
            if isinstance(objs, str):
                one = _norm_identifier(objs)
                objs_list = [one] if one else []
            else:
                objs_list = _norm_identifier_list(objs) if isinstance(objs, list) else []
            truth = _parse_bool(sp.get("truth"))
            if truth is None:
                truth = True
            if rel and objs_list:
                out_list.append({"relation": rel, "objects": objs_list, "truth": truth})
        return out_list

    def _norm_affordance_states(v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            return []
        out_list: List[Dict[str, Any]] = []
        for ap in v:
            if not isinstance(ap, dict):
                continue
            obj = _norm_identifier(ap.get("object_name", ""))
            affs = ap.get("affordance_types")
            if isinstance(affs, str):
                one = _norm_identifier(affs)
                aff_list = [one] if one else []
            else:
                aff_list = _norm_identifier_list(affs) if isinstance(affs, list) else []
            reasons = _norm_str(ap.get("reasons", ""))
            if obj and aff_list and reasons:
                out_list.append({"object_name": obj, "affordance_types": aff_list, "reasons": reasons})
        return out_list

    out: Dict[str, Any] = {
        "high_level_goal": _norm_str(plan.get("high_level_goal", "")),
        "steps": [],
    }
    if not out["high_level_goal"]:
        warnings.append("Missing/empty high_level_goal.")

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        steps = []
        warnings.append("Top-level 'steps' is not a list; replaced with empty list.")

    seen_goals: Dict[str, int] = {}
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            warnings.append(f"Step #{idx} is not an object; skipped.")
            continue
        if "critical_frames" in step:
            warnings.append(f"Removed unexpected 'critical_frames' in step_id={step.get('step_id')}.")

        tool_usage = step.get("tool_and_material_usage")
        if not isinstance(tool_usage, dict):
            tool_usage = {"tools": [], "materials": []}
        tools = _norm_identifier_list(tool_usage.get("tools"))
        materials = _norm_identifier_list(tool_usage.get("materials"))

        fh = step.get("failure_handling")
        if not isinstance(fh, dict):
            fh = {"reason": "", "recovery_strategy": ""}
        reason = _norm_str(fh.get("reason", ""))
        recovery = _norm_str(fh.get("recovery_strategy", ""))

        step_goal = _norm_str(step.get("step_goal", ""))
        if not step_goal:
            step_goal = f"unnamed_step_{idx:02d}"
            warnings.append(f"Empty step_goal at step #{idx}; replaced with '{step_goal}'.")
        if step_goal in seen_goals:
            warnings.append(f"Duplicate step_goal detected: '{step_goal}' (first at step {seen_goals[step_goal]}).")
        else:
            seen_goals[step_goal] = idx

        spatial_post = _norm_spatial_relations(step.get("spatial_postconditions_detail"))
        afford_post = _norm_affordance_states(step.get("affordance_postconditions_detail"))
        predicted_next = _norm_str_list(step.get("predicted_next_actions"))

        if not spatial_post:
            warnings.append(f"Missing/empty spatial_postconditions_detail at step #{idx}.")
        if not afford_post:
            warnings.append(f"Missing/empty affordance_postconditions_detail at step #{idx}.")
        if not predicted_next:
            warnings.append(f"Missing/empty predicted_next_actions at step #{idx}.")

        out["steps"].append(
            {
                # Enforce canonical step_id to avoid downstream ambiguity.
                "step_id": idx,
                "step_goal": step_goal,
                "rationale": _norm_str(step.get("rationale", "")),
                "preconditions": _norm_str_list(step.get("preconditions")),
                "expected_effects": _norm_str_list(step.get("expected_effects")),
                "spatial_postconditions_detail": spatial_post,
                "affordance_postconditions_detail": afford_post,
                "predicted_next_actions": predicted_next,
                "tool_and_material_usage": {
                    "tools": tools,
                    "materials": materials,
                },
                "causal_challenge_question": _norm_str(step.get("causal_challenge_question", "")),
                "expected_challenge_outcome": _norm_str(step.get("expected_challenge_outcome", "")),
                "failure_handling": {
                    "reason": reason,
                    "recovery_strategy": recovery,
                },
            }
        )

    if not out["steps"]:
        warnings.append("Draft contains 0 usable steps after normalization.")
    return out, warnings


def validate_stage2_localization(
    draft_plan: Dict[str, Any],
    localization: Any,
    num_frames: int,
    *,
    frame_timestamps: Optional[List[float]] = None,
) -> Tuple[bool, List[str], Dict[int, Dict[str, int]]]:
    errors: List[str] = []
    steps = draft_plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return False, ["Draft plan has no steps."], {}

    if not isinstance(localization, dict):
        return False, ["Localization output must be a JSON object with a 'steps' list."], {}
    extra_top = sorted(set(localization.keys()) - {"steps"})
    if extra_top:
        errors.append(f"Localization output contains extra top-level keys (not allowed): {extra_top}")

    loc_steps = localization.get("steps", [])
    if not isinstance(loc_steps, list):
        return False, ["Localization JSON missing 'steps' list."], {}

    step_ids: List[int] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        try:
            step_ids.append(int(s.get("step_id")))
        except Exception:
            continue
    if not step_ids:
        return False, ["Draft plan contains no valid step_id values."], {}
    if len(set(step_ids)) != len(step_ids):
        errors.append("Draft plan has duplicate step_id values (unexpected).")

    expected_ids = set(step_ids)

    allowed_entry_keys = {"step_id", "start_frame_index", "end_frame_index"}
    by_id: Dict[int, Dict[str, int]] = {}
    for i, obj in enumerate(loc_steps):
        if not isinstance(obj, dict):
            errors.append(f"localization.steps[{i}] is not an object.")
            continue
        extra = sorted(set(obj.keys()) - allowed_entry_keys)
        if extra:
            errors.append(f"step_id={obj.get('step_id')} contains extra keys (not allowed): {extra}")
        raw_sid = obj.get("step_id")
        try:
            sid = int(raw_sid)
        except Exception:
            errors.append(f"localization.steps[{i}].step_id missing or non-int.")
            continue
        if sid not in expected_ids:
            errors.append(f"Unexpected step_id in localization output: {sid}")
            continue
        if sid in by_id:
            errors.append(f"Duplicate localization entries for step_id={sid}")
            continue
        try:
            s = int(obj.get("start_frame_index"))
            e = int(obj.get("end_frame_index"))
        except Exception:
            errors.append(f"step_id={sid} start_frame_index/end_frame_index missing or non-int.")
            continue
        by_id[sid] = {"start_frame_index": s, "end_frame_index": e}

    # Ensure all steps present and no extras
    for sid in step_ids:
        if sid not in by_id:
            errors.append(f"Missing localization for step_id={sid}")

    # Validate constraints in step_id order
    ordered = sorted(step_ids)
    prev_end: Optional[int] = None
    prev_end_ts: Optional[float] = None
    for sid in ordered:
        seg = by_id.get(sid)
        if not seg:
            continue
        s = seg["start_frame_index"]
        e = seg["end_frame_index"]
        if s < 1 or s > num_frames:
            errors.append(f"step_id={sid} start_frame_index out of range: {s}")
        if e < 1 or e > num_frames:
            errors.append(f"step_id={sid} end_frame_index out of range: {e}")
        if not (s < e):
            errors.append(f"step_id={sid} requires start_frame_index < end_frame_index (got {s}, {e})")
        if prev_end is not None and s < prev_end:
            errors.append(f"Monotonic constraint violated: step_id={sid} start {s} < prev_end {prev_end}")

        if frame_timestamps and 1 <= s <= len(frame_timestamps) and 1 <= e <= len(frame_timestamps):
            s_ts = float(frame_timestamps[s - 1])
            e_ts = float(frame_timestamps[e - 1])
            if not (s_ts < e_ts):
                if abs(s_ts - e_ts) < 1e-9:
                    errors.append(
                        f"step_id={sid} selected indices map to identical timestamps (got {s_ts:.2f}, {e_ts:.2f}); "
                        "this often happens when sampled frames are duplicates. Choose a larger end_frame_index that shows clear progress."
                    )
                else:
                    errors.append(f"step_id={sid} requires start_sec < end_sec (got {s_ts:.2f}, {e_ts:.2f})")
            if prev_end_ts is not None and s_ts < prev_end_ts:
                errors.append(
                    f"Monotonic constraint violated in seconds: step_id={sid} start_sec {s_ts:.2f} < prev_end_sec {prev_end_ts:.2f}"
                )
            prev_end_ts = e_ts
        prev_end = e

    return len(errors) == 0, errors, by_id


def normalize_stage3_step_output(
    step_json: Dict[str, Any],
    expected_step_id: int,
    expected_step_goal: str,
    num_frames: int,
    *,
    frame_timestamps: Optional[List[float]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(step_json, dict):
        return None, ["Stage 3 output is not an object."]

    allowed_top_keys = {
        "step_id",
        "step_goal",
        "rationale",
        "preconditions",
        "expected_effects",
        "spatial_postconditions_detail",
        "affordance_postconditions_detail",
        "predicted_next_actions",
        "tool_and_material_usage",
        "causal_challenge_question",
        "expected_challenge_outcome",
        "failure_handling",
        "critical_frames",
    }
    extra_top = sorted(set(step_json.keys()) - allowed_top_keys)
    if extra_top:
        errors.append(f"Unexpected top-level keys (not allowed): {extra_top}")

    def _norm_str(v: Any) -> str:
        s = str(v).strip() if v is not None else ""
        return "" if _is_placeholder_str(s) else s

    def _norm_identifier(v: Any) -> str:
        s = _norm_str(v)
        if not s:
            return ""
        ident = sanitize_filename(s)
        return "" if ident == "unnamed" else ident

    def _norm_identifier_list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        out_list: List[str] = []
        for x in v:
            s = _norm_identifier(x)
            if s:
                out_list.append(s)
        return _dedupe_keep_order(out_list)

    def _norm_str_list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        out_list: List[str] = []
        for x in v:
            s = _norm_str(x)
            if s:
                out_list.append(s)
        return _dedupe_keep_order(out_list, key_fn=_text_dedupe_key)

    def _parse_bool(v: Any) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            t = v.strip().lower()
            if t in {"true", "t", "yes", "y", "1"}:
                return True
            if t in {"false", "f", "no", "n", "0"}:
                return False
        if isinstance(v, (int, float)) and v in {0, 1}:
            return bool(v)
        return None

    def _norm_spatial_preconditions(v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            return []
        out_list: List[Dict[str, Any]] = []
        for i, sp in enumerate(v):
            if not isinstance(sp, dict):
                continue
            rel = _norm_str(sp.get("relation", ""))
            objs = sp.get("objects")
            if isinstance(objs, str):
                one = _norm_identifier(objs)
                objs_list = [one] if one else []
            else:
                objs_list = _norm_identifier_list(objs) if isinstance(objs, list) else []
            truth = _parse_bool(sp.get("truth"))
            if truth is None:
                truth = True
            if rel and objs_list:
                out_list.append({"relation": rel, "objects": objs_list, "truth": truth})
        return out_list

    def _norm_affordance_preconditions(v: Any) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            return []
        out_list: List[Dict[str, Any]] = []
        for ap in v:
            if not isinstance(ap, dict):
                continue
            obj = _norm_identifier(ap.get("object_name", ""))
            affs = ap.get("affordance_types")
            if isinstance(affs, str):
                one = _norm_identifier(affs)
                aff_list = [one] if one else []
            else:
                aff_list = _norm_identifier_list(affs) if isinstance(affs, list) else []
            reasons = _norm_str(ap.get("reasons", ""))
            if obj and aff_list:
                out_list.append({"object_name": obj, "affordance_types": aff_list, "reasons": reasons})
        return out_list

    def _norm_causal_chain(v: Any) -> Dict[str, str]:
        d = v if isinstance(v, dict) else {}
        out_cc = {
            "agent": _norm_identifier(d.get("agent", "")),
            "action": _norm_str(d.get("action", "")),
            "patient": _norm_identifier(d.get("patient", "")),
            "causal_effect_on_patient": _norm_str(d.get("causal_effect_on_patient", "")),
            "causal_effect_on_environment": _norm_str(d.get("causal_effect_on_environment", "")),
        }
        return out_cc

    def _norm_affordance_hotspot(v: Any) -> Dict[str, str]:
        d = v if isinstance(v, dict) else {}
        return {
            "description": _norm_str(d.get("description", "")),
            "affordance_type": _norm_identifier(d.get("affordance_type", "")),
            "mechanism": _norm_str(d.get("mechanism", "")),
        }

    sid = step_json.get("step_id")
    try:
        sid_int = int(sid)
    except Exception:
        sid_int = None

    if sid_int != expected_step_id:
        errors.append(f"step_id mismatch: expected {expected_step_id}, got {sid}")

    goal = _norm_str(step_json.get("step_goal", ""))
    if goal != expected_step_goal:
        errors.append(
            "step_goal mismatch (Stage 3 is not allowed to change step_goal): "
            f"expected '{expected_step_goal}', got '{goal}'"
        )

    rationale = _norm_str(step_json.get("rationale", ""))
    preconditions = _norm_str_list(step_json.get("preconditions"))
    expected_effects = _norm_str_list(step_json.get("expected_effects"))
    spatial_post = _norm_spatial_preconditions(step_json.get("spatial_postconditions_detail", []))
    afford_post = _norm_affordance_preconditions(step_json.get("affordance_postconditions_detail", []))
    predicted_next = _norm_str_list(step_json.get("predicted_next_actions"))
    causal_q = _norm_str(step_json.get("causal_challenge_question", ""))
    causal_outcome = _norm_str(step_json.get("expected_challenge_outcome", ""))

    if _contains_frame_ref(rationale):
        errors.append("rationale must not reference frame/image indices.")
    for j, s in enumerate(preconditions):
        if _contains_frame_ref(s):
            errors.append(f"preconditions[{j}] must not reference frame/image indices.")
    for j, s in enumerate(expected_effects):
        if _contains_frame_ref(s):
            errors.append(f"expected_effects[{j}] must not reference frame/image indices.")
    for j, s in enumerate(predicted_next):
        if _contains_frame_ref(s):
            errors.append(f"predicted_next_actions[{j}] must not reference frame/image indices.")
    if _contains_frame_ref(causal_q):
        errors.append("causal_challenge_question must not reference frame/image indices.")
    if _contains_frame_ref(causal_outcome):
        errors.append("expected_challenge_outcome must not reference frame/image indices.")

    for j, sp in enumerate(spatial_post):
        if _contains_frame_ref(sp.get("relation", "")):
            errors.append(f"spatial_postconditions_detail[{j}].relation must not reference frame/image indices.")
        objs = sp.get("objects", [])
        if isinstance(objs, list):
            for o in objs:
                if _contains_frame_ref(o):
                    errors.append(f"spatial_postconditions_detail[{j}].objects must not reference frame/image indices.")
                    break
    for j, ap in enumerate(afford_post):
        if _contains_frame_ref(ap.get("object_name", "")) or _contains_frame_ref(ap.get("reasons", "")):
            errors.append(f"affordance_postconditions_detail[{j}] must not reference frame/image indices.")
        affs = ap.get("affordance_types", [])
        if isinstance(affs, list) and any(_contains_frame_ref(a) for a in affs):
            errors.append(f"affordance_postconditions_detail[{j}].affordance_types must not reference frame/image indices.")

    if not rationale:
        errors.append("Missing/empty rationale.")
    if not preconditions:
        errors.append("Missing/empty preconditions (expected a non-empty list).")
    if not expected_effects:
        errors.append("Missing/empty expected_effects (expected a non-empty list).")
    if not spatial_post:
        errors.append("Missing/empty spatial_postconditions_detail (expected >= 1 concrete relation after completion).")
    if not afford_post:
        errors.append("Missing/empty affordance_postconditions_detail (expected >= 1 grounded post-step affordance/state).")
    for j, ap in enumerate(afford_post):
        if not ap.get("reasons"):
            errors.append(f"affordance_postconditions_detail[{j}].reasons is empty.")
    if not predicted_next:
        errors.append("Missing/empty predicted_next_actions (expected a non-empty list).")
    elif not (2 <= len(predicted_next) <= 4):
        errors.append(f"predicted_next_actions must have length 2-4 (got {len(predicted_next)}).")
    if not causal_q:
        errors.append("Missing/empty causal_challenge_question.")
    if not causal_outcome:
        errors.append("Missing/empty expected_challenge_outcome.")

    raw_spost = step_json.get("spatial_postconditions_detail")
    if isinstance(raw_spost, list):
        for j, sp in enumerate(raw_spost):
            if not isinstance(sp, dict):
                continue
            extra = sorted(set(sp.keys()) - {"relation", "objects", "truth"})
            if extra:
                errors.append(f"spatial_postconditions_detail[{j}] contains extra keys (not allowed): {extra}")
    raw_apost = step_json.get("affordance_postconditions_detail")
    if isinstance(raw_apost, list):
        for j, ap in enumerate(raw_apost):
            if not isinstance(ap, dict):
                continue
            extra = sorted(set(ap.keys()) - {"object_name", "affordance_types", "reasons"})
            if extra:
                errors.append(f"affordance_postconditions_detail[{j}] contains extra keys (not allowed): {extra}")

    cfs = step_json.get("critical_frames")
    if not isinstance(cfs, list):
        errors.append("Missing 'critical_frames' list.")
        cfs = []
    if not (1 <= len(cfs) <= 2):
        errors.append(f"critical_frames must have length 1-2 (got {len(cfs)}).")

    normalized_cfs: List[Dict[str, Any]] = []
    prev_idx = -1
    prev_ts: Optional[float] = None
    for i, cf in enumerate(cfs):
        if not isinstance(cf, dict):
            errors.append(f"critical_frames[{i}] is not an object.")
            continue
        extra_cf = sorted(
            set(cf.keys())
            - {
                "frame_index",
                "action_description",
                "state_change_description",
                "spatial_preconditions",
                "affordance_preconditions",
                "causal_chain",
                "affordance_hotspot",
            }
        )
        if extra_cf:
            errors.append(f"critical_frames[{i}] contains extra keys (not allowed): {extra_cf}")
        try:
            fi = int(cf.get("frame_index"))
        except Exception:
            errors.append(f"critical_frames[{i}].frame_index missing or non-int.")
            continue
        if fi < 1 or fi > num_frames:
            errors.append(f"critical_frames[{i}].frame_index out of range: {fi}")
        if fi <= prev_idx:
            errors.append("critical_frames indices must be strictly increasing within a step.")
        prev_idx = fi
        if frame_timestamps and 1 <= fi <= len(frame_timestamps):
            ts = float(frame_timestamps[fi - 1])
            if prev_ts is not None and not (ts > prev_ts + 1e-9):
                if abs(ts - prev_ts) < 1e-9:
                    errors.append(
                        "Two critical_frames map to identical timestamps; choose a different second frame with real time progress "
                        "(or output only 1 critical_frame if the clip is too short/duplicated)."
                    )
                else:
                    errors.append("critical_frames timestamps must be strictly increasing within a step.")
            prev_ts = ts

        action_desc = _norm_str(cf.get("action_description", ""))
        if not action_desc:
            errors.append(f"critical_frames[{i}].action_description is empty.")
        if _contains_frame_ref(action_desc):
            errors.append(f"critical_frames[{i}].action_description must not reference frame/image indices.")
        state_change = _norm_str(cf.get("state_change_description", ""))
        if not state_change:
            errors.append(f"critical_frames[{i}].state_change_description is empty.")
        if _contains_frame_ref(state_change):
            errors.append(f"critical_frames[{i}].state_change_description must not reference frame/image indices.")

        spatial = _norm_spatial_preconditions(cf.get("spatial_preconditions", []))
        afford = _norm_affordance_preconditions(cf.get("affordance_preconditions", []))
        cc = _norm_causal_chain(cf.get("causal_chain", {}))
        hs = _norm_affordance_hotspot(cf.get("affordance_hotspot", {}))

        if not spatial:
            errors.append(f"critical_frames[{i}].spatial_preconditions is empty (expected >= 1 visually verifiable relation).")
        if not afford:
            errors.append(f"critical_frames[{i}].affordance_preconditions is empty (expected >= 1 grounded affordance state).")
        for j, ap in enumerate(afford):
            if not ap.get("reasons"):
                errors.append(f"critical_frames[{i}].affordance_preconditions[{j}].reasons is empty.")
            if _contains_frame_ref(ap.get("object_name", "")) or _contains_frame_ref(ap.get("reasons", "")):
                errors.append(f"critical_frames[{i}].affordance_preconditions[{j}] must not reference frame/image indices.")
            affs = ap.get("affordance_types", [])
            if isinstance(affs, list) and any(_contains_frame_ref(a) for a in affs):
                errors.append(
                    f"critical_frames[{i}].affordance_preconditions[{j}].affordance_types must not reference frame/image indices."
                )

        for j, sp in enumerate(spatial):
            if _contains_frame_ref(sp.get("relation", "")):
                errors.append(f"critical_frames[{i}].spatial_preconditions[{j}].relation must not reference frame/image indices.")
            objs = sp.get("objects", [])
            if isinstance(objs, list) and any(_contains_frame_ref(o) for o in objs):
                errors.append(f"critical_frames[{i}].spatial_preconditions[{j}].objects must not reference frame/image indices.")

        raw_sp = cf.get("spatial_preconditions")
        if isinstance(raw_sp, list):
            for j, sp in enumerate(raw_sp):
                if not isinstance(sp, dict):
                    continue
                extra_sp = sorted(set(sp.keys()) - {"relation", "objects", "truth"})
                if extra_sp:
                    errors.append(f"critical_frames[{i}].spatial_preconditions[{j}] contains extra keys (not allowed): {extra_sp}")
        raw_ap = cf.get("affordance_preconditions")
        if isinstance(raw_ap, list):
            for j, ap in enumerate(raw_ap):
                if not isinstance(ap, dict):
                    continue
                extra_ap = sorted(set(ap.keys()) - {"object_name", "affordance_types", "reasons"})
                if extra_ap:
                    errors.append(
                        f"critical_frames[{i}].affordance_preconditions[{j}] contains extra keys (not allowed): {extra_ap}"
                    )
        raw_cc = cf.get("causal_chain")
        if isinstance(raw_cc, dict):
            extra_cc = sorted(
                set(raw_cc.keys())
                - {
                    "agent",
                    "action",
                    "patient",
                    "causal_effect_on_patient",
                    "causal_effect_on_environment",
                }
            )
            if extra_cc:
                errors.append(f"critical_frames[{i}].causal_chain contains extra keys (not allowed): {extra_cc}")
        raw_hs = cf.get("affordance_hotspot")
        if isinstance(raw_hs, dict):
            extra_hs = sorted(set(raw_hs.keys()) - {"description", "affordance_type", "mechanism"})
            if extra_hs:
                errors.append(f"critical_frames[{i}].affordance_hotspot contains extra keys (not allowed): {extra_hs}")

        if not cc.get("agent") or not cc.get("action") or not cc.get("patient"):
            errors.append(f"critical_frames[{i}].causal_chain must include non-empty agent/action/patient.")
        if not cc.get("causal_effect_on_patient") or not cc.get("causal_effect_on_environment"):
            errors.append(
                f"critical_frames[{i}].causal_chain must include non-empty causal_effect_on_patient/causal_effect_on_environment."
            )
        if not hs.get("description") or not hs.get("affordance_type") or not hs.get("mechanism"):
            errors.append(
                f"critical_frames[{i}].affordance_hotspot must include non-empty description/affordance_type/mechanism."
            )

        for k, v in cc.items():
            if _contains_frame_ref(v):
                errors.append(f"critical_frames[{i}].causal_chain.{k} must not reference frame/image indices.")
                break
        for k, v in hs.items():
            if _contains_frame_ref(v):
                errors.append(f"critical_frames[{i}].affordance_hotspot.{k} must not reference frame/image indices.")
                break

        normalized_cfs.append(
            {
                "frame_index": fi,
                "action_description": action_desc,
                "state_change_description": state_change,
                "spatial_preconditions": spatial,
                "affordance_preconditions": afford,
                "causal_chain": cc,
                "affordance_hotspot": hs,
                # keyframe_image_path is set by the script after extracting images
                "keyframe_image_path": None,
            }
        )

    tool_usage = step_json.get("tool_and_material_usage")
    if not isinstance(tool_usage, dict):
        tool_usage = {"tools": [], "materials": []}
    else:
        extra_tu = sorted(set(tool_usage.keys()) - {"tools", "materials"})
        if extra_tu:
            errors.append(f"tool_and_material_usage contains extra keys (not allowed): {extra_tu}")
    tools = _norm_identifier_list(tool_usage.get("tools"))
    materials = _norm_identifier_list(tool_usage.get("materials"))
    if not tools and not materials:
        errors.append("tool_and_material_usage must include at least one tool or material (use 'hands' if applicable).")
    for j, s in enumerate(tools):
        if _contains_frame_ref(s):
            errors.append(f"tool_and_material_usage.tools[{j}] must not reference frame/image indices.")
    for j, s in enumerate(materials):
        if _contains_frame_ref(s):
            errors.append(f"tool_and_material_usage.materials[{j}] must not reference frame/image indices.")

    fh = step_json.get("failure_handling")
    if not isinstance(fh, dict):
        fh = {"reason": "", "recovery_strategy": ""}
    else:
        extra_fh = sorted(set(fh.keys()) - {"reason", "recovery_strategy"})
        if extra_fh:
            errors.append(f"failure_handling contains extra keys (not allowed): {extra_fh}")
    reason = _norm_str(fh.get("reason", ""))
    recovery = _norm_str(fh.get("recovery_strategy", ""))
    if not reason:
        errors.append("Missing/empty failure_handling.reason.")
    if not recovery:
        errors.append("Missing/empty failure_handling.recovery_strategy.")
    if _contains_frame_ref(reason):
        errors.append("failure_handling.reason must not reference frame/image indices.")
    if _contains_frame_ref(recovery):
        errors.append("failure_handling.recovery_strategy must not reference frame/image indices.")

    normalized = {
        "step_id": expected_step_id,
        "step_goal": expected_step_goal,
        "rationale": rationale,
        "preconditions": preconditions,
        "expected_effects": expected_effects,
        "spatial_postconditions_detail": spatial_post,
        "affordance_postconditions_detail": afford_post,
        "predicted_next_actions": predicted_next,
        "tool_and_material_usage": {
            "tools": tools,
            "materials": materials,
        },
        "causal_challenge_question": causal_q,
        "expected_challenge_outcome": causal_outcome,
        "failure_handling": {
            "reason": reason,
            "recovery_strategy": recovery,
        },
        "critical_frames": normalized_cfs,
    }

    if errors:
        return None, errors
    return normalized, []


def save_keyframe_images_from_manifest(
    manifest_path: str,
    frame_indices_1based: List[int],
    output_dir: str,
) -> Dict[int, str]:
    manifest = read_json(manifest_path)
    base_dir = os.path.dirname(manifest_path)

    by_idx: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("frames", []):
        if not isinstance(entry, dict):
            continue
        try:
            idx1 = int(entry.get("frame_index_1based"))
        except Exception:
            continue
        by_idx[idx1] = entry

    os.makedirs(output_dir, exist_ok=True)
    out: Dict[int, str] = {}
    for idx1 in frame_indices_1based:
        if idx1 not in by_idx:
            raise ValueError(f"frame_index not found in manifest: {idx1}")
        entry = by_idx[idx1]
        rel = entry.get("image_relpath")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"Manifest entry missing image_relpath for frame_index={idx1}")
        src = os.path.join(base_dir, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        ts = float(entry.get("timestamp_sec", 0.0))
        name = f"frame_{idx1:03d}_ts_{ts:.2f}s.jpg"
        dst = os.path.abspath(os.path.join(output_dir, name))
        shutil.copyfile(src, dst)
        out[idx1] = dst
    return out


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def call_chat_completion(client: Any, cfg: ApiConfig, messages: List[Dict[str, Any]], max_tokens: int) -> str:
    max_attempts = int(getattr(cfg, "api_call_retries", 1) or 1)
    max_attempts = max(1, max_attempts)
    temperature = float(getattr(cfg, "temperature", 0.2) or 0.0)
    backoff_sec = float(getattr(cfg, "api_call_retry_backoff_sec", 1.0) or 0.0)

    start = time.time()
    last_err: Optional[Exception] = None
    resp: Any = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            break
        except Exception as e:
            last_err = e
            if attempt >= max_attempts:
                raise RuntimeError(f"Model call failed after {max_attempts} attempts: {e}") from e
            sleep_sec = min(max(0.0, backoff_sec) * (2 ** (attempt - 1)), 8.0)
            logger.warning(f"Model call error (attempt {attempt}/{max_attempts}): {e}; retrying in {sleep_sec:.1f}s")
            time.sleep(sleep_sec)

    end = time.time()
    if not (resp and getattr(resp, "choices", None) and len(resp.choices) > 0):
        raise RuntimeError("Model response missing choices.")
    choice0 = resp.choices[0]
    if not hasattr(choice0, "message") or not hasattr(choice0.message, "content"):
        raise RuntimeError("Model response missing message.content.")
    content = choice0.message.content or ""
    logger.info(f"Model call finished in {end - start:.2f}s")
    if cfg.verbose:
        logger.info("Raw model output:\n" + content)
    return content
