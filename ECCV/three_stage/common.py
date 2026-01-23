from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
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


def add_api_cli_args(parser: Any, *, include_no_embed_index: bool = True) -> None:
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"), help="API key (or set env: API_KEY).")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE_URL", "http://model.mify.ai.srv/v1"),
        help="OpenAI-compatible base URL (or set env: API_BASE_URL).",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("MODEL_PROVIDER_ID", "vertex_ai"),
        help="Model provider id (sent as header X-Model-Provider-Id; or set env: MODEL_PROVIDER_ID).",
    )
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemini-3-pro-preview"), help="Model name.")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "30000")), help="Max tokens.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.2")),
        help="Sampling temperature (lower is usually more stable).",
    )
    parser.add_argument(
        "--api-call-retries",
        type=int,
        default=int(os.environ.get("API_CALL_RETRIES", "3")),
        help="Retries for a single model call (transport/service errors).",
    )
    parser.add_argument(
        "--api-call-retry-backoff-sec",
        type=float,
        default=float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0")),
        help="Backoff seconds for model-call retries (exponential).",
    )
    if include_no_embed_index:
        parser.add_argument(
            "--no-embed-index",
            action="store_true",
            help="Do not draw 1-based frame indices onto images (will use text labels instead).",
        )
    parser.add_argument("--verbose", action="store_true", help="Log raw model outputs.")


def api_config_from_args(args: Any) -> ApiConfig:
    return ApiConfig(
        api_key=getattr(args, "api_key"),
        api_base_url=getattr(args, "api_base"),
        model_provider_id=getattr(args, "provider"),
        model_name=getattr(args, "model"),
        max_tokens=int(getattr(args, "max_tokens")),
        temperature=float(getattr(args, "temperature")),
        api_call_retries=int(getattr(args, "api_call_retries")),
        api_call_retry_backoff_sec=float(getattr(args, "api_call_retry_backoff_sec")),
        embed_index_on_api_images=not bool(getattr(args, "no_embed_index", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


@dataclass
class SamplingConfig:
    max_frames: int = DEFAULT_MAX_FRAMES
    resize_dimension: Optional[Tuple[int, int]] = None
    jpeg_quality: int = 95


def add_sampling_cli_args(parser: Any, *, default_max_frames: int = 50, default_jpeg_quality: int = 95) -> None:
    parser.add_argument("--max-frames", type=int, default=int(default_max_frames), help="Max frames sampled per pool.")
    parser.add_argument("--jpeg-quality", type=int, default=int(default_jpeg_quality), help="JPEG quality (1-100).")


def sampling_config_from_args(args: Any) -> SamplingConfig:
    return SamplingConfig(max_frames=int(getattr(args, "max_frames")), jpeg_quality=int(getattr(args, "jpeg_quality")))


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


_TIME_REF_RE = re.compile(
    r"(?:"
    # Explicit timestamp assignment like "t=3.2s"
    r"\bt\s*=\s*\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"
    # Numeric duration like "3.2s", "3 seconds"
    r"\b\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|ms|msec|milliseconds?)\b"
    r"|"
    # Timecode like "00:03" or "1:02:03.5"
    r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b"
    r")",
    re.IGNORECASE,
)


def _contains_time_ref(text: Any) -> bool:
    s = str(text or "")
    return bool(_TIME_REF_RE.search(s))


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
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_path or ".",
            prefix=os.path.basename(path) + ".tmp.",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def write_text(path: str, text: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_path or ".",
            prefix=os.path.basename(path) + ".tmp.",
            delete=False,
        ) as f:
            tmp_path = f.name
            f.write(text or "")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class OutputDirCollisionError(RuntimeError):
    """Raised when the output directory is unsafe to use for the current source video."""


def ensure_video_out_dir_safe(video_out: str, video_path: str) -> None:
    """Fail fast if `video_out` already contains outputs for a different source video.

    The pipeline uses `video_id_from_path()` (stem of filename) to name `<video_out>`. If two different
    videos share the same stem, resuming would silently mix artifacts and corrupt outputs.
    """
    if not os.path.exists(video_out):
        return
    try:
        entries = [x for x in os.listdir(video_out) if x not in {".", ".."}]
    except Exception:
        return
    if not entries:
        return

    run_summary_path = os.path.join(video_out, "run_summary.json")
    if os.path.exists(run_summary_path):
        try:
            rs = read_json(run_summary_path)
        except Exception:
            rs = {}
        src = rs.get("source_video")
        if isinstance(src, str) and src.strip():
            cur = os.path.abspath(video_path)
            old = os.path.abspath(src)
            if cur != old:
                raise OutputDirCollisionError(
                    "Output dir collision detected.\n"
                    f"- video_out: {os.path.abspath(video_out)}\n"
                    f"- existing source_video: {old}\n"
                    f"- current  source_video: {cur}\n"
                    "Rename the video file (or use a different --output-root), or delete the existing output folder."
                )
            return

    raise OutputDirCollisionError(
        f"Output dir is non-empty but missing a readable `run_summary.json` with `source_video`: {os.path.abspath(video_out)}. "
        "To avoid mixing outputs across videos, delete this folder (or pick a different --output-root) and re-run."
    )


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

    def _norm_identifier_list(v: Any) -> List[str]:
        if isinstance(v, str):
            one = _norm_identifier(v)
            return [one] if one else []
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
            # Normalize relation token to a short snake_case identifier (as required by prompts.py).
            rel = _norm_identifier(sp.get("relation", ""))
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
        return _dedupe_keep_order(
            out_list,
            key_fn=lambda d: (d.get("relation"), tuple(d.get("objects") or []), bool(d.get("truth"))),
        )

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
        return _dedupe_keep_order(
            out_list,
            key_fn=lambda d: (d.get("object_name"), tuple(d.get("affordance_types") or []), d.get("reasons")),
        )

    def _norm_causal_chain(v: Any) -> Dict[str, Any]:
        d = v if isinstance(v, dict) else {}
        return {
            "agent": _norm_identifier(d.get("agent", "")),
            "action": _norm_str(d.get("action", "")),
            "patient": _norm_identifier(d.get("patient", "")),
            "causal_precondition_on_spatial": _norm_spatial_relations(d.get("causal_precondition_on_spatial")),
            "causal_precondition_on_affordance": _norm_affordance_states(d.get("causal_precondition_on_affordance")),
            "causal_effect_on_spatial": _norm_spatial_relations(d.get("causal_effect_on_spatial")),
            "causal_effect_on_affordance": _norm_affordance_states(d.get("causal_effect_on_affordance")),
        }

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

    # If the model provided a coherent step_id sequence (1..N), prefer that ordering to preserve chronology.
    non_object_steps = sum(1 for s in steps if not isinstance(s, dict))
    if non_object_steps:
        warnings.append(f"Found {non_object_steps} non-object step entries; they will be skipped.")

    dict_steps = [s for s in steps if isinstance(s, dict)]
    parsed_step_ids: List[int] = []
    step_ids_ok = bool(dict_steps)
    for s in dict_steps:
        try:
            sid = int(s.get("step_id"))
        except Exception:
            step_ids_ok = False
            break
        if sid <= 0:
            step_ids_ok = False
            break
        parsed_step_ids.append(sid)
    if (
        step_ids_ok
        and len(set(parsed_step_ids)) == len(parsed_step_ids)
        and set(parsed_step_ids) == set(range(1, len(parsed_step_ids) + 1))
    ):
        if parsed_step_ids != sorted(parsed_step_ids):
            warnings.append("Reordered steps by provided step_id (1..N) to preserve chronology.")
        steps = sorted(dict_steps, key=lambda x: int(x.get("step_id")))

    seen_goals: Dict[str, int] = {}
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            warnings.append(f"Step #{idx} is not an object; skipped.")
            continue
        for forbidden in ("critical_frames", "frame_index", "interaction", "keyframe_image_path"):
            if forbidden in step:
                warnings.append(f"Removed unexpected '{forbidden}' in step_id={step.get('step_id')}.")
                step.pop(forbidden, None)

        step_goal = _norm_str(step.get("step_goal", ""))
        if not step_goal:
            step_goal = f"unnamed_step_{idx:02d}"
            warnings.append(f"Empty step_goal at step #{idx}; replaced with '{step_goal}'.")
        if step_goal in seen_goals:
            warnings.append(f"Duplicate step_goal detected: '{step_goal}' (first at step {seen_goals[step_goal]}).")
        else:
            seen_goals[step_goal] = idx

        cc = _norm_causal_chain(step.get("causal_chain"))
        if not cc.get("agent") or not cc.get("action") or not cc.get("patient"):
            warnings.append(f"Missing/empty causal_chain agent/action/patient at step #{idx}.")
        if not cc.get("causal_precondition_on_spatial"):
            warnings.append(f"Missing/empty causal_chain.causal_precondition_on_spatial at step #{idx}.")
        if not cc.get("causal_precondition_on_affordance"):
            warnings.append(f"Missing/empty causal_chain.causal_precondition_on_affordance at step #{idx}.")
        if not cc.get("causal_effect_on_spatial"):
            warnings.append(f"Missing/empty causal_chain.causal_effect_on_spatial at step #{idx}.")
        if not cc.get("causal_effect_on_affordance"):
            warnings.append(f"Missing/empty causal_chain.causal_effect_on_affordance at step #{idx}.")

        counterfactual_q = _norm_str(step.get("counterfactual_challenge_question", ""))
        if not counterfactual_q:
            warnings.append(f"Missing/empty counterfactual_challenge_question at step #{idx}.")

        expected_outcome = _norm_str(step.get("expected_challenge_outcome", ""))
        if not expected_outcome:
            warnings.append(f"Missing/empty expected_challenge_outcome at step #{idx}.")

        fr = step.get("failure_reflecting")
        if not isinstance(fr, dict):
            fr = {"reason": "", "recovery_strategy": ""}
        fr_reason = _norm_str(fr.get("reason", ""))
        fr_recovery = _norm_str(fr.get("recovery_strategy", ""))
        if not fr_reason:
            warnings.append(f"Missing/empty failure_reflecting.reason at step #{idx}.")
        if not fr_recovery:
            warnings.append(f"Missing/empty failure_reflecting.recovery_strategy at step #{idx}.")

        out["steps"].append(
            {
                # Enforce canonical step_id to avoid downstream ambiguity.
                "step_id": idx,
                "step_goal": step_goal,
                "rationale": _norm_str(step.get("rationale", "")),
                "causal_chain": cc,
                "counterfactual_challenge_question": counterfactual_q,
                "expected_challenge_outcome": expected_outcome,
                "failure_reflecting": {"reason": fr_reason, "recovery_strategy": fr_recovery},
            }
        )

    if not out["steps"]:
        warnings.append("Draft contains 0 usable steps after normalization.")
    return out, warnings


def estimate_min_positive_delta_sec(timestamps: List[float]) -> float:
    uniq = sorted({float(x) for x in (timestamps or [])})
    deltas = [b - a for a, b in zip(uniq, uniq[1:]) if b > a]
    if not deltas:
        return 0.1
    # Avoid epsilon too close to 0 which may produce empty clips after seeking/rounding.
    return max(min(deltas), 0.05)


def validate_stage2_localization(
    draft_plan: Dict[str, Any],
    localization: Any,
    num_frames: int,
    *,
    frame_timestamps: Optional[List[float]] = None,
) -> Tuple[bool, List[str], Dict[int, Dict[str, int]]]:
    errors: List[str] = []

    def _is_strict_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

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
        if not _is_strict_int(raw_sid):
            errors.append(f"localization.steps[{i}].step_id missing/invalid (expected int).")
            continue
        sid = int(raw_sid)
        if sid not in expected_ids:
            errors.append(f"Unexpected step_id in localization output: {sid}")
            continue
        if sid in by_id:
            errors.append(f"Duplicate localization entries for step_id={sid}")
            continue
        raw_s = obj.get("start_frame_index")
        raw_e = obj.get("end_frame_index")
        if not _is_strict_int(raw_s) or not _is_strict_int(raw_e):
            errors.append(f"step_id={sid} start_frame_index/end_frame_index missing/invalid (expected int).")
            continue
        s = int(raw_s)
        e = int(raw_e)
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
        if e < 1 or e > num_frames + 1:
            errors.append(f"step_id={sid} end_frame_index out of range: {e}")
        if not (s < e):
            errors.append(f"step_id={sid} requires start_frame_index < end_frame_index (got {s}, {e})")
        if prev_end is not None and s < prev_end:
            errors.append(f"Monotonic constraint violated: step_id={sid} start {s} < prev_end {prev_end}")

        if frame_timestamps and 1 <= s <= len(frame_timestamps) and 1 <= e <= len(frame_timestamps) + 1:
            s_ts = float(frame_timestamps[s - 1])
            if e == len(frame_timestamps) + 1:
                e_ts = float(frame_timestamps[-1]) + estimate_min_positive_delta_sec(frame_timestamps)
            else:
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

    def _is_strict_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    allowed_top_keys = {
        "step_id",
        "step_goal",
        "rationale",
        "causal_chain",
        "counterfactual_challenge_question",
        "expected_challenge_outcome",
        "failure_reflecting",
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

    def _canon_text(v: Any) -> str:
        return re.sub(r"\s+", " ", str(v or "").strip())

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

    allowed_cc_keys = {
        "agent",
        "action",
        "patient",
        "causal_precondition_on_spatial",
        "causal_precondition_on_affordance",
        "causal_effect_on_spatial",
        "causal_effect_on_affordance",
    }
    allowed_rel_keys = {"relation", "objects", "truth"}
    allowed_aff_keys = {"object_name", "affordance_types", "reasons"}

    def _norm_spatial_relations(v: Any, *, label: str) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            return []
        out_list: List[Dict[str, Any]] = []
        for j, sp in enumerate(v):
            if not isinstance(sp, dict):
                continue
            extra_sp = sorted(set(sp.keys()) - allowed_rel_keys)
            if extra_sp:
                errors.append(f"{label}[{j}] contains extra keys (not allowed): {extra_sp}")
            rel_raw = sp.get("relation", "")
            if _contains_frame_ref(rel_raw):
                errors.append(f"{label}[{j}].relation must not reference frame/image indices.")
            if _contains_time_ref(rel_raw):
                errors.append(f"{label}[{j}].relation must not reference timestamps/durations.")
            if isinstance(rel_raw, str) and ("=>" in rel_raw or "(" in rel_raw or ")" in rel_raw):
                errors.append(
                    f"{label}[{j}].relation must be a short token (no '=>', no parentheses); got: {rel_raw!r}"
                )
            # Normalize to snake_case token.
            rel = _norm_identifier(rel_raw)
            if rel and len(rel) > 40:
                errors.append(f"{label}[{j}].relation is too long (>40 chars); got: {rel!r}")
            objs_raw = sp.get("objects")
            if isinstance(objs_raw, str):
                if _contains_frame_ref(objs_raw):
                    errors.append(f"{label}[{j}].objects must not reference frame/image indices.")
                if _contains_time_ref(objs_raw):
                    errors.append(f"{label}[{j}].objects must not reference timestamps/durations.")
            elif isinstance(objs_raw, list) and any(_contains_frame_ref(x) for x in objs_raw):
                errors.append(f"{label}[{j}].objects must not reference frame/image indices.")
            elif isinstance(objs_raw, list) and any(_contains_time_ref(x) for x in objs_raw):
                errors.append(f"{label}[{j}].objects must not reference timestamps/durations.")
            objs_list = _norm_identifier_list(objs_raw)
            raw_truth = sp.get("truth")
            if not isinstance(raw_truth, bool):
                errors.append(f"{label}[{j}].truth must be a JSON boolean (true/false).")
            truth = _parse_bool(raw_truth)
            if truth is None:
                errors.append(f"{label}[{j}].truth missing/invalid (expected boolean true/false).")
                truth = True
            if not rel:
                errors.append(f"{label}[{j}].relation is empty (expected a non-empty string).")
            if not objs_list:
                errors.append(f"{label}[{j}].objects is empty (expected a non-empty list of strings).")
            if rel and objs_list:
                out_list.append({"relation": rel, "objects": objs_list, "truth": truth})
        return _dedupe_keep_order(
            out_list,
            key_fn=lambda d: (d.get("relation"), tuple(d.get("objects") or []), bool(d.get("truth"))),
        )

    def _norm_affordance_states(v: Any, *, label: str) -> List[Dict[str, Any]]:
        if not isinstance(v, list):
            return []
        out_list: List[Dict[str, Any]] = []
        for j, ap in enumerate(v):
            if not isinstance(ap, dict):
                continue
            extra_ap = sorted(set(ap.keys()) - allowed_aff_keys)
            if extra_ap:
                errors.append(f"{label}[{j}] contains extra keys (not allowed): {extra_ap}")
            obj_raw = ap.get("object_name", "")
            if _contains_frame_ref(obj_raw):
                errors.append(f"{label}[{j}].object_name must not reference frame/image indices.")
            if _contains_time_ref(obj_raw):
                errors.append(f"{label}[{j}].object_name must not reference timestamps/durations.")
            obj = _norm_identifier(obj_raw)
            affs_raw = ap.get("affordance_types")
            if isinstance(affs_raw, str):
                if _contains_frame_ref(affs_raw):
                    errors.append(f"{label}[{j}].affordance_types must not reference frame/image indices.")
                if _contains_time_ref(affs_raw):
                    errors.append(f"{label}[{j}].affordance_types must not reference timestamps/durations.")
            elif isinstance(affs_raw, list) and any(_contains_frame_ref(x) for x in affs_raw):
                errors.append(f"{label}[{j}].affordance_types must not reference frame/image indices.")
            elif isinstance(affs_raw, list) and any(_contains_time_ref(x) for x in affs_raw):
                errors.append(f"{label}[{j}].affordance_types must not reference timestamps/durations.")
            aff_list = _norm_identifier_list(affs_raw)
            reasons_raw = ap.get("reasons", "")
            if _contains_frame_ref(reasons_raw):
                errors.append(f"{label}[{j}].reasons must not reference frame/image indices.")
            if _contains_time_ref(reasons_raw):
                errors.append(f"{label}[{j}].reasons must not reference timestamps/durations.")
            reasons = _norm_str(reasons_raw)
            if not obj:
                errors.append(f"{label}[{j}].object_name is empty (expected a non-empty string).")
            if not aff_list:
                errors.append(f"{label}[{j}].affordance_types is empty (expected a non-empty list).")
            if not reasons:
                errors.append(f"{label}[{j}].reasons is empty (expected a non-empty grounded string).")
            if obj and aff_list and reasons:
                out_list.append({"object_name": obj, "affordance_types": aff_list, "reasons": reasons})
        return _dedupe_keep_order(
            out_list,
            key_fn=lambda d: (d.get("object_name"), tuple(d.get("affordance_types") or []), d.get("reasons")),
        )

    def _norm_causal_chain(v: Any, *, label: str) -> Dict[str, Any]:
        d = v if isinstance(v, dict) else {}
        if isinstance(v, dict):
            extra_cc = sorted(set(d.keys()) - allowed_cc_keys)
            if extra_cc:
                errors.append(f"{label}.causal_chain contains extra keys (not allowed): {extra_cc}")
        if _contains_frame_ref(d.get("agent", "")) or _contains_frame_ref(d.get("action", "")) or _contains_frame_ref(
            d.get("patient", "")
        ):
            errors.append(f"{label}.causal_chain agent/action/patient must not reference frame/image indices.")
        if _contains_time_ref(d.get("agent", "")) or _contains_time_ref(d.get("action", "")) or _contains_time_ref(
            d.get("patient", "")
        ):
            errors.append(f"{label}.causal_chain agent/action/patient must not reference timestamps/durations.")
        cc = {
            "agent": _norm_identifier(d.get("agent", "")),
            "action": _norm_str(d.get("action", "")),
            "patient": _norm_identifier(d.get("patient", "")),
            "causal_precondition_on_spatial": _norm_spatial_relations(
                d.get("causal_precondition_on_spatial"), label=f"{label}.causal_chain.causal_precondition_on_spatial"
            ),
            "causal_precondition_on_affordance": _norm_affordance_states(
                d.get("causal_precondition_on_affordance"),
                label=f"{label}.causal_chain.causal_precondition_on_affordance",
            ),
            "causal_effect_on_spatial": _norm_spatial_relations(
                d.get("causal_effect_on_spatial"), label=f"{label}.causal_chain.causal_effect_on_spatial"
            ),
            "causal_effect_on_affordance": _norm_affordance_states(
                d.get("causal_effect_on_affordance"), label=f"{label}.causal_chain.causal_effect_on_affordance"
            ),
        }
        if not cc["agent"] or not cc["action"] or not cc["patient"]:
            errors.append(f"{label}.causal_chain must include non-empty agent/action/patient.")
        if not cc["causal_precondition_on_spatial"]:
            errors.append(f"{label}.causal_chain.causal_precondition_on_spatial is empty (expected a non-empty list).")
        if not cc["causal_precondition_on_affordance"]:
            errors.append(f"{label}.causal_chain.causal_precondition_on_affordance is empty (expected a non-empty list).")
        if not cc["causal_effect_on_spatial"]:
            errors.append(f"{label}.causal_chain.causal_effect_on_spatial is empty (expected a non-empty list).")
        if not cc["causal_effect_on_affordance"]:
            errors.append(f"{label}.causal_chain.causal_effect_on_affordance is empty (expected a non-empty list).")
        return cc

    sid = step_json.get("step_id")
    sid_int: Optional[int] = int(sid) if _is_strict_int(sid) else None
    if sid_int is None:
        errors.append("step_id missing/invalid (expected int).")
    if sid_int != expected_step_id:
        errors.append(f"step_id mismatch: expected {expected_step_id}, got {sid}")

    goal = _norm_str(step_json.get("step_goal", ""))
    if goal != expected_step_goal:
        errors.append(
            "step_goal mismatch (Stage 3 is not allowed to change step_goal): "
            f"expected '{expected_step_goal}', got '{goal}'"
        )

    rationale = _norm_str(step_json.get("rationale", ""))
    if not rationale:
        errors.append("Missing/empty rationale.")
    if _contains_frame_ref(rationale):
        errors.append("rationale must not reference frame/image indices.")
    if _contains_time_ref(rationale):
        errors.append("rationale must not reference timestamps/durations.")

    counterfactual_q = _norm_str(step_json.get("counterfactual_challenge_question", ""))
    if not counterfactual_q:
        errors.append("Missing/empty counterfactual_challenge_question.")
    if _contains_frame_ref(counterfactual_q):
        errors.append("counterfactual_challenge_question must not reference frame/image indices.")
    if _contains_time_ref(counterfactual_q):
        errors.append("counterfactual_challenge_question must not reference timestamps/durations.")

    expected_outcome = _norm_str(step_json.get("expected_challenge_outcome", ""))
    if not expected_outcome:
        errors.append("Missing/empty expected_challenge_outcome.")
    if _contains_frame_ref(expected_outcome):
        errors.append("expected_challenge_outcome must not reference frame/image indices.")
    if _contains_time_ref(expected_outcome):
        errors.append("expected_challenge_outcome must not reference timestamps/durations.")

    fr_raw = step_json.get("failure_reflecting")
    if not isinstance(fr_raw, dict):
        fr_raw = {}
    extra_fr = sorted(set(fr_raw.keys()) - {"reason", "recovery_strategy"})
    if extra_fr:
        errors.append(f"failure_reflecting contains extra keys (not allowed): {extra_fr}")
    fr_reason = _norm_str(fr_raw.get("reason", ""))
    fr_recovery = _norm_str(fr_raw.get("recovery_strategy", ""))
    if not fr_reason:
        errors.append("Missing/empty failure_reflecting.reason.")
    if not fr_recovery:
        errors.append("Missing/empty failure_reflecting.recovery_strategy.")
    if _contains_frame_ref(fr_reason) or _contains_frame_ref(fr_recovery):
        errors.append("failure_reflecting fields must not reference frame/image indices.")
    if _contains_time_ref(fr_reason) or _contains_time_ref(fr_recovery):
        errors.append("failure_reflecting fields must not reference timestamps/durations.")

    step_cc = _norm_causal_chain(step_json.get("causal_chain", {}), label="step")

    cfs = step_json.get("critical_frames")
    if not isinstance(cfs, list):
        errors.append("Missing 'critical_frames' list.")
        cfs = []
    if len(cfs) != 2:
        errors.append(f"critical_frames must have length exactly 2 (got {len(cfs)}).")

    normalized_cfs: List[Dict[str, Any]] = []
    prev_idx = -1
    prev_ts: Optional[float] = None
    for i, cf in enumerate(cfs):
        if not isinstance(cf, dict):
            errors.append(f"critical_frames[{i}] is not an object.")
            continue

        extra_cf = sorted(set(cf.keys()) - {"frame_index", "action_state_change_description", "causal_chain", "interaction"})
        if extra_cf:
            errors.append(f"critical_frames[{i}] contains extra keys (not allowed): {extra_cf}")

        raw_fi = cf.get("frame_index")
        if not _is_strict_int(raw_fi):
            errors.append(f"critical_frames[{i}].frame_index missing/invalid (expected int).")
            continue
        fi = int(raw_fi)
        if fi < 1 or fi > int(num_frames):
            errors.append(f"critical_frames[{i}].frame_index out of range: {fi}")
        if fi <= prev_idx:
            errors.append("critical_frames indices must be strictly increasing within a step.")
        prev_idx = fi

        if frame_timestamps and 1 <= fi <= len(frame_timestamps):
            ts = float(frame_timestamps[fi - 1])
            if prev_ts is not None and not (ts > prev_ts + 1e-9):
                if abs(ts - prev_ts) < 1e-9:
                    errors.append(
                        "Two critical_frames map to identical timestamps; choose a different second frame with real time progress."
                    )
                else:
                    errors.append("critical_frames timestamps must be strictly increasing within a step.")
            prev_ts = ts

        desc = _norm_str(cf.get("action_state_change_description", ""))
        if not desc:
            errors.append(f"critical_frames[{i}].action_state_change_description is empty.")
        if _contains_frame_ref(desc):
            errors.append(f"critical_frames[{i}].action_state_change_description must not reference frame/image indices.")
        if _contains_time_ref(desc):
            errors.append(f"critical_frames[{i}].action_state_change_description must not reference timestamps/durations.")

        cf_cc = _norm_causal_chain(cf.get("causal_chain", {}), label=f"critical_frames[{i}]")
        # Enforce that the keyframe-level causal_chain agrees with the step-level one.
        if cf_cc.get("agent") != step_cc.get("agent"):
            errors.append(
                f"critical_frames[{i}].causal_chain.agent must match step.causal_chain.agent "
                f"(expected {step_cc.get('agent')!r}, got {cf_cc.get('agent')!r})."
            )
        if _canon_text(cf_cc.get("action")) != _canon_text(step_cc.get("action")):
            errors.append(
                f"critical_frames[{i}].causal_chain.action must match step.causal_chain.action "
                f"(expected {step_cc.get('action')!r}, got {cf_cc.get('action')!r})."
            )
        if cf_cc.get("patient") != step_cc.get("patient"):
            errors.append(
                f"critical_frames[{i}].causal_chain.patient must match step.causal_chain.patient "
                f"(expected {step_cc.get('patient')!r}, got {cf_cc.get('patient')!r})."
            )

        interaction = cf.get("interaction")
        if not isinstance(interaction, dict):
            errors.append(f"critical_frames[{i}].interaction missing/invalid (expected an object).")
            interaction = {}
        extra_inter = sorted(set(interaction.keys()) - {"tools", "materials", "hotspot"})
        if extra_inter:
            errors.append(f"critical_frames[{i}].interaction contains extra keys (not allowed): {extra_inter}")
        raw_tools = interaction.get("tools")
        raw_materials = interaction.get("materials")

        if not isinstance(raw_tools, list):
            errors.append(f"critical_frames[{i}].interaction.tools missing/invalid (expected a list[string]).")
            raw_tools = []
        if not isinstance(raw_materials, list):
            errors.append(f"critical_frames[{i}].interaction.materials missing/invalid (expected a list[string]).")
            raw_materials = []

        if any(_contains_frame_ref(x) for x in raw_tools) or any(_contains_frame_ref(x) for x in raw_materials):
            errors.append(f"critical_frames[{i}].interaction.tools/materials must not reference frame/image indices.")
        if any(_contains_time_ref(x) for x in raw_tools) or any(_contains_time_ref(x) for x in raw_materials):
            errors.append(f"critical_frames[{i}].interaction.tools/materials must not reference timestamps/durations.")

        if any(not (isinstance(x, str) and x.strip()) for x in raw_tools):
            errors.append(f"critical_frames[{i}].interaction.tools must contain only non-empty strings.")
        if any(not (isinstance(x, str) and x.strip()) for x in raw_materials):
            errors.append(f"critical_frames[{i}].interaction.materials must contain only non-empty strings.")

        tools = _norm_identifier_list(raw_tools)
        materials = _norm_identifier_list(raw_materials)

        if not tools:
            errors.append(f"critical_frames[{i}].interaction.tools is empty (expected a non-empty list[string]).")
        if not materials:
            errors.append(f"critical_frames[{i}].interaction.materials is empty (expected a non-empty list[string]).")

        # Quality: keep interaction aligned with causal_chain identifiers.
        agent = str(step_cc.get("agent", "") or "")
        if agent and agent not in tools:
            errors.append(
                f"critical_frames[{i}].interaction.tools must include the causal_chain.agent "
                f"(expected agent={agent!r}, tools={tools})."
            )
        patient = str(step_cc.get("patient", "") or "")
        if patient and patient not in materials:
            errors.append(
                f"critical_frames[{i}].interaction.materials must include the causal_chain.patient "
                f"(expected patient={patient!r}, materials={materials})."
            )

        hotspot = interaction.get("hotspot")
        if not isinstance(hotspot, dict):
            errors.append(f"critical_frames[{i}].interaction.hotspot missing/invalid (expected an object).")
            hotspot = {}
        extra_hs = sorted(set(hotspot.keys()) - {"description", "affordance_type", "mechanism"})
        if extra_hs:
            errors.append(f"critical_frames[{i}].interaction.hotspot contains extra keys (not allowed): {extra_hs}")
        hs_desc_raw = hotspot.get("description", "")
        hs_type_raw = hotspot.get("affordance_type", "")
        hs_mech_raw = hotspot.get("mechanism", "")
        hs_desc = _norm_str(hs_desc_raw)
        hs_type = _norm_identifier(hs_type_raw)
        hs_mech = _norm_str(hs_mech_raw)
        if _contains_frame_ref(hs_desc_raw) or _contains_frame_ref(hs_type_raw) or _contains_frame_ref(hs_mech_raw):
            errors.append(f"critical_frames[{i}].interaction.hotspot must not reference frame/image indices.")
        if _contains_time_ref(hs_desc_raw) or _contains_time_ref(hs_type_raw) or _contains_time_ref(hs_mech_raw):
            errors.append(f"critical_frames[{i}].interaction.hotspot must not reference timestamps/durations.")
        if not hs_desc or not hs_type or not hs_mech:
            errors.append(
                f"critical_frames[{i}].interaction.hotspot must include non-empty description/affordance_type/mechanism."
            )

        normalized_cfs.append(
            {
                "frame_index": fi,
                "action_state_change_description": desc,
                "causal_chain": cf_cc,
                "interaction": {
                    "tools": tools,
                    "materials": materials,
                    "hotspot": {"description": hs_desc, "affordance_type": hs_type, "mechanism": hs_mech},
                },
            }
        )

    normalized = {
        "step_id": expected_step_id,
        "step_goal": expected_step_goal,
        "rationale": rationale,
        "causal_chain": step_cc,
        "counterfactual_challenge_question": counterfactual_q,
        "expected_challenge_outcome": expected_outcome,
        "failure_reflecting": {"reason": fr_reason, "recovery_strategy": fr_recovery},
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


def get_run_summary_schema_fingerprint(run_summary_path: str) -> Optional[str]:
    if not os.path.exists(run_summary_path):
        return None
    try:
        rs = read_json(run_summary_path)
    except Exception:
        return None
    raw = rs.get("schema_fingerprint")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


@lru_cache(maxsize=1)
def three_stage_schema_fingerprint() -> str:
    """Fingerprint the prompts/schema spec used by the three-stage pipeline.

    This is primarily used to prevent unsafe "resume" across prompt/schema changes.
    """
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.py")
    try:
        with open(prompts_path, "rb") as f:
            data = f.read()
    except Exception:
        return "sha256:unknown"
    return "sha256:" + hashlib.sha256(data).hexdigest()


def guard_schema_fingerprint(
    run_summary_path: str,
    video_out: str,
    *,
    stage: str,
    overwrite: bool,
    allow_legacy_resume: bool,
    will_resume: bool,
) -> str:
    current_fp = three_stage_schema_fingerprint()
    existing_fp = get_run_summary_schema_fingerprint(run_summary_path)
    if existing_fp is not None and existing_fp != current_fp and not overwrite:
        raise RuntimeError(
            f"Refusing to run {stage}: schema_fingerprint mismatch.\n"
            f"- video_out: {os.path.abspath(video_out)}\n"
            f"- existing: {existing_fp}\n"
            f"- current : {current_fp}\n"
            "Use --overwrite to regenerate outputs, or use a different --output-root."
        )
    if will_resume and existing_fp is None and not allow_legacy_resume:
        raise RuntimeError(
            f"Refusing to resume {stage} legacy outputs without schema_fingerprint in run_summary.json.\n"
            f"- video_out: {os.path.abspath(video_out)}\n"
            "Re-run with --overwrite, or pass --allow-legacy-resume to bypass this check."
        )
    return current_fp


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
