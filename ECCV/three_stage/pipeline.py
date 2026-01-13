#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import List

from common import (
    ApiConfig,
    SamplingConfig,
    VIDEO_EXTS,
    collect_videos,
    default_output_root,
    logger,
    now_utc_iso,
    update_run_summary,
    video_id_from_path,
)
from stage1_generate_draft import run_stage1_for_video
from stage2_localize_and_cut import run_stage2_for_video
from stage3_refine_and_keyframes import run_stage3_for_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-stage pipeline: draft -> localize/cut -> refine+keyframes.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-video", help="Path to one video file.")
    src.add_argument("--input-video-dir", help="Directory of videos to process.")
    parser.add_argument("--output-root", default=default_output_root(), help="Output root under ECCV/three_stage/...")

    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--api-base", default=os.environ.get("API_BASE_URL", "http://model.mify.ai.srv/v1"))
    parser.add_argument("--provider", default=os.environ.get("MODEL_PROVIDER_ID", "vertex_ai"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemini-3-pro-preview"))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "30000")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.2")))
    parser.add_argument("--api-call-retries", type=int, default=int(os.environ.get("API_CALL_RETRIES", "3")))
    parser.add_argument(
        "--api-call-retry-backoff-sec",
        type=float,
        default=float(os.environ.get("API_CALL_RETRY_BACKOFF_SEC", "1.0")),
    )
    parser.add_argument("--no-embed-index", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=95)

    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--cut-mode", choices=["copy", "reencode"], default="reencode")
    parser.add_argument("--seek-slop-sec", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--stage1-retries", type=int, default=3)
    parser.add_argument("--stage2-retries", type=int, default=3)
    parser.add_argument("--stage3-retries", type=int, default=3)

    parser.add_argument("--stages", default="1,2,3", help="Comma-separated subset of stages to run (e.g., 1,2 or 3).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing the next video if one video fails; writes failure info into <video_out>/run_summary.json.",
    )
    parser.add_argument(
        "--post-validate",
        action="store_true",
        help="After successful Stage 3, run `validate_three_stage_output.py` checks on the output folder.",
    )
    args = parser.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if not stages.issubset({"1", "2", "3"}):
        raise SystemExit(f"Invalid --stages: {args.stages}")

    api_cfg = ApiConfig(
        api_key=args.api_key,
        api_base_url=args.api_base,
        model_provider_id=args.provider,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_call_retries=args.api_call_retries,
        api_call_retry_backoff_sec=args.api_call_retry_backoff_sec,
        embed_index_on_api_images=not args.no_embed_index,
        verbose=args.verbose,
    )
    sampling_cfg = SamplingConfig(max_frames=args.max_frames, jpeg_quality=args.jpeg_quality)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)
    if not videos:
        raise SystemExit("No videos found.")

    for vp in videos:
        vid = video_id_from_path(vp)
        video_out = os.path.join(args.output_root, vid)

        stage_failed = False
        for sid in ("1", "2", "3"):
            if sid not in stages:
                continue
            try:
                if sid == "1":
                    run_stage1_for_video(
                        vp,
                        args.output_root,
                        api_cfg,
                        sampling_cfg,
                        overwrite=args.overwrite,
                        max_retries=args.stage1_retries,
                    )
                elif sid == "2":
                    run_stage2_for_video(
                        vp,
                        args.output_root,
                        api_cfg,
                        ffmpeg_bin=args.ffmpeg_bin,
                        overwrite=args.overwrite,
                        max_retries=args.stage2_retries,
                        cut_mode=args.cut_mode,
                        seek_slop_sec=args.seek_slop_sec,
                        crf=args.crf,
                        preset=args.preset,
                        keep_audio=args.keep_audio,
                    )
                else:
                    run_stage3_for_video(
                        vp,
                        args.output_root,
                        api_cfg,
                        sampling_cfg,
                        overwrite=args.overwrite,
                        max_retries=args.stage3_retries,
                    )
            except (Exception, SystemExit) as e:
                stage_failed = True
                msg = f"{type(e).__name__}: {e}"
                logger.exception(f"[pipeline] video_id={vid} stage={sid} failed: {msg}")
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "updated_at_utc": now_utc_iso(),
                            f"stage{sid}": {
                                "status": "failed",
                                "failed_at_utc": now_utc_iso(),
                                "error": msg,
                            },
                        },
                    )
                except Exception:
                    pass
                if args.continue_on_error:
                    break
                raise

        if stage_failed:
            continue

        if args.post_validate and "3" in stages:
            try:
                from validate_three_stage_output import validate_three_stage_video_output_dir

                ok, errors, warnings = validate_three_stage_video_output_dir(video_out, check_deps=False)
                for w in warnings:
                    logger.warning(f"[validate] video_id={vid}: {w}")
                if not ok:
                    raise RuntimeError(" | ".join(errors[:20]))
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "updated_at_utc": now_utc_iso(),
                            "post_validate": {
                                "status": "completed",
                                "validated_at_utc": now_utc_iso(),
                                "warnings_count": len(warnings),
                                "warnings_sample": warnings[:10],
                            },
                        },
                    )
                except Exception:
                    pass
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                logger.exception(f"[pipeline] post-validate failed for video_id={vid}: {msg}")
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "updated_at_utc": now_utc_iso(),
                            "post_validate": {
                                "status": "failed",
                                "failed_at_utc": now_utc_iso(),
                                "error": msg,
                            },
                        },
                    )
                except Exception:
                    pass
                if not args.continue_on_error:
                    raise


if __name__ == "__main__":
    main()
