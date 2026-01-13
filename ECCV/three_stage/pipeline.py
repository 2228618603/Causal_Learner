#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from typing import List

from common import ApiConfig, SamplingConfig, VIDEO_EXTS, collect_videos, default_output_root
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
        if "1" in stages:
            run_stage1_for_video(
                vp,
                args.output_root,
                api_cfg,
                sampling_cfg,
                overwrite=args.overwrite,
                max_retries=args.stage1_retries,
            )
        if "2" in stages:
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
        if "3" in stages:
            run_stage3_for_video(
                vp,
                args.output_root,
                api_cfg,
                sampling_cfg,
                overwrite=args.overwrite,
                max_retries=args.stage3_retries,
            )


if __name__ == "__main__":
    main()
