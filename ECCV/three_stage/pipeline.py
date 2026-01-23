#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
from typing import List

from common import (
    add_api_cli_args,
    add_sampling_cli_args,
    OutputDirCollisionError,
    VIDEO_EXTS,
    api_config_from_args,
    collect_videos,
    default_output_root,
    ensure_video_out_dir_safe,
    logger,
    now_utc_iso,
    sampling_config_from_args,
    three_stage_schema_fingerprint,
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

    add_api_cli_args(parser, include_no_embed_index=True)
    add_sampling_cli_args(parser, default_max_frames=50, default_jpeg_quality=95)

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
        "--allow-legacy-resume",
        action="store_true",
        help="Allow resuming cached outputs whose run_summary.json lacks schema_fingerprint (legacy outputs).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run dependency/collision checks and exit (does not call the model, does not cut clips).",
    )
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

    api_cfg = api_config_from_args(args)
    sampling_cfg = sampling_config_from_args(args)

    videos: List[str] = []
    if args.input_video:
        videos = [args.input_video]
    else:
        videos = collect_videos(args.input_video_dir, VIDEO_EXTS)
    if not videos:
        raise SystemExit("No videos found.")

    schema_fp = three_stage_schema_fingerprint()

    missing_inputs = [vp for vp in videos if not os.path.isfile(vp)]
    if missing_inputs:
        raise SystemExit("Missing/non-file inputs:\n" + "\n".join(f"- {p}" for p in missing_inputs))

    # Fail fast on filename-stem collisions: the pipeline uses `video_id_from_path()` for `<video_out>/`.
    vid_to_paths: dict[str, List[str]] = {}
    for vp in videos:
        vid_to_paths.setdefault(video_id_from_path(vp), []).append(vp)
    dup = {vid: ps for vid, ps in vid_to_paths.items() if len(ps) > 1}
    if dup:
        lines: List[str] = ["Duplicate video_id detected (filename stem collision):"]
        for vid, ps in sorted(dup.items()):
            lines.append(f"- video_id={vid}:")
            for p in ps:
                lines.append(f"  - {p}")
        lines.append("Rename colliding videos (or use different --output-root per source set) to avoid corrupt outputs.")
        raise SystemExit("\n".join(lines))

    if args.preflight_only:
        errs: List[str] = []
        try:
            import cv2  # noqa: F401
        except Exception as e:
            errs.append(f"Missing dependency: opencv-python (cv2). Install: pip install opencv-python. Detail: {e}")
        try:
            import openai  # noqa: F401
        except Exception as e:
            errs.append(f"Missing dependency: openai. Install: pip install openai. Detail: {e}")

        if "2" in stages:
            if shutil.which(args.ffmpeg_bin) is None and not os.path.exists(args.ffmpeg_bin):
                errs.append(
                    f"ffmpeg binary not found: '{args.ffmpeg_bin}'. Install ffmpeg or pass a valid path via --ffmpeg-bin."
                )
        try:
            os.makedirs(args.output_root, exist_ok=True)
        except Exception as e:
            errs.append(f"Failed to create output_root: {args.output_root}: {e}")

        for vp in videos:
            vid = video_id_from_path(vp)
            video_out = os.path.join(args.output_root, vid)
            try:
                ensure_video_out_dir_safe(video_out, vp)
            except OutputDirCollisionError as e:
                errs.append(str(e).strip())

        if errs:
            for e in errs:
                logger.error("[preflight] " + e.replace("\n", " | "))
            raise SystemExit(1)
        logger.info(f"[preflight] OK (schema_fingerprint={schema_fp})")
        raise SystemExit(0)

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
                        allow_legacy_resume=args.allow_legacy_resume,
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
                        allow_legacy_resume=args.allow_legacy_resume,
                    )
                else:
                    run_stage3_for_video(
                        vp,
                        args.output_root,
                        api_cfg,
                        sampling_cfg,
                        overwrite=args.overwrite,
                        max_retries=args.stage3_retries,
                        allow_legacy_resume=args.allow_legacy_resume,
                    )
            except OutputDirCollisionError as e:
                stage_failed = True
                msg = f"{type(e).__name__}: {e}"
                logger.error(f"[pipeline] video_id={vid} stage={sid} aborted: {msg}")
                # Do NOT write into run_summary.json here: the directory may belong to another source video.
                if args.continue_on_error:
                    break
                raise
            except (Exception, SystemExit) as e:
                stage_failed = True
                msg = f"{type(e).__name__}: {e}"
                logger.exception(f"[pipeline] video_id={vid} stage={sid} failed: {msg}")
                try:
                    update_run_summary(
                        os.path.join(video_out, "run_summary.json"),
                        {
                            "source_video": os.path.abspath(vp),
                            "video_id": vid,
                            "output_root": os.path.abspath(args.output_root),
                            "schema_fingerprint": schema_fp,
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
                            "source_video": os.path.abspath(vp),
                            "video_id": vid,
                            "output_root": os.path.abspath(args.output_root),
                            "schema_fingerprint": schema_fp,
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
                            "source_video": os.path.abspath(vp),
                            "video_id": vid,
                            "output_root": os.path.abspath(args.output_root),
                            "schema_fingerprint": schema_fp,
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
