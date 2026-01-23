#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import List, Optional

from common import (
    add_api_cli_args,
    add_sampling_cli_args,
    OutputDirCollisionError,
    VIDEO_EXTS,
    api_config_from_args,
    collect_videos,
    default_output_root,
    ensure_video_out_dir_safe,
    format_duration,
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


def _one_line(text: str, *, max_len: int = 500) -> str:
    line = str(text or "").replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if len(line) > max_len:
        return line[: max_len - 3] + "..."
    return line


def _append_schema_mismatch_txt(
    path: str,
    *,
    video_id: str,
    source_video: str,
    video_out: str,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    note: str = "",
) -> None:
    if not path:
        return
    errors = errors or []
    warnings = warnings or []
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    try:
        with open(path, "a", encoding="utf-8") as txt_file:
            if is_new:
                txt_file.write("# three_stage schema mismatch records (post-validate failures)\n")
                txt_file.write("# columns: time_utc, video_id, source_video, video_out, errors, warnings, note, error_sample\n")
            err_sample = " | ".join(_one_line(err, max_len=200) for err in errors[:6])
            txt_file.write(
                "\t".join(
                    [
                        now_utc_iso(),
                        f"video_id={video_id}",
                        f"source_video={_one_line(os.path.abspath(source_video), max_len=300)}",
                        f"video_out={_one_line(os.path.abspath(video_out), max_len=300)}",
                        f"errors={len(errors)}",
                        f"warnings={len(warnings)}",
                        f"note={_one_line(note, max_len=200)}" if note else "note=",
                        f"error_sample={err_sample}",
                    ]
                )
                + "\n"
            )
    except Exception:
        # Best-effort logging; never fail the pipeline due to an auxiliary file write.
        return


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
    parser.add_argument(
        "--schema-mismatch-txt",
        default="",
        help=(
            "Optional: write post-validate schema-mismatch records to this .txt file (default: "
            "<output_root>/schema_mismatch_videos.txt when --post-validate is enabled)."
        ),
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

    total = len(videos)
    run_started = time.perf_counter()
    ok_videos = 0
    failed_videos = 0

    schema_mismatch_txt = str(getattr(args, "schema_mismatch_txt", "") or "").strip()
    if args.post_validate and not schema_mismatch_txt:
        schema_mismatch_txt = os.path.join(args.output_root, "schema_mismatch_videos.txt")

    logger.info(
        "[pipeline] Start: "
        f"videos={total} stages={','.join(sorted(stages))} "
        f"output_root={os.path.abspath(args.output_root)} "
        f"overwrite={bool(args.overwrite)} continue_on_error={bool(args.continue_on_error)} post_validate={bool(args.post_validate)}"
    )
    logger.info(
        "[pipeline] API: "
        f"base={api_cfg.api_base_url} provider={api_cfg.model_provider_id} model={api_cfg.model_name} "
        f"max_tokens={int(api_cfg.max_tokens)} temp={float(api_cfg.temperature)} "
        f"call_retries={int(api_cfg.api_call_retries)} backoff_sec={float(api_cfg.api_call_retry_backoff_sec)}"
    )
    logger.info(
        "[pipeline] Sampling: "
        f"max_frames={int(sampling_cfg.max_frames)} jpeg_quality={int(sampling_cfg.jpeg_quality)} "
        f"embed_index_on_api_images={bool(api_cfg.embed_index_on_api_images)}"
    )

    for idx, vp in enumerate(videos, start=1):
        vid = video_id_from_path(vp)
        video_out = os.path.join(args.output_root, vid)

        logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} start: {os.path.abspath(vp)}")

        stage_failed = False
        for sid in ("1", "2", "3"):
            if sid not in stages:
                continue
            try:
                t0 = time.perf_counter()
                logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} stage={sid} start")
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
                dt = time.perf_counter() - t0
                logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} stage={sid} done in {format_duration(dt)}")
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
            failed_videos += 1
            elapsed = time.perf_counter() - run_started
            processed = ok_videos + failed_videos
            eta = (elapsed / processed) * (total - processed) if processed > 0 else 0.0
            logger.info(
                "[pipeline] Progress: "
                f"{processed}/{total} processed (ok={ok_videos}, failed={failed_videos}) "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            )
            continue

        video_ok = True
        if args.post_validate and "3" in stages:
            logged_to_txt = False
            try:
                from validate_three_stage_output import validate_three_stage_video_output_dir

                v0 = time.perf_counter()
                logger.info(f"[pipeline] ({idx}/{total}) video_id={vid} post-validate start")
                ok, errors, warnings = validate_three_stage_video_output_dir(video_out, check_deps=False)
                for w in warnings:
                    logger.warning(f"[validate] video_id={vid}: {w}")
                if not ok:
                    _append_schema_mismatch_txt(
                        schema_mismatch_txt,
                        video_id=vid,
                        source_video=vp,
                        video_out=video_out,
                        errors=errors,
                        warnings=warnings,
                        note="post_validate_schema_mismatch",
                    )
                    logged_to_txt = True
                    raise RuntimeError(" | ".join(errors[:20]))
                logger.info(
                    f"[pipeline] ({idx}/{total}) video_id={vid} post-validate OK in {format_duration(time.perf_counter() - v0)} "
                    f"(warnings={len(warnings)})"
                )
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
                video_ok = False
                msg = f"{type(e).__name__}: {e}"
                logger.exception(f"[pipeline] post-validate failed for video_id={vid}: {msg}")
                if not logged_to_txt:
                    _append_schema_mismatch_txt(
                        schema_mismatch_txt,
                        video_id=vid,
                        source_video=vp,
                        video_out=video_out,
                        errors=[msg],
                        warnings=[],
                        note="post_validate_exception",
                    )
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

        if video_ok:
            ok_videos += 1
        else:
            failed_videos += 1
        elapsed = time.perf_counter() - run_started
        processed = ok_videos + failed_videos
        eta = (elapsed / processed) * (total - processed) if processed > 0 else 0.0
        logger.info(
            "[pipeline] Progress: "
            f"{processed}/{total} processed (ok={ok_videos}, failed={failed_videos}) "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
        )

    total_elapsed = time.perf_counter() - run_started
    logger.info(
        "[pipeline] Done: "
        f"processed={total} ok={ok_videos} failed={failed_videos} elapsed={format_duration(total_elapsed)}"
    )


if __name__ == "__main__":
    main()
