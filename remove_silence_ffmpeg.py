#!/usr/bin/env python3
"""Remove silence from video files using ffmpeg silencedetect filter."""

import argparse
import logging
import os
import re
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def detect_silence(
    input_path: str, noise_db: float = -30, min_duration: float = 0.5
) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect filter and parse silence intervals.

    Args:
        input_path: Path to the input video file.
        noise_db: Noise threshold in dB for silence detection.
        min_duration: Minimum silence duration in seconds.

    Returns:
        List of (start, end) tuples representing silence intervals.
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]
    logger.info("Running silence detection: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    starts = [
        float(m.group(1))
        for m in re.finditer(r"silence_start:\s*([0-9.]+)", stderr)
    ]
    ends = [
        float(m.group(1))
        for m in re.finditer(r"silence_end:\s*([0-9.]+)", stderr)
    ]

    # Pair up starts and ends
    segments = list(zip(starts, ends))

    # If there's an unpaired start (silence extends to the end), handle it
    if len(starts) > len(ends):
        total = get_video_duration(input_path)
        segments.append((starts[-1], total))

    logger.info("Detected %d silence segment(s)", len(segments))
    for i, (s, e) in enumerate(segments):
        logger.debug("  Silence %d: %.3f - %.3f (%.3f s)", i, s, e, e - s)

    return segments


def get_speech_segments(
    silence_segments: list[tuple[float, float]], total_duration: float
) -> list[tuple[float, float]]:
    """Invert silence segments to get speech/sound segments.

    Args:
        silence_segments: List of (start, end) silence intervals.
        total_duration: Total duration of the video in seconds.

    Returns:
        List of (start, end) tuples representing speech segments.
    """
    if not silence_segments:
        return [(0, total_duration)]

    speech = []
    prev_end = 0.0

    for s_start, s_end in sorted(silence_segments):
        if s_start > prev_end:
            speech.append((prev_end, s_start))
        prev_end = s_end

    if prev_end < total_duration:
        speech.append((prev_end, total_duration))

    logger.info("Identified %d speech segment(s)", len(speech))
    for i, (s, e) in enumerate(speech):
        logger.debug("  Speech %d: %.3f - %.3f (%.3f s)", i, s, e, e - s)

    return speech


def get_video_duration(input_path: str) -> float:
    """Use ffprobe to get total video duration.

    Args:
        input_path: Path to the input video file.

    Returns:
        Duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    logger.info("Getting video duration: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    logger.info("Video duration: %.3f s", duration)
    return duration


def cut_and_concat(
    input_path: str,
    segments: list[tuple[float, float]],
    output_path: str,
) -> None:
    """Cut speech segments and concatenate them into the output file.

    Args:
        input_path: Path to the input video file.
        segments: List of (start, end) tuples to keep.
        output_path: Path to the output video file.
    """
    if not segments:
        logger.warning("No segments to concatenate")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        segment_files = []

        for i, (start, end) in enumerate(segments):
            seg_path = os.path.join(tmpdir, f"segment_{i:04d}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ss",
                str(start),
                "-to",
                str(end),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                seg_path,
            ]
            logger.info("Cutting segment %d: %.3f - %.3f", i, start, end)
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            segment_files.append(seg_path)

        # Create concat list file
        concat_path = os.path.join(tmpdir, "concat_list.txt")
        with open(concat_path, "w") as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path}'\n")

        # Concatenate segments
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-c",
            "copy",
            output_path,
        ]
        logger.info("Concatenating %d segments into %s", len(segment_files), output_path)
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    logger.info("Output written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove silence from video files using ffmpeg."
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path (default: <input>_nosilence.mp4)",
    )
    parser.add_argument(
        "--noise-db",
        type=float,
        default=-30,
        help="Noise threshold in dB (default: -30)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum silence duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_nosilence{ext or '.mp4'}"

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    # Step 1: Get video duration
    total_duration = get_video_duration(args.input)

    # Step 2: Detect silence
    silence_segments = detect_silence(args.input, args.noise_db, args.min_duration)
    if not silence_segments:
        print("No silence detected. No processing needed.")
        return

    print(f"Detected {len(silence_segments)} silence segment(s):")
    for i, (s, e) in enumerate(silence_segments):
        print(f"  [{i}] {s:.3f}s - {e:.3f}s (duration: {e - s:.3f}s)")

    # Step 3: Get speech segments
    speech_segments = get_speech_segments(silence_segments, total_duration)
    print(f"\nKeeping {len(speech_segments)} speech segment(s):")
    for i, (s, e) in enumerate(speech_segments):
        print(f"  [{i}] {s:.3f}s - {e:.3f}s (duration: {e - s:.3f}s)")

    # Step 4: Cut and concatenate
    cut_and_concat(args.input, speech_segments, args.output)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
