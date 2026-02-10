#!/usr/bin/env python3
"""Remove filler words from video files using faster-whisper (local) and ffmpeg."""

import argparse
import logging
import os
import subprocess
import tempfile

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEFAULT_FILLERS = [
    "えー",
    "えーと",
    "えっと",
    "あー",
    "あのー",
    "あの",
    "うーん",
    "んー",
    "まあ",
    "そのー",
    "その",
]


def transcribe_with_timestamps(
    input_path: str, model_size: str = "large-v3"
) -> list[dict]:
    """Transcribe audio using faster-whisper with word-level timestamps.

    Args:
        input_path: Path to the input video/audio file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3, etc.).

    Returns:
        List of word dicts with "word", "start", and "end" keys.
    """
    logger.info("Loading faster-whisper model: %s", model_size)
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    logger.info("Transcribing: %s", input_path)
    segments, info = model.transcribe(input_path, word_timestamps=True, language="ja")

    logger.info("Detected language: %s (prob=%.2f)", info.language, info.language_probability)

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})

    logger.info("Transcribed %d word(s)", len(words))
    return words


def _normalize_word(text: str) -> str:
    """Strip whitespace and trailing punctuation from a word."""
    return text.strip().rstrip("、。，．,.!?！？")


def detect_fillers(
    words: list[dict], filler_list: list[str], max_combine: int = 3
) -> list[tuple[float, float]]:
    """Detect filler words and return their time intervals.

    Supports multi-word filler detection by combining up to max_combine
    adjacent words (e.g. "えー" + "と" → "えーと").

    Args:
        words: List of word dicts from transcription.
        filler_list: List of filler words to detect.
        max_combine: Maximum number of adjacent words to combine for matching.

    Returns:
        List of (start, end) tuples for detected fillers.
    """
    filler_set = {f.strip() for f in filler_list}
    fillers = []
    skip_until = -1

    for i, w in enumerate(words):
        if i < skip_until:
            continue

        matched = False
        # Try combining N words (longest match first)
        for n in range(min(max_combine, len(words) - i), 0, -1):
            combined = "".join(
                _normalize_word(words[i + j]["word"]) for j in range(n)
            )
            if combined in filler_set:
                start = words[i]["start"]
                end = words[i + n - 1]["end"]
                fillers.append((start, end))
                logger.debug(
                    "Filler detected: '%s' (%d word(s)) at %.3f - %.3f",
                    combined, n, start, end,
                )
                skip_until = i + n
                matched = True
                break

        if not matched:
            word = _normalize_word(w["word"])
            if word in filler_set:
                fillers.append((w["start"], w["end"]))
                logger.debug("Filler detected: '%s' at %.3f - %.3f", word, w["start"], w["end"])

    logger.info("Detected %d filler(s)", len(fillers))
    return fillers


def get_non_filler_segments(
    filler_segments: list[tuple[float, float]], total_duration: float
) -> list[tuple[float, float]]:
    """Invert filler segments to get segments to keep.

    Args:
        filler_segments: List of (start, end) filler intervals.
        total_duration: Total duration of the video in seconds.

    Returns:
        List of (start, end) tuples representing non-filler segments.
    """
    if not filler_segments:
        return [(0, total_duration)]

    segments = []
    prev_end = 0.0

    for f_start, f_end in sorted(filler_segments):
        if f_start > prev_end:
            segments.append((prev_end, f_start))
        prev_end = max(prev_end, f_end)

    if prev_end < total_duration:
        segments.append((prev_end, total_duration))

    logger.info("Identified %d non-filler segment(s)", len(segments))
    for i, (s, e) in enumerate(segments):
        logger.debug("  Segment %d: %.3f - %.3f (%.3f s)", i, s, e, e - s)

    return segments


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
    """Cut non-filler segments and concatenate them into the output file.

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
        description="Remove filler words from video using faster-whisper (local) and ffmpeg."
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path (default: <input>_nofiller.mp4)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: large-v3)",
    )
    parser.add_argument(
        "--fillers",
        nargs="+",
        default=[],
        help="Additional filler words to detect",
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
        args.output = f"{base}_nofiller{ext or '.mp4'}"

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    filler_list = DEFAULT_FILLERS + args.fillers

    # Step 1: Transcribe with word timestamps (faster-whisper handles audio extraction internally)
    words = transcribe_with_timestamps(args.input, model_size=args.model)

    # Step 2: Detect fillers
    filler_segments = detect_fillers(words, filler_list)
    if not filler_segments:
        print("No filler words detected. No processing needed.")
        return

    print(f"Detected {len(filler_segments)} filler word(s):")
    for i, (s, e) in enumerate(filler_segments):
        matching = [w for w in words if w["start"] == s and w["end"] == e]
        word_text = matching[0]["word"] if matching else "?"
        print(f"  [{i}] '{word_text}' at {s:.3f}s - {e:.3f}s (duration: {e - s:.3f}s)")

    # Step 3: Get video duration and non-filler segments
    total_duration = get_video_duration(args.input)
    keep_segments = get_non_filler_segments(filler_segments, total_duration)

    print(f"\nKeeping {len(keep_segments)} segment(s):")
    for i, (s, e) in enumerate(keep_segments):
        print(f"  [{i}] {s:.3f}s - {e:.3f}s (duration: {e - s:.3f}s)")

    # Step 4: Cut and concatenate
    cut_and_concat(args.input, keep_segments, args.output)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
