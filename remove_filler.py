#!/usr/bin/env python3
"""Remove filler words from video files using OpenAI Whisper API and ffmpeg."""

import argparse
import logging
import os
import subprocess
import tempfile

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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

MAX_AUDIO_SIZE_MB = 25


def extract_audio(input_path: str, output_audio_path: str) -> None:
    """Extract audio from video as mp3 using ffmpeg.

    Args:
        input_path: Path to the input video file.
        output_audio_path: Path to the output audio file.

    Raises:
        RuntimeError: If the extracted audio exceeds the Whisper API size limit.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        output_audio_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    file_size_mb = os.path.getsize(output_audio_path) / (1024 * 1024)
    logger.info("Audio file size: %.2f MB", file_size_mb)

    if file_size_mb > MAX_AUDIO_SIZE_MB:
        raise RuntimeError(
            f"Audio file size ({file_size_mb:.1f} MB) exceeds Whisper API limit "
            f"({MAX_AUDIO_SIZE_MB} MB). Use a shorter video or compress the audio."
        )


def transcribe_with_timestamps(audio_path: str) -> list[dict]:
    """Transcribe audio using OpenAI Whisper API with word-level timestamps.

    Args:
        audio_path: Path to the audio file.

    Returns:
        List of word dicts with "word", "start", and "end" keys.
    """
    client = OpenAI()

    logger.info("Transcribing audio with Whisper API: %s", audio_path)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = response.words or []
    logger.info("Transcribed %d word(s)", len(words))
    return [{"word": w.word, "start": w.start, "end": w.end} for w in words]


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
                "-c",
                "copy",
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
        description="Remove filler words from video files using Whisper API and ffmpeg."
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path (default: <input>_nofiller.mp4)",
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

    # Step 1: Extract audio
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        extract_audio(args.input, audio_path)

        # Step 2: Transcribe with timestamps
        words = transcribe_with_timestamps(audio_path)

    # Step 3: Detect fillers
    filler_segments = detect_fillers(words, filler_list)
    if not filler_segments:
        print("No filler words detected. No processing needed.")
        return

    print(f"Detected {len(filler_segments)} filler word(s):")
    for i, (s, e) in enumerate(filler_segments):
        # Find the matching word for display
        matching = [w for w in words if w["start"] == s and w["end"] == e]
        word_text = matching[0]["word"] if matching else "?"
        print(f"  [{i}] '{word_text}' at {s:.3f}s - {e:.3f}s (duration: {e - s:.3f}s)")

    # Step 4: Get video duration and non-filler segments
    total_duration = get_video_duration(args.input)
    keep_segments = get_non_filler_segments(filler_segments, total_duration)

    print(f"\nKeeping {len(keep_segments)} segment(s):")
    for i, (s, e) in enumerate(keep_segments):
        print(f"  [{i}] {s:.3f}s - {e:.3f}s (duration: {e - s:.3f}s)")

    # Step 5: Cut and concatenate
    cut_and_concat(args.input, keep_segments, args.output)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
