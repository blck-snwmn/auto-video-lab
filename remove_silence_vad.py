#!/usr/bin/env python3
"""Remove silence from video files using Silero VAD."""

import argparse
import logging
import os
import subprocess
import tempfile
import wave

import numpy as np
import torch

logger = logging.getLogger(__name__)


def read_wav(path: str) -> torch.Tensor:
    """Read a WAV file and return a float32 torch tensor.

    Args:
        path: Path to a 16kHz mono WAV file.

    Returns:
        1-D float32 tensor with samples normalized to [-1, 1].
    """
    with wave.open(path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        wav = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(wav)


def extract_audio(input_path: str, output_wav_path: str) -> None:
    """Extract audio from video and convert to 16kHz mono WAV.

    Args:
        input_path: Path to the input video file.
        output_wav_path: Path to the output WAV file.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_wav_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    logger.info("Audio extracted to %s", output_wav_path)


def detect_speech_segments(
    wav_path: str,
    threshold: float = 0.5,
    min_silence_duration_ms: int = 700,
    speech_pad_ms: int = 30,
) -> list[dict[str, float]]:
    """Detect speech segments using Silero VAD.

    Args:
        wav_path: Path to the 16kHz mono WAV file.
        threshold: Speech detection threshold (0.0 - 1.0).
        min_silence_duration_ms: Minimum silence duration in ms to split segments.
        speech_pad_ms: Padding added to each side of speech segments in ms.

    Returns:
        List of dicts with "start" and "end" keys in seconds.
    """
    from silero_vad import get_speech_timestamps, load_silero_vad

    logger.info("Loading Silero VAD model")
    model = load_silero_vad(onnx=True)

    logger.info("Reading audio from %s", wav_path)
    wav = read_wav(wav_path)

    logger.info(
        "Detecting speech (threshold=%.2f, min_silence=%dms, pad=%dms)",
        threshold,
        min_silence_duration_ms,
        speech_pad_ms,
    )
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    logger.info("Detected %d speech segment(s)", len(speech_timestamps))
    return speech_timestamps


def cut_and_concat(
    input_path: str,
    segments: list[dict[str, float]],
    output_path: str,
) -> None:
    """Cut speech segments from video and concatenate them.

    Args:
        input_path: Path to the input video file.
        segments: List of dicts with "start" and "end" keys in seconds.
        output_path: Path to the output video file.
    """
    if not segments:
        logger.warning("No segments to concatenate")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        segment_files = []

        for i, seg in enumerate(segments):
            start = seg["start"]
            end = seg["end"]
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

        concat_path = os.path.join(tmpdir, "concat_list.txt")
        with open(concat_path, "w") as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path}'\n")

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
        description="Remove silence from video files using Silero VAD."
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video file path (default: <input>_nosilence.mp4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD speech detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=700,
        help="Minimum silence duration in ms (default: 700)",
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

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")

        # Step 1: Extract audio
        print("Extracting audio...")
        extract_audio(args.input, wav_path)

        # Step 2: Detect speech segments using VAD
        print("Detecting speech segments with Silero VAD...")
        segments = detect_speech_segments(
            wav_path,
            threshold=args.threshold,
            min_silence_duration_ms=args.min_silence_ms,
        )

    if not segments:
        print("No speech detected. No processing needed.")
        return

    print(f"\nDetected {len(segments)} speech segment(s):")
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        print(f"  [{i}] {start:.3f}s - {end:.3f}s (duration: {end - start:.3f}s)")

    # Step 3: Cut and concatenate
    print("\nCutting and concatenating segments...")
    cut_and_concat(args.input, segments, args.output)
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
