#!/usr/bin/env python3
"""Detect and split video scenes using PySceneDetect."""

import argparse
import logging
import os

from scenedetect import ContentDetector, detect, split_video_ffmpeg

logger = logging.getLogger(__name__)


def detect_scenes(
    input_path: str, threshold: float = 27.0
) -> list[tuple]:
    """Detect scenes in a video using content-based detection.

    Args:
        input_path: Path to the input video file.
        threshold: Content detection threshold (default: 27.0).

    Returns:
        List of (start_timecode, end_timecode) tuples for each scene.
    """
    logger.info("Detecting scenes in %s (threshold=%.1f)", input_path, threshold)
    scene_list = detect(input_path, ContentDetector(threshold=threshold))

    logger.info("Detected %d scene(s)", len(scene_list))
    for i, (start, end) in enumerate(scene_list):
        logger.info("  Scene %d: %s - %s", i, start, end)

    return scene_list


def split_scenes(
    input_path: str, scene_list: list[tuple], output_dir: str
) -> None:
    """Split a video into separate files for each scene.

    Args:
        input_path: Path to the input video file.
        scene_list: List of (start_timecode, end_timecode) tuples.
        output_dir: Directory to write output scene files.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Splitting video into %d scene(s) in %s", len(scene_list), output_dir)
    split_video_ffmpeg(input_path, scene_list, output_dir=output_dir)
    logger.info("Scene splitting complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and split video scenes using PySceneDetect."
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument(
        "--output-dir",
        default="scenes",
        help="Output directory for split scenes (default: scenes)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Content detection threshold (default: 27.0)",
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

    if not os.path.isfile(args.input):
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    # Step 1: Detect scenes
    scene_list = detect_scenes(args.input, args.threshold)
    if not scene_list:
        print("No scene changes detected.")
        return

    # Step 2: Print summary
    print(f"Detected {len(scene_list)} scene(s):")
    for i, (start, end) in enumerate(scene_list):
        duration = end.get_seconds() - start.get_seconds()
        print(f"  [{i}] {start} - {end} (duration: {duration:.3f}s)")

    # Step 3: Split scenes
    split_scenes(args.input, scene_list, args.output_dir)
    print(f"\nScenes saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
