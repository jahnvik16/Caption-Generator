"""CLI entry point: orchestrates the full pipeline for one clip or a folder of clips.

    python main.py path/to/clip.mp4
    python main.py path/to/clips_folder/ --output result.json

For each clip: validate -> extract audio -> transcribe & detect speech ->
(if no speech) extract frames & describe visuals -> generate platform copy ->
assemble into the output schema. Every stage is wrapped so that one clip
failing (missing file, corrupt video, API error) produces a clear error
entry for that clip instead of crashing the whole run - a batch of clips
should get you N results even if one of them is bad.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from config import SUPPORTED_VIDEO_EXTENSIONS, VISION_FRAME_COUNT
from exceptions import CaptionGeneratorError
from llm import generate_copy
from schemas import ClipError, ClipResult
from transcriber import transcribe_audio
from utils import get_logger
from video_processor import extract_audio, extract_frames, validate_video
from vision import describe_visuals

logger = get_logger(__name__)


def process_clip(path: Path) -> ClipResult | ClipError:
    """Run the full pipeline for a single video file.

    Never raises: any failure at any stage is caught and returned as a
    ClipError, so callers processing multiple clips can keep going.
    """
    try:
        video = validate_video(path)

        with tempfile.TemporaryDirectory() as workdir_str:
            workdir = Path(workdir_str)

            audio_path = extract_audio(video, workdir)
            transcription = transcribe_audio(audio_path)

            if transcription.has_speech:
                copy_result = generate_copy(has_speech=True, transcript_text=transcription.full_text)
                return ClipResult(
                    source_file=path.name,
                    has_speech=True,
                    subtitles=transcription.subtitles,
                    subtitles_note=None,
                    content_summary=transcription.full_text,
                    platforms=copy_result.platforms,
                    dutch=copy_result.dutch,
                )

            frames = extract_frames(video, workdir, VISION_FRAME_COUNT)
            visual_description = describe_visuals(frames)
            copy_result = generate_copy(has_speech=False, visual_description=visual_description)
            return ClipResult(
                source_file=path.name,
                has_speech=False,
                subtitles=None,
                subtitles_note=(
                    "no speech detected in this clip, subtitles skipped, "
                    "copy generated from the visuals"
                ),
                content_summary=visual_description,
                platforms=copy_result.platforms,
                dutch=copy_result.dutch,
            )

    except CaptionGeneratorError as exc:
        logger.error("Failed to process %s: %s", path.name, exc)
        return ClipError(source_file=path.name, error=str(exc))
    except Exception as exc:  # last-resort catch: never let one bad clip crash the run
        logger.exception("Unexpected error processing %s", path.name)
        return ClipError(source_file=path.name, error=f"Unexpected error: {exc}")


def collect_input_files(input_path: Path) -> list[Path]:
    """Resolve the CLI input argument to a list of video files to process.

    Accepts either a single file or a directory (all supported-extension
    files inside it, sorted for deterministic output ordering).
    """
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )
        return files
    return [input_path]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate platform-native hooks, captions, hashtags and subtitles from a video clip."
    )
    parser.add_argument("input", help="Path to a video file, or a folder of video files.")
    parser.add_argument(
        "--output", "-o",
        help="Path to write the output JSON to. Defaults to <input name>_output.json.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.", file=sys.stderr)
        return 1

    files = collect_input_files(input_path)
    if not files:
        print(f"Error: no supported video files found in '{input_path}'.", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}
    for i, file_path in enumerate(files, start=1):
        logger.info("Processing clip %d/%d: %s", i, len(files), file_path.name)
        result = process_clip(file_path)
        results[f"clip_{i}"] = result.model_dump(exclude_none=True)

    output_path = Path(args.output) if args.output else Path(f"{input_path.stem}_output.json")
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    succeeded = sum(1 for r in results.values() if "error" not in r)
    failed = len(results) - succeeded
    print(f"\nProcessed {len(results)} clip(s): {succeeded} succeeded, {failed} failed.")
    print(f"Output written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
