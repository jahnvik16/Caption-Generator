"""Video validation and media extraction, via the system ffmpeg/ffprobe binaries.

Deliberately shells out to ffmpeg/ffprobe directly (through utils.run_subprocess)
rather than adding a wrapper library: ffmpeg-python and similar packages are
thin CLI wrappers anyway, and calling the binary directly keeps one fewer
dependency and makes the actual command being run easy to see and debug.

This module never calls an LLM API. Its only job is: is this file usable,
and if so, hand back an audio track (for transcription) and a handful of
sample frames (for vision), on demand.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from config import MAX_VIDEO_SIZE_MB, SUPPORTED_VIDEO_EXTENSIONS
from exceptions import InvalidVideoError, UnsupportedFormatError
from utils import get_logger, run_subprocess

logger = get_logger(__name__)


@dataclass
class VideoInfo:
    """Basic facts about a validated video, used to drive the rest of the pipeline."""

    path: Path
    duration_seconds: float
    has_audio_stream: bool


def _require_binary(name: str) -> None:
    """Fail fast with a clear message if ffmpeg/ffprobe isn't installed."""
    if shutil.which(name) is None:
        raise InvalidVideoError(
            f"'{name}' was not found on PATH. Install ffmpeg "
            f"(https://ffmpeg.org/download.html) and ensure it's on your PATH."
        )


def validate_video(path: Path) -> VideoInfo:
    """Confirm a video file exists, is a supported format, and is decodable.

    Raises:
        InvalidVideoError: file missing, empty, too large, or ffprobe can't
            parse it (corrupt/unreadable).
        UnsupportedFormatError: file extension isn't one we attempt to handle.
    """
    _require_binary("ffprobe")

    if not path.exists():
        raise InvalidVideoError(f"File not found: {path}")
    if not path.is_file():
        raise InvalidVideoError(f"Not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file extension '{path.suffix}'. Supported: {supported}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        raise InvalidVideoError(f"File is empty: {path}")
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise InvalidVideoError(
            f"File is {size_mb:.0f}MB, which exceeds the {MAX_VIDEO_SIZE_MB}MB "
            f"limit for this tool."
        )

    # ffprobe is the actual corruption/readability check: if it can't parse
    # the container/streams, the file is treated as invalid regardless of
    # what its extension claims.
    try:
        result = run_subprocess(
            [
                "ffprobe",
                "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
    except RuntimeError as exc:
        # ffprobe failed to parse the container/streams entirely - this is
        # the actual "corrupt file" case, not just a missing stream. Convert
        # to InvalidVideoError so it's caught and reported the same way as
        # every other bad-input case, with a clean message instead of raw
        # ffprobe stderr.
        raise InvalidVideoError(f"'{path}' could not be read; it may be corrupt or not a valid video file.") from exc

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidVideoError(f"Could not parse video metadata for {path}") from exc

    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        raise InvalidVideoError(f"No readable video stream found in {path} (file may be corrupt).")

    duration_raw = probe.get("format", {}).get("duration") or video_streams[0].get("duration")
    if duration_raw is None:
        raise InvalidVideoError(f"Could not determine duration for {path} (file may be corrupt).")

    return VideoInfo(
        path=path,
        duration_seconds=float(duration_raw),
        has_audio_stream=bool(audio_streams),
    )


def extract_audio(video: VideoInfo, workdir: Path) -> Path | None:
    """Extract the audio track as a small mono MP3 for transcription.

    Returns None if the container has no audio stream at all (skips a
    guaranteed-to-fail Whisper call). Downsampled to 16kHz mono, which is
    all Whisper needs for speech and keeps the file well under the 25MB
    API upload limit even for a longer clip.
    """
    if not video.has_audio_stream:
        logger.info("No audio stream present in %s; skipping audio extraction.", video.path.name)
        return None

    _require_binary("ffmpeg")
    audio_path = workdir / "audio.mp3"
    try:
        run_subprocess(
            [
                "ffmpeg", "-y",
                "-i", str(video.path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-q:a", "4",
                str(audio_path),
            ]
        )
    except RuntimeError as exc:
        raise InvalidVideoError(f"Could not extract audio from '{video.path}': {exc}") from exc
    return audio_path


def extract_frames(video: VideoInfo, workdir: Path, count: int) -> list[Path]:
    """Extract `count` evenly spaced JPEG frames for vision analysis.

    Frames are sampled from the 10%-90% span of the clip (not the exact
    start/end) so we skip any fade-in/fade-out or title-card padding that
    would give the vision model a blank frame to describe.
    """
    _require_binary("ffmpeg")
    frame_paths: list[Path] = []
    span_start = video.duration_seconds * 0.1
    span_end = video.duration_seconds * 0.9
    span = max(span_end - span_start, 0.1)

    for i in range(count):
        timestamp = span_start + span * (i / max(count - 1, 1))
        frame_path = workdir / f"frame_{i}.jpg"
        try:
            run_subprocess(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{timestamp:.3f}",
                    "-i", str(video.path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(frame_path),
                ]
            )
        except RuntimeError as exc:
            raise InvalidVideoError(f"Could not extract a frame from '{video.path}': {exc}") from exc
        frame_paths.append(frame_path)

    return frame_paths
