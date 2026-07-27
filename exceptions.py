"""Custom exceptions for the caption generator pipeline.

Named exception types (rather than bare RuntimeError/ValueError everywhere)
let main.py catch each failure mode separately and turn it into the clear,
specific error message the brief asks for, instead of a stack trace.
"""


class CaptionGeneratorError(Exception):
    """Base class for all errors raised by this pipeline."""


class InvalidVideoError(CaptionGeneratorError):
    """Raised when a video file is missing, corrupt, or unreadable."""


class UnsupportedFormatError(CaptionGeneratorError):
    """Raised when a file's container/codec isn't one ffmpeg can decode."""


class TranscriptionError(CaptionGeneratorError):
    """Raised when the speech-to-text API call fails."""


class VisionAnalysisError(CaptionGeneratorError):
    """Raised when the vision (frame description) API call fails."""


class CopyGenerationError(CaptionGeneratorError):
    """Raised when the LLM copywriting call fails or returns invalid output."""
