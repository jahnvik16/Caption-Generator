"""Central configuration: env vars, model names, and tunable constants.

Everything that might change between environments or that someone would
want to tweak without reading code lives here, in one place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- API credentials -------------------------------------------------------

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# --- Model selection ---------------------------------------------------------
# Centralised so swapping a model (or provider, e.g. Groq for transcription)
# later is a one-line change rather than a hunt through every module.

TRANSCRIPTION_MODEL = "whisper-1"
VISION_MODEL = "gpt-4o-mini"
COPYWRITING_MODEL = "gpt-4o-mini"

# --- Pipeline tuning ---------------------------------------------------------

# If Whisper's average no_speech_prob across segments is at or above this
# value, a clip is treated as having no meaningful speech rather than
# trusting whatever text Whisper guessed. See transcriber.py for how this
# is used.
NO_SPEECH_PROBABILITY_THRESHOLD = 0.6

# Number of evenly spaced frames extracted from a clip for vision analysis.
VISION_FRAME_COUNT = 4

# Video containers this tool will attempt to process.
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}

# Max input file size, as a basic guard against accidentally trying to
# process something that isn't a short social clip.
MAX_VIDEO_SIZE_MB = 500
