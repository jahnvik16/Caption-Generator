"""Speech-to-text and speech/no-speech detection via the OpenAI Whisper API.

The tricky requirement from the brief isn't transcription itself, it's
telling the two clips apart automatically: a clip can have an audio track
(both sample clips do) without having speech on it. Whisper being willing to
transcribe *something* isn't proof of speech - background music or ambient
noise can produce a low-confidence guess. So detection relies on Whisper's
own `no_speech_prob` per segment rather than "did we get text back".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from config import NO_SPEECH_PROBABILITY_THRESHOLD, OPENAI_API_KEY, TRANSCRIPTION_MODEL
from exceptions import TranscriptionError
from schemas import SubtitleLine
from utils import get_logger

logger = get_logger(__name__)

MAX_WORDS_PER_LINE = 7
MAX_SECONDS_PER_LINE = 3.5

# Even the fastest real speech rarely produces individual word durations
# under this. A real-corpus case (a genuinely silent/music-only clip) showed
# Whisper's known failure mode on non-speech audio: it hallucinates a
# plausible-sounding generic phrase ("thank you", "watch the video" - the
# kind of boilerplate that's common in its training data's silent/outro
# segments) while its own word-level forced alignment breaks down, producing
# words crammed into a near-zero span. no_speech_prob can come back
# deceptively low for this exact failure, so it's checked separately here
# rather than trusted alone.
MIN_PLAUSIBLE_WORD_DURATION_SECONDS = 0.05
HALLUCINATED_WORD_FRACTION_THRESHOLD = 0.3


@dataclass
class TranscriptionResult:
    has_speech: bool
    subtitles: list[SubtitleLine]
    full_text: str


def transcribe_audio(audio_path: Path | None) -> TranscriptionResult:
    """Transcribe an audio file and decide whether it actually contains speech.

    Args:
        audio_path: Path to an extracted audio track, or None if the source
            video had no audio stream at all (short-circuits without an
            API call).

    Returns:
        A TranscriptionResult with has_speech=False and empty subtitles if
        no speech was detected - callers must not invent subtitles in that
        case, per the brief.

    Raises:
        TranscriptionError: the Whisper API call itself failed (network,
        auth, rate limit, etc.) - distinct from "call succeeded but found
        no speech".
    """
    if audio_path is None:
        logger.info("No audio track to transcribe; treating as no-speech.")
        return TranscriptionResult(has_speech=False, subtitles=[], full_text="")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"],
            )
    except Exception as exc:  # OpenAI raises several distinct exception types
        raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

    segments = getattr(response, "segments", None) or []
    words = getattr(response, "words", None) or []
    text = (getattr(response, "text", "") or "").strip()

    if not segments or not text:
        logger.info("Whisper returned no segments/text; treating as no-speech.")
        return TranscriptionResult(has_speech=False, subtitles=[], full_text="")

    avg_no_speech_prob = sum(getattr(s, "no_speech_prob", 0.0) for s in segments) / len(segments)
    if avg_no_speech_prob >= NO_SPEECH_PROBABILITY_THRESHOLD:
        logger.info(
            "Average no_speech_prob %.2f >= threshold %.2f; treating as no-speech.",
            avg_no_speech_prob,
            NO_SPEECH_PROBABILITY_THRESHOLD,
        )
        return TranscriptionResult(has_speech=False, subtitles=[], full_text="")

    if _looks_like_hallucinated_transcript(words):
        logger.info(
            "Word timestamps look hallucinated (implausibly short word "
            "durations despite a low no_speech_prob); treating as no-speech."
        )
        return TranscriptionResult(has_speech=False, subtitles=[], full_text="")

    subtitles = _build_subtitle_lines(words)
    if not subtitles:
        # Safety net: if word-level timestamps came back empty (e.g. an API
        # response that only populated segments), fall back to one subtitle
        # line per Whisper segment instead of silently returning has_speech=
        # True with an empty subtitle list. Segments are guaranteed non-empty
        # here since we already returned early above if they were.
        logger.warning("No word-level timestamps available; falling back to segment-level subtitles.")
        subtitles = [
            SubtitleLine(
                start=f"{float(getattr(s, 'start', 0.0)):.1f}",
                end=f"{float(getattr(s, 'end', 0.0)):.1f}",
                text=(getattr(s, "text", "") or "").strip(),
            )
            for s in segments
        ]
    return TranscriptionResult(has_speech=True, subtitles=subtitles, full_text=text)


def _looks_like_hallucinated_transcript(words: list) -> bool:
    """Detect Whisper's forced-alignment breakdown on non-speech audio.

    Flags the transcript if a large share of words have an implied duration
    no real speaker could produce (a cluster of words crammed into a
    near-zero time span), which is the concrete signature this was written
    to catch - see the module-level comment above the threshold constants.
    """
    if not words:
        return False
    durations = [
        float(getattr(w, "end", 0.0)) - float(getattr(w, "start", 0.0))
        for w in words
    ]
    implausible = sum(1 for d in durations if d < MIN_PLAUSIBLE_WORD_DURATION_SECONDS)
    return (implausible / len(durations)) >= HALLUCINATED_WORD_FRACTION_THRESHOLD


def _build_subtitle_lines(words: list) -> list[SubtitleLine]:
    """Group word-level timestamps into short, readable subtitle lines.

    Greedily fills each line until it hits MAX_WORDS_PER_LINE words or
    MAX_SECONDS_PER_LINE of duration, whichever comes first - keeps pace
    with speech instead of producing one long line per Whisper segment.
    """
    lines: list[SubtitleLine] = []
    current_words: list[str] = []
    line_start: float | None = None
    line_end: float = 0.0

    for w in words:
        word_text = (getattr(w, "word", "") or "").strip()
        if not word_text:
            continue
        word_start = float(getattr(w, "start", line_end))
        word_end = float(getattr(w, "end", word_start))

        if line_start is None:
            line_start = word_start

        current_words.append(word_text)
        line_end = word_end

        duration = line_end - line_start
        if len(current_words) >= MAX_WORDS_PER_LINE or duration >= MAX_SECONDS_PER_LINE:
            lines.append(
                SubtitleLine(start=f"{line_start:.1f}", end=f"{line_end:.1f}", text=" ".join(current_words))
            )
            current_words = []
            line_start = None

    if current_words and line_start is not None:
        lines.append(
            SubtitleLine(start=f"{line_start:.1f}", end=f"{line_end:.1f}", text=" ".join(current_words))
        )

    return lines
