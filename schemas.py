"""Pydantic models defining the tool's output contract.

Keeping the schema in one place means every other module (llm.py, main.py)
imports the same source of truth instead of hand-building dicts, and the
final JSON is validated automatically before it ever reaches disk.

Field names and nesting deliberately mirror `example_output.json` provided
in the brief, so the output is a drop-in match for what the client expects.
Deviations from that example are noted inline and called out in the README.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SubtitleLine(BaseModel):
    """One timed subtitle line.

    `start`/`end` are kept as strings (e.g. "2.1") to match the example
    JSON's format exactly, since matching the client's expected shape
    matters more here than using native numeric types.
    """

    start: str
    end: str
    text: str


class PlatformCopy(BaseModel):
    """English hook/caption/hashtag set for a single platform."""

    on_screen_hook: str
    on_screen_hook_alt: str
    caption: str
    hashtags: list[str] = Field(min_length=3, max_length=5)

    @field_validator("hashtags")
    @classmethod
    def _hashtags_must_start_with_hash(cls, value: list[str]) -> list[str]:
        cleaned = []
        for tag in value:
            tag = tag.strip()
            if not tag.startswith("#"):
                tag = f"#{tag}"
            cleaned.append(tag)
        return cleaned


class DutchPlatformCopy(BaseModel):
    """Dutch rewrite of the hook/alt hook/caption for a single platform.

    Hashtags are intentionally excluded: the brief asks for a Dutch version
    of "hooks and captions" specifically, not hashtags. See README for the
    reasoning (hashtags are discovery tags; mixing translated and English
    tags in one list is more likely to hurt reach than help it).
    """

    on_screen_hook: str
    on_screen_hook_alt: str
    caption: str


class PlatformsBlock(BaseModel):
    instagram: PlatformCopy
    tiktok: PlatformCopy
    x: PlatformCopy


class DutchBlock(BaseModel):
    instagram: DutchPlatformCopy
    tiktok: DutchPlatformCopy
    x: DutchPlatformCopy


class ClipResult(BaseModel):
    """Full result for a single clip that was processed successfully."""

    source_file: str
    has_speech: bool
    subtitles: list[SubtitleLine] | None = None
    subtitles_note: str | None = None
    content_summary: str = Field(
        description=(
            "The transcript- or vision-derived description of the clip that "
            "grounded copy generation. Not in the original example JSON; "
            "included for transparency/debuggability, per the brief's "
            "interest in predictable, inspectable output."
        )
    )
    platforms: PlatformsBlock
    dutch: DutchBlock


class ClipError(BaseModel):
    """Result for a clip that could not be processed.

    Used instead of ClipResult under the same top-level key, so a consumer
    can distinguish success from failure by checking for the "error" field
    without the whole run raising an exception.
    """

    source_file: str
    error: str
