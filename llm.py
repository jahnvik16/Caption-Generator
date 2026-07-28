"""Copywriting generation via GPT-4o-mini with Structured Outputs.

Uses the OpenAI SDK's `.parse()` method with a Pydantic response model,
which constrains the model's output to match the schema at the API level -
the model is unable to return malformed JSON or omit a required field, as
opposed to prompting for JSON and hoping. This directly serves the brief's
"structured, predictable output" requirement: it removes an entire class of
failure (retry-on-bad-JSON loops, regex patching) rather than mitigating it.
"""

from __future__ import annotations

from pydantic import BaseModel

from config import COPYWRITING_MODEL, OPENAI_API_KEY
from exceptions import CopyGenerationError
from openai import OpenAI
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import DutchBlock, PlatformsBlock
from utils import get_logger

logger = get_logger(__name__)


class CopyGenerationOutput(BaseModel):
    """The shape the LLM must return - platform copy plus its Dutch rewrite."""

    platforms: PlatformsBlock
    dutch: DutchBlock


def generate_copy(
    has_speech: bool,
    transcript_text: str = "",
    visual_description: str = "",
) -> CopyGenerationOutput:
    """Generate EN hooks/captions/hashtags and Dutch rewrites for one clip.

    Args:
        has_speech: whether this clip had detected speech.
        transcript_text: full transcript text, used when has_speech is True.
        visual_description: vision model's description, used otherwise.

    Raises:
        CopyGenerationError: the API call failed, or returned output that
            didn't satisfy the schema (e.g. wrong hashtag count) after the
            SDK's own parsing/validation.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = build_user_prompt(has_speech, transcript_text, visual_description)

    try:
        completion = client.chat.completions.parse(
            model=COPYWRITING_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=CopyGenerationOutput,
            temperature=0.8,
            # Unbounded output let one real run burn the full 16384-token
            # ceiling on a degenerate repetition loop before failing with an
            # unhelpful "length limit reached" parse error. A real response
            # here is a few hundred tokens (short hooks/captions/hashtags x3
            # platforms x2 languages); this cap fails fast and cheap instead.
            max_completion_tokens=2000,
        )
    except Exception as exc:
        raise CopyGenerationError(f"Copy generation failed: {exc}") from exc

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        raise CopyGenerationError(
            f"Model declined to generate copy for this clip: {refusal or 'no reason given'}"
        )

    _normalize_hashtag_counts(parsed.platforms)
    return parsed


def _normalize_hashtag_counts(platforms: PlatformsBlock) -> None:
    """Enforce the 3-5 hashtag count the schema itself can't guarantee.

    OpenAI's Structured Outputs strict mode only constrains type/structure,
    not JSON Schema list-length keywords - the model can legally return 2 or
    6 hashtags despite the system prompt asking for 3-5. Truncate anything
    over 5 rather than failing the whole clip over an easily-fixed excess;
    just log a warning if under 3, since fabricating hashtags to hit a count
    would violate the "never invent details" rule.
    """
    for platform_name in ("instagram", "tiktok", "x"):
        copy = getattr(platforms, platform_name)
        if len(copy.hashtags) > 5:
            logger.warning(
                "%s returned %d hashtags, truncating to 5: %s",
                platform_name, len(copy.hashtags), copy.hashtags,
            )
            copy.hashtags = copy.hashtags[:5]
        elif len(copy.hashtags) < 3:
            logger.warning(
                "%s returned only %d hashtag(s): %s",
                platform_name, len(copy.hashtags), copy.hashtags,
            )
