"""Visual understanding for clips with no speech, via GPT-4o-mini vision.

Used only when transcriber.transcribe_audio() reports has_speech=False.
Sends a handful of sample frames (extracted by video_processor) as images
in one chat completion and asks for a plain-text description grounded
enough to write social copy from - not a shot-by-shot breakdown.
"""

from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY, VISION_MODEL
from exceptions import VisionAnalysisError
from utils import get_logger

logger = get_logger(__name__)

VISION_SYSTEM_PROMPT = (
    "You are describing a short vertical social media video clip for a copywriter "
    "who cannot see it. You are shown several frames sampled evenly across the clip. "
    "Describe, in 3-5 plain sentences: what is happening, who/what is on screen, the "
    "setting, the mood or tone, and anything that looks like the 'point' of the clip "
    "(a product, an outfit, a joke, a transformation, etc). Be concrete and specific "
    "rather than generic. Do not mention that you were shown frames or make up sound, "
    "dialogue, or on-screen text - describe only what is visually evident."
)


def _encode_frame(frame_path: Path) -> str:
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_visuals(frame_paths: list[Path]) -> str:
    """Return a text description of a clip's content, grounded in sample frames.

    Raises:
        VisionAnalysisError: the vision API call failed, or no frames were
            provided to analyse.
    """
    if not frame_paths:
        raise VisionAnalysisError("No frames were extracted to analyse.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode_frame(p)}"},
        }
        for p in frame_paths
    ]

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here are frames from the clip, in order:"},
                        *image_content,
                    ],
                },
            ],
            max_completion_tokens=300,
        )
    except Exception as exc:
        raise VisionAnalysisError(f"Vision analysis failed: {exc}") from exc

    description = (response.choices[0].message.content or "").strip()
    if not description:
        raise VisionAnalysisError("Vision model returned an empty description.")

    logger.info("Visual description: %s", description)
    return description
