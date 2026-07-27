"""Small shared helpers: logging setup and a subprocess convenience wrapper."""

from __future__ import annotations

import logging
import subprocess


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting.

    Configuring this once here (instead of each module calling
    `logging.basicConfig` independently) avoids duplicate log lines and
    keeps the format consistent across the whole pipeline.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess:
    """Run a command (ffmpeg/ffprobe) and raise with stderr on failure.

    Centralising this means every caller gets the same error-message
    behaviour instead of each call site re-implementing try/except around
    subprocess.run.
    """
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(command)}): {result.stderr.strip()}"
        )
    return result
