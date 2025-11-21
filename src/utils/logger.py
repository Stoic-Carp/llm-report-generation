"""Logging utilities using loguru and pydantic-settings."""

import sys
from loguru import logger
from config.settings import get_settings


def setup_logger():
    """Configure logger using pydantic-settings.

    Returns:
        Configured logger instance.
    """
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Add console handler
    if settings.logging.log_to_console:
        logger.add(
            sys.stdout,
            format=settings.logging.format,
            level=settings.logging.level,
            colorize=True,
        )

    # Add file handler
    if settings.logging.log_to_file:
        log_file = settings.logging.log_dir / "app_{time}.log"
        try:
            logger.add(
                log_file,
                format=settings.logging.format,
                level=settings.logging.level,
                rotation=settings.logging.rotation,
                retention=settings.logging.retention,
                compression="zip",
            )
        except (PermissionError, OSError) as e:
            # Fallback to console-only logging if file logging fails
            # This can happen in Docker containers with volume mount permission issues
            logger.warning(
                f"Failed to initialize file logging: {e}. "
                "Falling back to console-only logging."
            )

    return logger


# Initialize logger
app_logger = setup_logger()

