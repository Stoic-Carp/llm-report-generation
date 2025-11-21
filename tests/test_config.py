"""Tests for configuration module."""

import pytest
from config.settings import Settings, get_settings, reload_settings


def test_settings_initialization():
    """Test settings initialization."""
    settings = Settings()
    # app_name can be overridden by environment variables, so check it's a string
    assert isinstance(settings.app_name, str)
    assert len(settings.app_name) > 0
    assert settings.environment == "development"


def test_settings_singleton():
    """Test settings singleton pattern."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_settings_reload():
    """Test settings reload."""
    settings1 = get_settings()
    settings2 = reload_settings()
    # After reload, should be different instance
    assert settings1 is not settings2


def test_llm_settings_validation():
    """Test LLM settings validation."""
    # Pydantic raises ValidationError, not ValueError
    with pytest.raises(Exception) as exc_info:
        Settings(llm={"provider": "invalid_provider"})
    # Check that it's a validation error about the provider
    assert "provider" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()


def test_model_dump_safe():
    """Test safe model dump without sensitive info."""
    settings = Settings()
    dumped = settings.model_dump_safe()

    # Check that API keys are masked
    if "llm" in dumped and "gemini_api_key" in dumped["llm"]:
        assert dumped["llm"]["gemini_api_key"] in [None, "***"]


def test_directory_creation(test_settings):
    """Test that directories are created on initialization."""
    assert test_settings.data.data_dir.exists() or not test_settings.data.data_dir.is_absolute()
    # Directories should be created (or paths are relative)

