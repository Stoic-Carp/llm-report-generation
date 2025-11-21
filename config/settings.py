"""Configuration management using pydantic-settings.

This module provides type-safe, validated configuration management with
automatic environment variable loading and hierarchical settings support.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional
from pathlib import Path


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    provider: Literal["ollama", "gemini", "anthropic", "openai"] = Field(
        default="ollama",
        description="LLM provider to use",
    )
    model: str = Field(
        default="granite4:latest",
        description="Model name/identifier",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens for generation",
    )
    timeout: int = Field(
        default=120,
        description="Request timeout in seconds",
    )

    # Provider-specific settings
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
        validation_alias="OLLAMA_HOST",  # Allow direct OLLAMA_HOST env var
    )

    # API Keys (loaded from environment)
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider.

        Args:
            v: Provider name to validate.

        Returns:
            Validated provider name.

        Raises:
            ValueError: If provider is not in valid list.
        """
        valid_providers = ["ollama", "gemini", "anthropic", "openai"]
        if v not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:
        """Validate model name based on provider.

        Args:
            v: Model name to validate.
            info: Validation info containing provider.

        Returns:
            Validated model name.
        """
        provider = info.data.get("provider")

        valid_models = {
            "ollama": ["granite4:latest", "llama3.1:8b", "llama3.1:70b", "mistral:7b", "granite3.1:8b"],
            "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        }

        if provider and provider in valid_models:
            if v not in valid_models[provider]:
                print(f"Warning: {v} may not be a valid {provider} model")

        return v


class DataSettings(BaseSettings):
    """Data processing settings."""

    data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw data files",
    )
    processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed data",
    )
    sample_dir: Path = Field(
        default=Path("data/sample"),
        description="Directory for sample data",
    )

    # Data validation
    required_columns: list[str] = Field(
        default=["date", "region", "model", "sales", "price"],
        description="Required columns in dataset",
    )
    date_format: str = Field(
        default="%Y-%m-%d",
        description="Expected date format",
    )

    # Processing options
    handle_missing: Literal["drop", "fill", "error"] = Field(
        default="error",
        description="How to handle missing values",
    )

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        case_sensitive=False,
        extra="ignore",
    )


class VisualizationSettings(BaseSettings):
    """Visualization settings."""

    output_dir: Path = Field(
        default=Path("outputs/plots"),
        description="Directory for generated plots",
    )
    dpi: int = Field(
        default=300,
        ge=72,
        description="DPI for saved plots",
    )
    format: Literal["png", "jpg", "svg", "pdf"] = Field(
        default="png",
        description="Plot output format",
    )
    style: str = Field(
        default="seaborn-v0_8",
        description="Matplotlib style",
    )
    figsize: tuple[int, int] = Field(
        default=(12, 8),
        description="Default figure size",
    )
    color_palette: str = Field(
        default="husl",
        description="Seaborn color palette",
    )

    # Plot generation
    min_plots: int = Field(default=4, ge=1)
    max_plots: int = Field(default=8, le=15)

    model_config = SettingsConfigDict(
        env_prefix="VIZ_",
        case_sensitive=False,
        extra="ignore",
    )


class ReportSettings(BaseSettings):
    """Report generation settings."""

    output_dir: Path = Field(
        default=Path("outputs/reports"),
        description="Directory for generated reports",
    )

    # Output formats
    generate_pdf: bool = Field(default=True)
    generate_html: bool = Field(default=True)
    generate_word: bool = Field(default=True)
    generate_markdown: bool = Field(default=True)

    # Report structure
    include_executive_summary: bool = Field(default=True)
    include_methodology: bool = Field(default=True)
    include_appendix: bool = Field(default=True)

    # Styling
    template_dir: Path = Field(
        default=Path("src/reporting/templates"),
        description="Report template directory",
    )
    company_name: str = Field(
        default="BMW Sales Analysis",
        description="Company/report name",
    )
    author: str = Field(
        default="AI-Powered Analytics",
        description="Report author",
    )

    model_config = SettingsConfigDict(
        env_prefix="REPORT_",
        case_sensitive=False,
        extra="ignore",
    )


class EvaluationSettings(BaseSettings):
    """Evaluation framework settings."""

    enabled: bool = Field(
        default=True,
        description="Enable quality evaluation",
    )
    output_dir: Path = Field(
        default=Path("outputs/evaluation"),
        description="Directory for evaluation reports",
    )

    # Evaluation criteria
    min_coverage_score: float = Field(default=0.8, ge=0.0, le=1.0)
    min_accuracy_score: float = Field(default=0.7, ge=0.0, le=1.0)
    min_readability_score: float = Field(default=0.75, ge=0.0, le=1.0)

    # LLM-as-judge settings
    use_llm_judge: bool = Field(default=True)
    judge_model: str = Field(default="gpt-4o-mini")

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        case_sensitive=False,
        extra="ignore",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_dir: Path = Field(
        default=Path("outputs/logs"),
        description="Directory for log files",
    )
    log_to_file: bool = Field(default=True)
    log_to_console: bool = Field(default=True)

    # Log formatting
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format string",
    )
    rotation: str = Field(
        default="10 MB",
        description="Log file rotation size",
    )
    retention: str = Field(
        default="1 week",
        description="Log file retention period",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        case_sensitive=False,
        extra="ignore",
    )


class StreamlitSettings(BaseSettings):
    """Streamlit application settings."""

    host: str = Field(default="localhost")
    port: int = Field(default=8501, ge=1024, le=65535)

    # UI Configuration
    page_title: str = Field(default="BMW Sales Analysis")
    page_icon: str = Field(default="🚗")
    layout: Literal["centered", "wide"] = Field(default="wide")

    # Features
    enable_file_upload: bool = Field(default=True)
    enable_llm_selection: bool = Field(default=True)
    enable_advanced_options: bool = Field(default=True)

    # Cache settings
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )

    model_config = SettingsConfigDict(
        env_prefix="STREAMLIT_",
        case_sensitive=False,
        extra="ignore",
    )


class Settings(BaseSettings):
    """Main application settings - aggregates all subsettings."""

    # Application metadata
    app_name: str = Field(default="BMW Sales LLM Analysis")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )

    # Debug mode
    debug: bool = Field(default=False)

    # Subsettings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories on initialization."""
        directories = [
            self.data.data_dir,
            self.data.processed_dir,
            self.data.sample_dir,
            self.visualization.output_dir,
            self.report.output_dir,
            self.evaluation.output_dir,
            self.logging.log_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_llm_config(self) -> dict:
        """Get LLM configuration as dict.

        Returns:
            Dictionary containing LLM configuration.
        """
        return {
            "provider": self.llm.provider,
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "timeout": self.llm.timeout,
        }

    def model_dump_safe(self) -> dict:
        """Dump settings without sensitive information.

        Returns:
            Dictionary of settings with sensitive keys masked.
        """
        data = self.model_dump()

        # Remove sensitive keys
        sensitive_keys = [
            "gemini_api_key",
            "anthropic_api_key",
            "openai_api_key",
        ]

        for key in sensitive_keys:
            if "llm" in data and key in data["llm"]:
                data["llm"][key] = "***" if data["llm"][key] else None

        return data


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment.

    Returns:
        Newly loaded Settings instance.
    """
    global _settings
    _settings = Settings()
    return _settings

