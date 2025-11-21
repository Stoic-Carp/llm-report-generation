"""LLM factory for creating LLM instances from multiple providers."""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from src.utils.logger import app_logger


class LLMFactory:
    """Factory for creating LLM instances using pydantic-settings."""

    @staticmethod
    def create_llm(
        provider: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> BaseLanguageModel:
        """Create LLM instance with settings from config.

        Args:
            provider: LLM provider name. If None, uses settings default.
            model: Model name. If None, uses settings default.
            **kwargs: Additional LLM parameters to override settings.

        Returns:
            Initialized LLM instance.

        Raises:
            ValueError: If provider is unsupported or API key missing.
        """
        settings = get_settings()

        # Use provided values or fall back to settings
        provider = provider or settings.llm.provider
        model = model or settings.llm.model

        temperature = kwargs.get("temperature", settings.llm.temperature)
        timeout = kwargs.get("timeout", settings.llm.timeout)
        max_tokens = kwargs.get("max_tokens", settings.llm.max_tokens)

        app_logger.info(f"Creating LLM: {provider}/{model}")

        if provider == "ollama":
            try:
                import requests

                # Test Ollama connection
                test_url = f"{settings.llm.ollama_host}/api/tags"
                response = requests.get(test_url, timeout=5)
                if response.status_code != 200:
                    raise ConnectionError(
                        f"Ollama server at {settings.llm.ollama_host} is not responding correctly. "
                        f"Status code: {response.status_code}. "
                        f"Please ensure Ollama is running: 'ollama serve' or check your OLLAMA_HOST setting."
                    )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to Ollama server at {settings.llm.ollama_host}. "
                    f"Please ensure Ollama is running. "
                    f"Start Ollama with: 'ollama serve' or check your network connection."
                )
            except requests.exceptions.Timeout:
                raise ConnectionError(
                    f"Connection to Ollama server at {settings.llm.ollama_host} timed out. "
                    f"Please check if Ollama is running and accessible."
                )
            except ImportError:
                # requests not available, skip connection test
                app_logger.warning(
                    "requests library not available, skipping Ollama connection test"
                )
            except Exception as e:
                app_logger.warning(f"Could not test Ollama connection: {e}")

            return ChatOllama(
                model=model,
                base_url=settings.llm.ollama_host,
                temperature=temperature,
            )
        elif provider == "gemini":
            if not settings.llm.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.llm.gemini_api_key,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        elif provider == "anthropic":
            if not settings.llm.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            return ChatAnthropic(
                model=model,
                anthropic_api_key=settings.llm.anthropic_api_key,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        elif provider == "openai":
            if not settings.llm.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            return ChatOpenAI(
                model=model,
                openai_api_key=settings.llm.openai_api_key,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_default_llm() -> BaseLanguageModel:
        """Get LLM with default settings.

        Returns:
            Default LLM instance.
        """
        return LLMFactory.create_llm()
