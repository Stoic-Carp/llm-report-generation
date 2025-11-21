"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from config.settings import Settings


@pytest.fixture
def test_settings():
    """Provide test settings."""
    return Settings(
        environment="development",  # Use valid environment value
        debug=True,
        llm={"provider": "ollama", "model": "granite4:latest"},
        data={"data_dir": "tests/test_data", "handle_missing": "drop"},
        logging={"level": "DEBUG", "log_to_console": False, "log_to_file": False},
    )


@pytest.fixture
def sample_data_path(tmp_path):
    """Create sample data file for testing."""
    # Create sample DataFrame
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="ME")  # ME = Month End
    regions = ["NA", "EU", "ASIA", "LATAM", "MEA"]
    models = ["X1", "X3", "X5", "3 Series", "5 Series"]

    data = []
    for date in dates:
        for region in regions:
            for model in models:
                data.append({
                    "date": date,
                    "region": region,
                    "model": model,
                    "sales": np.random.randint(100, 1000),
                    "price": np.random.uniform(30000, 80000),
                })

    df = pd.DataFrame(data)
    file_path = tmp_path / "sample_sales.xlsx"
    df.to_excel(file_path, index=False)

    return file_path


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="ME")  # ME = Month End
    regions = ["NA", "EU", "ASIA"]
    models = ["X1", "X3", "X5"]

    data = []
    for date in dates[:12]:  # Smaller dataset for faster tests
        for region in regions:
            for model in models:
                data.append({
                    "date": date,
                    "region": region,
                    "model": model,
                    "sales": np.random.randint(100, 1000),
                    "price": np.random.uniform(30000, 80000),
                })

    return pd.DataFrame(data)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": '[{"insight": "Sales increased over time", "evidence": "Data shows 20% growth", "impact": "Positive trend", "confidence": 0.85}]',
    }

