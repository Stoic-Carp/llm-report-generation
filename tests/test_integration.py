"""Integration tests for end-to-end workflow."""

import pytest
from pathlib import Path
from src.data.loader import BMWDataLoader
from src.data.analyzer import StatisticalAnalyzer


def test_data_loading_and_analysis(sample_data_path):
    """Test complete data loading and analysis pipeline."""
    # Load data
    loader = BMWDataLoader(sample_data_path)
    df = loader.load()

    assert len(df) > 0

    # Analyze
    analyzer = StatisticalAnalyzer(df)
    profile = analyzer.generate_profile()

    assert profile is not None
    assert "basic_stats" in profile


def test_report_generation_workflow(sample_data_path, test_settings):
    """Test report generation workflow (without LLM calls)."""
    # This test would require mocking LLM calls
    # For now, we test the data pipeline part

    loader = BMWDataLoader(sample_data_path)
    df = loader.load()

    analyzer = StatisticalAnalyzer(df)
    profile = analyzer.generate_profile()

    # Verify profile structure
    assert "basic_stats" in profile
    assert "temporal_trends" in profile

