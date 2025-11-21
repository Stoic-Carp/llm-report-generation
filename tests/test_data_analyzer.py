"""Tests for data analyzer module."""

import pytest
from src.data.analyzer import StatisticalAnalyzer


def test_generate_profile(sample_dataframe):
    """Test profile generation."""
    analyzer = StatisticalAnalyzer(sample_dataframe)
    profile = analyzer.generate_profile()

    assert "basic_stats" in profile
    assert "temporal_trends" in profile
    assert "regional_breakdown" in profile
    assert "model_performance" in profile
    assert "correlations" in profile
    assert "summary_insights" in profile


def test_basic_statistics(sample_dataframe):
    """Test basic statistics calculation."""
    analyzer = StatisticalAnalyzer(sample_dataframe)
    stats = analyzer.get_summary_stats()

    assert "total_records" in stats
    assert stats["total_records"] == len(sample_dataframe)
    assert "total_sales" in stats
    assert stats["total_sales"] > 0


def test_temporal_analysis(sample_dataframe):
    """Test temporal analysis."""
    analyzer = StatisticalAnalyzer(sample_dataframe)
    profile = analyzer.generate_profile()

    temporal = profile["temporal_trends"]
    assert "yearly_sales" in temporal
    assert "growth_rates" in temporal


def test_regional_analysis(sample_dataframe):
    """Test regional analysis."""
    analyzer = StatisticalAnalyzer(sample_dataframe)
    profile = analyzer.generate_profile()

    regional = profile["regional_breakdown"]
    assert "regional_stats" in regional or len(regional) == 0  # May be empty if no region column


def test_model_analysis(sample_dataframe):
    """Test model analysis."""
    analyzer = StatisticalAnalyzer(sample_dataframe)
    profile = analyzer.generate_profile()

    model_perf = profile["model_performance"]
    assert "model_stats" in model_perf or len(model_perf) == 0

