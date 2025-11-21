"""Tests for visualization module."""

import pytest
from pathlib import Path
from src.visualization.plotter import VisualizationEngine


def test_visualization_engine_init(sample_dataframe, tmp_path):
    """Test visualization engine initialization."""
    engine = VisualizationEngine(sample_dataframe, output_dir=tmp_path)
    assert engine.data is not None
    assert engine.output_dir == tmp_path


def test_generate_from_specs(sample_dataframe, tmp_path):
    """Test plot generation from specifications."""
    engine = VisualizationEngine(sample_dataframe, output_dir=tmp_path)

    plot_specs = [
        {
            "plot_type": "line",
            "title": "Sales Over Time",
            "x_axis": "date",
            "y_axis": "sales",
            "description": "Sales trend",
        },
        {
            "plot_type": "bar",
            "title": "Sales by Region",
            "x_axis": "region",
            "y_axis": "sales",
            "description": "Regional comparison",
        },
    ]

    plot_paths = engine.generate_from_specs(plot_specs)

    assert len(plot_paths) == len(plot_specs)
    for path in plot_paths:
        assert Path(path).exists()


def test_unsupported_plot_type(sample_dataframe, tmp_path):
    """Test error handling for unsupported plot types."""
    engine = VisualizationEngine(sample_dataframe, output_dir=tmp_path)

    plot_specs = [
        {
            "plot_type": "invalid_type",
            "title": "Test",
            "x_axis": "date",
            "y_axis": "sales",
            "description": "Test",
        },
    ]

    with pytest.raises(ValueError, match="Unsupported plot type"):
        engine._generate_plot(plot_specs[0], 0)

