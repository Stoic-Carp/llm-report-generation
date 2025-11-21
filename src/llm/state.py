"""LangGraph state schema definitions."""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ReportState(TypedDict):
    """State schema for the report generation workflow."""

    # Input
    data_path: str
    data: Optional[Any]  # DataFrame

    # Data profiling
    data_profile: Optional[Dict[str, Any]]
    statistical_summary: Optional[Dict[str, Any]]

    # Analysis
    insights: Optional[List[Dict[str, str]]]
    trends: Optional[Dict[str, Any]]
    key_findings: Optional[List[str]]
    key_metrics: Optional[Dict[str, Any]]

    # Visualization
    plot_specs: Optional[List[Dict[str, Any]]]
    plot_paths: Optional[List[str]]

    # Report generation
    executive_summary: Optional[str]
    analysis_sections: Optional[Dict[str, str]]
    recommendations: Optional[List[str]]

    # Output
    report_markdown: Optional[str]
    report_paths: Optional[Dict[str, str]]

    # Evaluation
    evaluation_results: Optional[Dict[str, float]]

    # Metadata
    errors: List[str]
    step_count: int


class PlotSpec(BaseModel):
    """Schema for plot specification."""

    plot_type: str = Field(description="Type of plot: line, bar, scatter, heatmap, etc.")
    title: str
    x_axis: str
    y_axis: str
    groupby: Optional[str] = None
    color: Optional[str] = None
    description: str


class Insight(BaseModel):
    """Schema for business insight."""

    insight: str
    evidence: str
    impact: str
    confidence: float = Field(ge=0.0, le=1.0)

