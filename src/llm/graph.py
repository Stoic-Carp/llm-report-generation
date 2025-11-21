"""LangGraph workflow definition for report generation.

This module defines the main workflow graph that orchestrates report generation.
The workflow integrates LangChain agents (via langchain.agents.create_agent) at
key stages to enable dynamic tool calling for data validation and analysis.
"""

from langgraph.graph import END, StateGraph

from src.llm.nodes import (
    analyze_data_node,
    compile_report_node,
    evaluate_report_node,
    generate_plots_node,
    generate_recommendations_node,
    load_and_profile_node,
    plan_visualizations_node,
    write_analysis_sections_node,
    write_executive_summary_node,
)
from src.llm.state import ReportState
from src.utils.logger import app_logger


def should_continue(state: ReportState) -> str:
    """Conditional edge: check if there are errors.

    Args:
        state: Current workflow state.

    Returns:
        "end" if errors exist, "continue" otherwise.
    """
    if state.get("errors") and len(state["errors"]) > 0:
        return "end"
    return "continue"


def create_report_workflow() -> StateGraph:
    """Create the LangGraph workflow for report generation.

    Returns:
        Compiled StateGraph workflow.
    """
    app_logger.info("Creating report generation workflow")

    workflow = StateGraph(ReportState)

    workflow.add_node("load_and_profile", load_and_profile_node)
    workflow.add_node("analyze", analyze_data_node)
    workflow.add_node("plan_viz", plan_visualizations_node)
    workflow.add_node("generate_plots", generate_plots_node)
    workflow.add_node("write_summary", write_executive_summary_node)
    workflow.add_node("write_analysis", write_analysis_sections_node)
    workflow.add_node("write_recommendations", generate_recommendations_node)
    workflow.add_node("compile", compile_report_node)
    workflow.add_node("evaluate", evaluate_report_node)

    workflow.set_entry_point("load_and_profile")

    workflow.add_edge("load_and_profile", "analyze")
    workflow.add_edge("analyze", "plan_viz")
    workflow.add_edge("plan_viz", "generate_plots")
    workflow.add_edge("generate_plots", "write_summary")
    workflow.add_edge("write_summary", "write_analysis")
    workflow.add_edge("write_analysis", "write_recommendations")
    workflow.add_edge("write_recommendations", "compile")
    workflow.add_edge("compile", "evaluate")
    workflow.add_edge("evaluate", END)

    app_logger.info("Workflow created successfully")

    return workflow.compile()


class ReportGenerator:
    """Main class for report generation using LangGraph."""

    def __init__(self):
        """Initialize report generator with workflow."""
        self.workflow = create_report_workflow()
        self.logger = app_logger

    def generate_report(self, data_path: str) -> dict:
        """Execute the workflow synchronously.

        Args:
            data_path: Path to the data file.

        Returns:
            Final state dictionary.
        """
        self.logger.info(f"Starting report generation for {data_path}")

        initial_state: ReportState = {
            "data_path": data_path,
            "data": None,
            "data_profile": None,
            "statistical_summary": None,
            "insights": None,
            "trends": None,
            "key_findings": None,
            "key_metrics": None,
            "plot_specs": None,
            "plot_paths": None,
            "executive_summary": None,
            "analysis_sections": None,
            "recommendations": None,
            "report_markdown": None,
            "report_paths": None,
            "evaluation_results": None,
            "errors": [],
            "step_count": 0,
        }

        final_state = self.workflow.invoke(initial_state)

        self.logger.info(
            f"Report generation complete. Steps: {final_state.get('step_count', 0)}"
        )

        return final_state

    async def generate_report_async(self, data_path: str):
        """Execute workflow asynchronously with streaming.

        Args:
            data_path: Path to the data file.

        Yields:
            State updates as workflow progresses.
        """
        self.logger.info(f"Starting async report generation for {data_path}")

        initial_state: ReportState = {
            "data_path": data_path,
            "data": None,
            "data_profile": None,
            "statistical_summary": None,
            "insights": None,
            "trends": None,
            "key_findings": None,
            "key_metrics": None,
            "plot_specs": None,
            "plot_paths": None,
            "executive_summary": None,
            "analysis_sections": None,
            "recommendations": None,
            "report_markdown": None,
            "report_paths": None,
            "evaluation_results": None,
            "errors": [],
            "step_count": 0,
        }

        async for state in self.workflow.astream(initial_state):
            yield state
