"""LangGraph node functions for report generation workflow."""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from langchain.agents import create_agent

try:
    from langchain.output_parsers import OutputFixingParser  # type: ignore[import]
except ImportError:  # pragma: no cover
    OutputFixingParser = None  # type: ignore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from config.settings import get_settings
from src.data.analyzer import StatisticalAnalyzer
from src.data.loader import BMWDataLoader
from src.llm.llm_factory import LLMFactory
from src.llm.prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    EXECUTIVE_SUMMARY_PROMPT_TEMPLATE,
    RECOMMENDATIONS_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_ANALYST,
    SYSTEM_PROMPT_VISUALIZATION,
    SYSTEM_PROMPT_WRITER,
    VISUALIZATION_PLANNING_PROMPT_TEMPLATE,
)
from src.llm.state import ReportState
from src.llm.tools import (
    analyze_relationships,
    check_data_quality,
    detect_date_columns,
    get_categorical_values,
    get_statistical_summary,
    inspect_data_columns,
)
from src.llm.utils import extract_list_items, normalize_markdown, parse_json_response
from src.utils.logger import app_logger


def initialize_llm(
    state: ReportState, settings=None
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Initialize LLM instance with unified error handling.

    Args:
        state: Current workflow state.
        settings: Cached settings instance to avoid redundant lookups.

    Returns:
        Tuple containing the LLM instance (or None) and an error state dict.
    """
    settings = settings or get_settings()
    try:
        return LLMFactory.get_default_llm(), None
    except ConnectionError as exc:
        error_msg = (
            f"LLM connection error: {exc}. "
            f"Please ensure your LLM provider ({settings.llm.provider}) is properly configured and running."
        )
    except Exception as exc:
        error_msg = (
            f"Failed to initialize LLM ({settings.llm.provider}/{settings.llm.model}): {exc}. "
            "Please check your LLM configuration in settings."
        )

    app_logger.error(error_msg)
    return None, {
        "errors": state["errors"] + [error_msg],
        "step_count": state["step_count"] + 1,
    }


def _create_args_schema_without_file_path(tool: StructuredTool) -> type[BaseModel]:
    """Generate args schema that excludes file_path."""

    schema = getattr(tool, "args_schema", None)
    if not schema:
        return create_model(
            f"{tool.name.title().replace(' ', '')}Args", __base__=BaseModel
        )

    fields: Dict[str, tuple[Any, Field]] = {}
    for name, field in schema.model_fields.items():
        if name == "file_path":
            continue

        annotation = field.annotation or Any
        default = field.default if not field.is_required() else ...
        field_info = Field(default=default, description=field.description)
        fields[name] = (annotation, field_info)

    if not fields:
        return create_model(f"{schema.__name__}Bound", __base__=BaseModel)

    return create_model(f"{schema.__name__}Bound", __base__=BaseModel, **fields)


def _bind_tool_to_data_path(tool: StructuredTool, data_path: str) -> StructuredTool:
    """Create a tool variant with file_path pre-populated."""

    args_schema = _create_args_schema_without_file_path(tool)

    def _call(**kwargs):
        return tool.func(file_path=data_path, **kwargs)

    bound_tool = StructuredTool(
        name=tool.name,
        description=f"{tool.description or ''}\n\n(Data path automatically set by workflow.)",
        func=_call,
        coroutine=None,
        args_schema=args_schema,
        return_direct=tool.return_direct,
    )
    bound_tool.tags = tool.tags
    bound_tool.metadata = tool.metadata
    return bound_tool


def _default_insights() -> List[Dict[str, Any]]:
    return [
        {
            "insight": "Analysis completed",
            "evidence": "Data analyzed",
            "impact": "N/A",
            "confidence": 0.5,
        }
    ]


def _parse_structured_output(
    content: str,
    parser: JsonOutputParser,
    llm: Any | None,
    default_factory: Callable[[], Any],
) -> Any:
    """Parse structured LLM output with layered fallbacks."""

    errors: List[str] = []

    try:
        return parse_json_response(content)
    except ValueError as exc:
        errors.append(f"json.loads: {exc}")

    try:
        return parser.parse(content)
    except Exception as exc:  # pragma: no cover - relies on langchain internals
        errors.append(f"JsonOutputParser: {exc}")

    if OutputFixingParser is not None and llm is not None:
        try:
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            return fixing_parser.parse(content)
        except Exception as exc:  # pragma: no cover - depends on provider
            errors.append(f"OutputFixingParser: {exc}")
    else:
        errors.append("OutputFixingParser unavailable")

    app_logger.warning(
        "Structured output parsing failed; using fallback. Details: %s",
        " | ".join(errors[-3:]),
    )
    return default_factory()


def _resolve_column_name(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column from candidate list (case-insensitive)."""

    normalized = {col.lower(): col for col in columns}
    stripped = {col.lower().replace("_", "").replace(" ", ""): col for col in columns}

    for candidate in candidates:
        cand_lower = candidate.lower()
        if cand_lower in normalized:
            return normalized[cand_lower]
        cand_stripped = cand_lower.replace("_", "").replace(" ", "")
        if cand_stripped in stripped:
            return stripped[cand_stripped]
    return None


def _calculate_key_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute dashboard metrics for the report."""

    metrics: Dict[str, Any] = {}

    available_columns = df.columns.tolist()
    sales_col = _resolve_column_name(available_columns, ["sales", "sales_volume"])
    price_col = _resolve_column_name(available_columns, ["price", "price_usd"])
    model_col = _resolve_column_name(available_columns, ["model", "vehicle"])
    region_col = _resolve_column_name(available_columns, ["region", "market"])
    date_col = _resolve_column_name(available_columns, ["date", "year"])

    if sales_col and sales_col in df.columns:
        total_sales = float(df[sales_col].sum())
        metrics["total_sales"] = int(total_sales)

        if date_col and date_col in df.columns:
            year_series = _coerce_year_series(df[date_col])
            if year_series is not None:
                sales_by_year = (
                    df.assign(__year=year_series)
                    .dropna(subset=["__year"])
                    .groupby("__year")[sales_col]
                    .sum()
                    .sort_index()
                )
                if len(sales_by_year) >= 2:
                    last, prev = sales_by_year.iloc[-1], sales_by_year.iloc[-2]
                    if prev != 0:
                        metrics["sales_yoy_pct"] = round((last - prev) / prev * 100, 2)
                        metrics["latest_sales_year"] = int(sales_by_year.index[-1])
                        metrics["previous_sales_year"] = int(sales_by_year.index[-2])

        if model_col and model_col in df.columns:
            model_sales = (
                df.groupby(model_col)[sales_col]
                .sum()
                .sort_values(ascending=False)
                .head(3)
            )
            metrics["top_models"] = [
                {
                    "model": name,
                    "sales": int(value),
                    "share": round(value / total_sales * 100, 2) if total_sales else 0,
                }
                for name, value in model_sales.items()
            ]

        if region_col and region_col in df.columns:
            region_sales = df.groupby(region_col)[sales_col].sum()
            metrics["market_share_by_region"] = {
                region: round(value / total_sales * 100, 2) if total_sales else 0
                for region, value in region_sales.items()
            }

    if price_col and price_col in df.columns:
        avg_price = float(df[price_col].mean())
        metrics["average_price"] = round(avg_price, 2)

        if date_col and date_col in df.columns:
            year_series = _coerce_year_series(df[date_col])
            if year_series is not None:
                price_by_year = (
                    df.assign(__year=year_series)
                    .dropna(subset=["__year"])
                    .groupby("__year")[price_col]
                    .mean()
                    .sort_index()
                )
                if len(price_by_year) >= 2:
                    last, prev = price_by_year.iloc[-1], price_by_year.iloc[-2]
                    if prev != 0:
                        metrics["price_yoy_pct"] = round((last - prev) / prev * 100, 2)

    return metrics


def _coerce_year_series(series: pd.Series) -> Optional[pd.Series]:
    """Convert a column into a numeric year series if possible."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.year

    datetime_series = pd.to_datetime(series, errors="coerce")
    if datetime_series.notna().any():
        return datetime_series.dt.year

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype("Int64")

    return None


def get_tools_with_data_path(data_path: str):
    """Get list of tools bound with data_path for LLM tool calling.

    Args:
        data_path: Path to the data file.

    Returns:
        List of tool instances with file_path preset to data_path.
    """
    base_tools = [
        inspect_data_columns,
        check_data_quality,
        get_statistical_summary,
        get_categorical_values,
        detect_date_columns,
        analyze_relationships,
    ]
    return [_bind_tool_to_data_path(tool, data_path) for tool in base_tools]


def create_agent_with_tools(llm, tools: List, data_path: str):
    """Create a LangChain agent with tools bound.

    Args:
        llm: LLM instance.
        tools: List of tools to bind to the agent.
        data_path: Path to data file (for tool context).

    Returns:
        Compiled agent graph ready for invocation.
    """
    agent = create_agent(llm, tools=tools)
    app_logger.info(f"Created agent with {len(tools)} tools")
    return agent


def invoke_agent(agent, messages: List[HumanMessage | SystemMessage]) -> str:
    """Invoke agent and extract final response content.

    Args:
        agent: Compiled agent graph from create_agent.
        messages: List of messages for the conversation.

    Returns:
        Final response content as string.
    """
    result = agent.invoke({"messages": messages})

    if isinstance(result, dict) and "messages" in result:
        messages_list = result["messages"]
        if messages_list:
            final_message = messages_list[-1]

            if hasattr(final_message, "content"):
                content = final_message.content
                if content:
                    return content if isinstance(content, str) else str(content)
            elif isinstance(final_message, dict):
                content = final_message.get("content", "")
                if content:
                    return content if isinstance(content, str) else str(content)

            for msg in reversed(messages_list):
                msg_type = (
                    type(msg).__name__ if hasattr(msg, "__class__") else str(type(msg))
                )
                if "AIMessage" in msg_type or (
                    isinstance(msg, dict) and msg.get("type") == "ai"
                ):
                    content = getattr(msg, "content", None) or (
                        msg.get("content") if isinstance(msg, dict) else None
                    )
                    if content:
                        return content if isinstance(content, str) else str(content)

    if isinstance(result, dict):
        if "output" in result:
            return (
                result["output"]
                if isinstance(result["output"], str)
                else str(result["output"])
            )
        if "structured_response" in result:
            return str(result["structured_response"])

    app_logger.warning(
        "Could not extract content from agent response, converting to string"
    )
    return str(result)


def _invoke_markdown_section(agent, prompt: str, section_label: str) -> str:
    """Generate and normalize markdown section content."""

    content = invoke_agent(agent, [HumanMessage(content=prompt)])
    normalized = normalize_markdown(content)

    if normalized:
        return normalized

    return f"_Section '{section_label}' could not be generated automatically. Please review the inputs and rerun this step._"


def load_and_profile_node(state: ReportState) -> Dict[str, Any]:
    """Node: Load data and generate profile using introspection tools.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with data, profile, and tool results.
    """
    try:
        app_logger.info(f"Loading data from {state['data_path']}")
        data_path = state["data_path"]

        loader = BMWDataLoader(data_path)
        df = loader.load()

        analyzer = StatisticalAnalyzer(df)
        profile = analyzer.generate_profile()
        summary_stats = analyzer.get_summary_stats()

        app_logger.info("Running data introspection tools...")

        key_metrics = _calculate_key_metrics(df)

        column_info = inspect_data_columns.invoke({"file_path": data_path})
        app_logger.info(f"Tool: Found {column_info['total_columns']} columns")

        quality_report = check_data_quality.invoke({"file_path": data_path})
        app_logger.info(
            f"Tool: Quality score = {quality_report['quality_summary']['completeness_score']}%"
        )

        stats_summary = get_statistical_summary.invoke({"file_path": data_path})
        app_logger.info(
            f"Tool: Analyzed {len(stats_summary.get('numeric_columns', []))} numeric columns"
        )

        categorical_info = get_categorical_values.invoke({"file_path": data_path})
        app_logger.info(
            f"Tool: Found {len(categorical_info.get('categorical_columns', []))} categorical columns"
        )

        date_info = detect_date_columns.invoke({"file_path": data_path})
        app_logger.info(
            f"Tool: Found {date_info.get('total_date_columns', 0)} date columns"
        )

        relationship_info = analyze_relationships.invoke({"file_path": data_path})
        app_logger.info(
            f"Tool: Identified {len(relationship_info.get('potential_grouping_keys', []))} grouping keys"
        )

        # Combine tool results for enhanced profiling
        enhanced_profile = {
            "traditional_profile": profile,
            "column_info": column_info,
            "quality_report": quality_report,
            "statistical_summary": stats_summary,
            "categorical_info": categorical_info,
            "date_info": date_info,
            "relationship_info": relationship_info,
        }

        app_logger.info("Data loaded and profiled successfully with tool augmentation")

        return {
            "data": df,
            "data_profile": enhanced_profile,  # Now includes tool results
            "statistical_summary": summary_stats,
            "key_metrics": key_metrics,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Data loading error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def analyze_data_node(state: ReportState) -> Dict[str, Any]:
    """Node: Analyze data and extract insights using LLM.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with insights and findings.
    """
    llm, error_state = initialize_llm(state)
    if error_state:
        return error_state

    try:
        app_logger.info("Starting LLM analysis with tool-augmented data")

        enhanced_profile = state["data_profile"]

        tool_context = f"""
DATA STRUCTURE (from Column Inspector Tool):
Columns: {enhanced_profile["column_info"]["columns"]}
Data Types: {json.dumps(enhanced_profile["column_info"]["dtypes"], indent=2)}

DATA QUALITY (from Quality Checker Tool):
Total Records: {enhanced_profile["quality_report"]["total_rows"]}
Missing Values: {json.dumps(enhanced_profile["quality_report"]["missing_percentage"], indent=2)}
Completeness Score: {enhanced_profile["quality_report"]["quality_summary"]["completeness_score"]}%
Potential Outliers: {json.dumps(enhanced_profile["quality_report"]["potential_outliers"], indent=2)}

STATISTICAL SUMMARY (from Statistical Tool):
{json.dumps(enhanced_profile["statistical_summary"]["statistics"], indent=2)}

CATEGORICAL VALUES (from Categorical Inspector Tool):
{json.dumps(enhanced_profile["categorical_info"].get("column_details", {}), indent=2)}

DATE ANALYSIS (from Date Detector Tool):
{json.dumps(enhanced_profile["date_info"].get("date_column_details", {}), indent=2)}

RELATIONSHIPS (from Relationship Analyzer Tool):
Grouping Keys: {json.dumps(enhanced_profile["relationship_info"]["potential_grouping_keys"], indent=2)}
Aggregation Suggestions: {json.dumps(enhanced_profile["relationship_info"]["aggregation_suggestions"][:5], indent=2)}
Strong Correlations: {json.dumps(enhanced_profile["statistical_summary"].get("strong_correlations", []), indent=2)}
"""

        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(data_profile=tool_context)

        tools = get_tools_with_data_path(state["data_path"])

        agent = create_agent_with_tools(llm, tools, state["data_path"])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT_ANALYST),
            HumanMessage(content=user_prompt),
        ]

        content = invoke_agent(agent, messages)

        parser = JsonOutputParser()
        insights_raw = _parse_structured_output(
            content, parser, llm, default_factory=_default_insights
        )
        if isinstance(insights_raw, list):
            insights = insights_raw
        elif insights_raw:
            insights = [insights_raw]
        else:
            insights = _default_insights()

        key_findings = [insight.get("insight", "") for insight in insights[:5]]

        app_logger.info(f"Analysis complete: {len(insights)} insights extracted")

        return {
            "insights": insights,
            "key_findings": key_findings,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def plan_visualizations_node(state: ReportState) -> Dict[str, Any]:
    """Node: Plan visualizations based on insights.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with plot specifications.
    """
    settings = get_settings()
    llm, error_state = initialize_llm(state, settings)
    if error_state:
        return error_state

    try:
        app_logger.info("Planning visualizations with tool-validated columns")

        enhanced_profile = state["data_profile"]
        column_info = enhanced_profile["column_info"]

        column_context = f"""
AVAILABLE COLUMNS (validated by tools):
{json.dumps(column_info["columns"], indent=2)}

COLUMN DATA TYPES:
{json.dumps(column_info["dtypes"], indent=2)}

IMPORTANT: Use ONLY the column names listed above. Do not assume column names.
You can use the available tools to verify column names and data types if needed.
"""

        parser = JsonOutputParser()
        format_instructions = parser.get_format_instructions()
        user_prompt = (
            VISUALIZATION_PLANNING_PROMPT_TEMPLATE.format(
                insights=str(state["insights"]),
                statistical_summary=column_context
                + "\n\n"
                + str(state["statistical_summary"]),
                min_plots=settings.visualization.min_plots,
                max_plots=settings.visualization.max_plots,
            )
            + "\n\nReturn ONLY valid JSON per the following instructions:\n"
            + format_instructions
        )

        tools = get_tools_with_data_path(state["data_path"])

        agent = create_agent_with_tools(llm, tools, state["data_path"])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT_VISUALIZATION),
            HumanMessage(content=user_prompt),
        ]

        content = invoke_agent(agent, messages)

        plot_specs_raw = _parse_structured_output(
            content, parser, llm, default_factory=list
        )

        if isinstance(plot_specs_raw, dict):
            plot_specs = [plot_specs_raw]
        elif isinstance(plot_specs_raw, list):
            plot_specs = [spec for spec in plot_specs_raw if isinstance(spec, dict)]
            invalid_count = len(plot_specs_raw) - len(plot_specs)
            if invalid_count > 0:
                app_logger.warning(
                    "Dropped %d invalid plot specs entries (non-dict).", invalid_count
                )
        else:
            app_logger.warning(
                "Visualization planner returned unsupported structure (%s); ignoring.",
                type(plot_specs_raw),
            )
            plot_specs = []

        if isinstance(plot_specs, dict):
            plot_specs = [plot_specs]
        elif not isinstance(plot_specs, list):
            app_logger.warning(
                f"Visualization planner returned unsupported structure ({type(plot_specs)}); ignoring."
            )
            plot_specs = []

        cleaned_specs = []
        for spec in plot_specs:
            if isinstance(spec, dict):
                cleaned_specs.append(spec)
            else:
                app_logger.warning(
                    f"Dropping invalid plot specification of type {type(spec)}"
                )
        plot_specs = cleaned_specs

        if not plot_specs:
            plot_specs = _build_default_plot_specs(state, settings)

        app_logger.info(
            f"Visualization planning complete: {len(plot_specs)} plots planned"
        )

        return {
            "plot_specs": plot_specs,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Visualization planning error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def generate_plots_node(state: ReportState) -> Dict[str, Any]:
    """Node: Generate actual plot files.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with plot paths.
    """
    settings = get_settings()

    try:
        app_logger.info("Generating plots")

        if not state.get("plot_specs"):
            error_msg = "No plot specifications available. Visualization planning may have failed."
            app_logger.warning(error_msg)
            return {
                "errors": state["errors"] + [error_msg],
                "plot_paths": [],
                "step_count": state["step_count"] + 1,
            }

        from src.visualization.plotter import VisualizationEngine

        viz_engine = VisualizationEngine(
            data=state["data"],
            output_dir=settings.visualization.output_dir,
        )

        plot_paths = viz_engine.generate_from_specs(state["plot_specs"])

        app_logger.info(f"Plots generated: {len(plot_paths)} files")

        return {
            "plot_paths": plot_paths,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Plot generation error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def _build_default_plot_specs(state: ReportState, settings) -> List[Dict[str, Any]]:
    """Generate fallback visualization specs when LLM output is unusable."""

    enhanced_profile = state.get("data_profile", {}) or {}
    column_info = enhanced_profile.get("column_info", {}) or {}
    columns: List[str] = column_info.get("columns", []) or []

    date_col = _resolve_column_name(columns, ["date", "year", "datetime"])
    sales_col = _resolve_column_name(columns, ["sales", "sales_volume", "volume"])
    model_col = _resolve_column_name(columns, ["model", "vehicle", "product"])
    region_col = _resolve_column_name(columns, ["region", "market", "location"])
    price_col = _resolve_column_name(columns, ["price", "price_usd", "avg_price"])

    specs: List[Dict[str, Any]] = []

    if date_col and sales_col:
        specs.append(
            {
                "plot_type": "line",
                "title": "Sales Trend Over Time",
                "x_axis": date_col,
                "y_axis": sales_col,
                "aggregation": "sum",
                "description": "Shows total sales aggregated by date.",
            }
        )

    if model_col and sales_col:
        specs.append(
            {
                "plot_type": "bar",
                "title": "Top Models by Sales",
                "x_axis": model_col,
                "y_axis": sales_col,
                "aggregation": "sum",
                "sort": "desc",
                "limit": min(settings.visualization.max_plots, 10),
                "description": "Highlights the leading models ranked by sales volume.",
            }
        )

    if region_col and price_col:
        specs.append(
            {
                "plot_type": "box",
                "title": "Price Distribution by Region",
                "x_axis": region_col,
                "y_axis": price_col,
                "aggregation": "none",
                "description": "Compares price spread across regions.",
            }
        )

    if not specs and columns:
        fallback_y = sales_col or (columns[1] if len(columns) > 1 else columns[0])
        specs.append(
            {
                "plot_type": "bar",
                "title": "Dataset Overview",
                "x_axis": columns[0],
                "y_axis": fallback_y,
                "aggregation": "count" if fallback_y == columns[0] else "sum",
                "description": "Automatically generated fallback visualization.",
            }
        )

    app_logger.info(
        f"Using fallback visualization specs ({len(specs)} generated automatically)"
    )
    return specs


def write_executive_summary_node(state: ReportState) -> Dict[str, Any]:
    """Node: Write executive summary.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with executive summary.
    """
    llm, error_state = initialize_llm(state)
    if error_state:
        return error_state

    try:
        app_logger.info("Writing executive summary")

        user_prompt = EXECUTIVE_SUMMARY_PROMPT_TEMPLATE.format(
            key_findings=str(state["key_findings"]),
            insights=str(state["insights"][:3] if state["insights"] else []),
        )

        tools = get_tools_with_data_path(state["data_path"])

        agent = create_agent_with_tools(llm, tools, state["data_path"])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT_WRITER),
            HumanMessage(content=user_prompt),
        ]

        content = invoke_agent(agent, messages)
        summary = normalize_markdown(content)
        if not summary:
            summary = (
                "Executive summary could not be generated automatically. "
                "Please re-run the workflow with the latest data."
            )

        app_logger.info("Executive summary written")

        return {
            "executive_summary": summary,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Executive summary error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def write_analysis_sections_node(state: ReportState) -> Dict[str, Any]:
    """Node: Write detailed analysis sections.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with analysis sections.
    """
    llm, error_state = initialize_llm(state)
    if error_state:
        return error_state

    sections = {}

    try:
        app_logger.info("Writing analysis sections")

        tools = get_tools_with_data_path(state["data_path"])

        agent = create_agent_with_tools(llm, tools, state["data_path"])

        plot_titles = (
            [spec.get("title", "") for spec in state["plot_specs"]]
            if state.get("plot_specs")
            else []
        )

        section_configs = {
            "sales_trends": (
                "Sales Performance Trends",
                f"""Write a detailed "Sales Performance Trends" section based on:

Insights: {state["insights"]}
Plots: {plot_titles}

Include:
- Temporal trends
- Growth patterns
- Notable changes

Use markdown format with subsections. Reference the plots.
You can use the available tools to verify data claims if needed.""",
            ),
            "regional_analysis": (
                "Regional Performance Analysis",
                f"""Write a "Regional Performance Analysis" section based on:

Insights: {state["insights"]}
Data Profile: {state.get("data_profile", {})}

Include:
- Regional comparisons
- Top performing regions
- Regional growth patterns

Use markdown format.
You can use the available tools to verify categorical values and relationships.""",
            ),
            "product_performance": (
                "Product Performance Analysis",
                f"""Write a "Product Performance Analysis" section based on:

Insights: {state["insights"]}

Include:
- Top and bottom performing models
- Model trends
- Price-performance analysis

Use markdown format.
You can use the available tools to verify statistical claims.""",
            ),
        }

        for section_key, (title, prompt) in section_configs.items():
            sections[section_key] = _invoke_markdown_section(agent, prompt, title)

        app_logger.info(f"Analysis sections written: {len(sections)} sections")

        return {
            "analysis_sections": sections,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Analysis sections error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def generate_recommendations_node(state: ReportState) -> Dict[str, Any]:
    """Node: Generate actionable recommendations.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with recommendations.
    """
    llm, error_state = initialize_llm(state)
    if error_state:
        return error_state

    try:
        app_logger.info("Generating recommendations")

        prompt = RECOMMENDATIONS_PROMPT_TEMPLATE.format(
            insights=str(state["insights"]),
            key_findings=str(state["key_findings"]),
        )

        tools = get_tools_with_data_path(state["data_path"])

        agent = create_agent_with_tools(llm, tools, state["data_path"])

        content = invoke_agent(agent, [HumanMessage(content=prompt)])
        recommendations = extract_list_items(content, max_items=10)

        if not recommendations:
            key_findings = state.get("key_findings") or []
            if key_findings:
                recommendations = [
                    f"Investigate '{finding}' to unlock further growth opportunities."
                    for finding in key_findings[:3]
                ]
            else:
                recommendations = [
                    "No recommendations were generated automatically. Please review the analysis results."
                ]

        app_logger.info(f"Recommendations generated: {len(recommendations)} items")

        return {
            "recommendations": recommendations,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Recommendations error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def compile_report_node(state: ReportState) -> Dict[str, Any]:
    """Node: Compile final report in multiple formats.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with report paths.
    """
    settings = get_settings()

    try:
        app_logger.info("Compiling report")

        from src.reporting.compiler import ReportCompiler

        compiler = ReportCompiler(settings=settings, state=state)

        report_paths = compiler.generate_all_formats()

        app_logger.info(f"Report compiled: {list(report_paths.keys())}")

        return {
            "report_paths": report_paths,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Report compilation error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }


def evaluate_report_node(state: ReportState) -> Dict[str, Any]:
    """Node: Evaluate report quality.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with evaluation results.
    """
    from src.evaluation.evaluator import ReportEvaluator

    settings = get_settings()

    if not settings.evaluation.enabled:
        return {"step_count": state["step_count"] + 1}

    try:
        app_logger.info("Evaluating report quality")

        evaluator = ReportEvaluator(settings=settings)

        report_paths = state.get("report_paths", {})
        if isinstance(report_paths, dict):
            markdown_path = report_paths.get("markdown")
        else:
            markdown_path = state.get("report_markdown")

        results = evaluator.evaluate(
            report_markdown=markdown_path,
            data=state.get("data"),
            insights=state.get("insights", []),
        )

        app_logger.info(f"Evaluation complete: {results}")

        return {
            "evaluation_results": results,
            "step_count": state["step_count"] + 1,
        }
    except Exception as e:
        error_msg = f"Evaluation error: {str(e)}"
        app_logger.error(error_msg)
        return {
            "errors": state["errors"] + [error_msg],
            "step_count": state["step_count"] + 1,
        }
