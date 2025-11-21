"""Main Streamlit application."""

import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.data.loader import BMWDataLoader
from src.llm.graph import ReportGenerator
from src.utils.logger import app_logger

# Page configuration
st.set_page_config(
    page_title="BMW Sales Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize settings
settings = get_settings()

# Session defaults
SESSION_DEFAULTS = {
    "data": None,
    "data_path": None,
    "analysis_result": None,
    "analysis_error": None,
    "auto_run_pending": False,
    "current_upload_signature": None,
    "processed_upload_signature": None,
}

for key, value in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_runtime_settings():
    """Apply sidebar selections to runtime settings."""

    settings.llm.provider = st.session_state.get("llm_provider", settings.llm.provider)
    settings.llm.model = st.session_state.get("llm_model", settings.llm.model)
    settings.llm.temperature = st.session_state.get(
        "temperature", settings.llm.temperature
    )
    settings.llm.max_tokens = st.session_state.get(
        "max_tokens", settings.llm.max_tokens
    )

    settings.visualization.max_plots = st.session_state.get(
        "num_plots", settings.visualization.max_plots
    )
    settings.visualization.format = st.session_state.get(
        "plot_format", settings.visualization.format
    )

    selected_formats = set(st.session_state.get("report_formats") or [])
    format_flags = {
        "PDF": "generate_pdf",
        "HTML": "generate_html",
        "Word": "generate_word",
        "Markdown": "generate_markdown",
    }
    for fmt, attr in format_flags.items():
        setattr(settings.report, attr, fmt in selected_formats)

    settings.evaluation.enabled = st.session_state.get(
        "include_evaluation", settings.evaluation.enabled
    )


def run_analysis_workflow(auto_trigger: bool = False) -> bool:
    """Execute the LangGraph workflow and persist results."""

    data_path = st.session_state.get("data_path")
    if not data_path:
        st.warning("⚠️ Please upload data first.")
        st.session_state["auto_run_pending"] = False
        return False

    apply_runtime_settings()

    spinner_text = (
        "Running full analysis automatically..."
        if auto_trigger
        else "Running analysis with the latest settings..."
    )

    with st.spinner(spinner_text):
        try:
            generator = ReportGenerator()
            result = generator.generate_report(data_path)

            st.session_state["analysis_result"] = result
            st.session_state["analysis_error"] = None
            st.session_state["auto_run_pending"] = False
            st.session_state["last_run_path"] = data_path
            st.session_state["processed_upload_signature"] = st.session_state.get(
                "current_upload_signature"
            )

            success_msg = (
                "✅ Analysis complete! Review the outputs in the tabs below."
                if auto_trigger
                else "✅ Analysis complete!"
            )
            st.success(success_msg)
            return True
        except ConnectionError as e:
            error_msg = str(e)
            st.session_state["analysis_error"] = error_msg
            st.session_state["auto_run_pending"] = False
            st.error(
                f"🔌 **LLM Connection Error**: {error_msg}\n\n"
                f"**Troubleshooting Steps:**\n"
                f"1. If using Ollama, ensure it's running: `ollama serve`\n"
                f"2. Check your LLM provider settings in the sidebar\n"
                f"3. Verify network connectivity\n"
                f"4. Check if the model is available: `ollama list` (for Ollama)"
            )
            app_logger.error(f"LLM connection error: {e}")
            return False
        except Exception as e:
            error_msg = str(e)
            st.session_state["analysis_error"] = error_msg
            st.session_state["auto_run_pending"] = False

            if "LLM" in error_msg or "connection" in error_msg.lower():
                st.error(
                    f"🤖 **LLM Error**: {error_msg}\n\n"
                    f"Please check your LLM configuration and ensure the service is running."
                )
            else:
                st.error(f"❌ **Analysis Error**: {error_msg}")

            app_logger.error(f"Analysis error: {e}")
            return False


def render_sidebar():
    """Render configuration sidebar."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Settings
        st.subheader("🤖 LLM Settings")

        llm_provider = st.selectbox(
            "Provider",
            ["ollama", "gemini", "anthropic", "openai"],
            index=["ollama", "gemini", "anthropic", "openai"].index(
                settings.llm.provider
            ),
        )

        # Model selection based on provider
        model_options = {
            "ollama": [
                "granite4:latest",
                "llama3.1:8b",
                "llama3.1:70b",
                "mistral:7b",
                "granite3.1:8b",
            ],
            "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        }

        llm_model = st.selectbox(
            "Model",
            model_options[llm_provider],
            index=0,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=settings.llm.temperature,
            step=0.1,
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=512,
            max_value=32000,
            value=settings.llm.max_tokens,
            step=512,
        )

        # Visualization Settings
        st.subheader("📊 Visualization")

        num_plots = st.slider(
            "Number of Plots",
            min_value=settings.visualization.min_plots,
            max_value=settings.visualization.max_plots,
            value=(settings.visualization.min_plots + settings.visualization.max_plots)
            // 2,
        )

        plot_format = st.selectbox(
            "Plot Format",
            ["png", "jpg", "svg"],
            index=0,
        )

        # Report Settings
        st.subheader("📄 Report")

        report_formats = st.multiselect(
            "Output Formats",
            ["PDF", "HTML", "Word", "Markdown"],
            default=[
                fmt
                for fmt, enabled in {
                    "PDF": settings.report.generate_pdf,
                    "HTML": settings.report.generate_html,
                    "Word": settings.report.generate_word,
                    "Markdown": settings.report.generate_markdown,
                }.items()
                if enabled
            ],
        )

        include_evaluation = st.checkbox(
            "Include Evaluation",
            value=settings.evaluation.enabled,
        )

        # Store settings in session state
        st.session_state["llm_provider"] = llm_provider
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["max_tokens"] = max_tokens
        st.session_state["num_plots"] = num_plots
        st.session_state["plot_format"] = plot_format
        st.session_state["report_formats"] = report_formats
        st.session_state["include_evaluation"] = include_evaluation

        st.markdown("---")
        st.caption(f"Environment: {settings.environment}")
        st.caption(f"Debug: {settings.debug}")


def main():
    """Main application function."""
    st.title("🚗 BMW Sales Analysis - LLM-Powered Reporting")
    st.markdown("---")

    # Render sidebar
    render_sidebar()

    st.header("1️⃣ Upload & Preview Data")
    uploaded_file = st.file_uploader(
        "Choose Excel file",
        type=["xlsx", "xls"],
        help="Upload the BMW sales dataset",
    )

    if uploaded_file:
        # Use a writable temporary directory for uploads
        # In Docker, data/raw is read-only, so we use /tmp/uploads
        upload_dir = Path("/tmp/uploads")
        upload_dir.mkdir(exist_ok=True, parents=True)
        temp_path = upload_dir / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        st.session_state["current_upload_signature"] = upload_signature

        try:
            loader = BMWDataLoader(temp_path)
            df = loader.load()

            st.success(f"✅ Loaded {len(df)} records")

            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                if "date" in df.columns:
                    st.metric(
                        "Date Range",
                        f"{df['date'].min().date()} - {df['date'].max().date()}",
                    )
            with col3:
                if "sales" in df.columns:
                    st.metric("Total Sales", f"{df['sales'].sum():,.0f}")
            with col4:
                if "price" in df.columns:
                    st.metric("Avg Price", f"${df['price'].mean():,.2f}")

            st.session_state["data"] = df
            st.session_state["data_path"] = str(temp_path)
            st.session_state["analysis_result"] = None
            st.session_state["analysis_error"] = None

            if st.session_state.get(
                "processed_upload_signature"
            ) != upload_signature or (
                st.session_state.get("last_run_path") != str(temp_path)
            ):
                st.session_state["auto_run_pending"] = True
            else:
                st.session_state["auto_run_pending"] = False

        except ValueError as e:
            error_msg = str(e)
            if (
                "Missing required columns" in error_msg
                or "essential column" in error_msg
            ):
                st.error(
                    f"⚠️ **Column Mapping Issue**: {error_msg}\n\n"
                    f"💡 Ensure your data has columns covering sales/volume, model/product, region/location, price/cost, and date/year."
                )
            else:
                st.error(f"❌ **Validation Error**: {error_msg}")
            app_logger.error(f"Data loading error: {e}")
        except Exception as e:
            st.error(f"❌ **Error loading data**: {str(e)}")
            app_logger.error(f"Data loading error: {e}")

    if st.session_state.get("auto_run_pending") and st.session_state.get("data_path"):
        run_analysis_workflow(auto_trigger=True)

    data_available = st.session_state.get("data") is not None

    st.header("2️⃣ Insights & Analysis")
    if not data_available:
        st.info("Upload data above to kick off the automated analysis.")
    else:
        st.caption(
            "The pipeline runs automatically after each upload. Use the button below to re-run with new settings."
        )
        if st.button("🔁 Re-run Analysis", type="primary"):
            run_analysis_workflow(auto_trigger=False)

    result = st.session_state.get("analysis_result")
    if st.session_state.get("analysis_error"):
        st.error(st.session_state["analysis_error"])

    if result:
        if result.get("errors"):
            st.error("Errors occurred during analysis:")
            for error in result["errors"]:
                st.error(f"- {error}")

        if result.get("insights"):
            st.subheader("📋 Key Insights")
            for i, insight in enumerate(result["insights"][:5], 1):
                with st.expander(
                    f"Insight {i}: {insight.get('insight', 'N/A')[:50]}..."
                ):
                    st.write(f"**Evidence:** {insight.get('evidence', 'N/A')}")
                    st.write(f"**Impact:** {insight.get('impact', 'N/A')}")
                    st.write(f"**Confidence:** {insight.get('confidence', 0):.2f}")

        if result and result.get("plot_specs"):
            st.subheader("📊 Planned Visualizations")
            st.json(result["plot_specs"])

    st.header("3️⃣ Visualization Gallery")
    if not result:
        st.info("Run the analysis to generate visualizations.")
    elif result.get("plot_paths"):
        for plot_path in result["plot_paths"]:
            if Path(plot_path).exists():
                st.image(plot_path, width="stretch")
            else:
                st.warning(f"Plot not found: {plot_path}")
    else:
        st.info("No plots generated yet.")

    st.header("4️⃣ Reports & Downloads")
    if not result:
        st.info("Run the analysis to generate downloadable reports.")
        return

    if result.get("report_paths"):
        st.subheader("📥 Download Reports")

        for format_type, path in result["report_paths"].items():
            if Path(path).exists():
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {format_type.upper()}",
                        data=f.read(),
                        file_name=Path(path).name,
                        mime=get_mime_type(format_type),
                    )
            else:
                st.warning(f"Report file not found: {path}")

        if "html" in result["report_paths"]:
            html_path = result["report_paths"]["html"]
            if Path(html_path).exists():
                st.subheader("📖 Report Preview")
                with open(html_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=800, scrolling=True)

        if result.get("evaluation_results"):
            st.subheader("📊 Evaluation Results")
            eval_results = result["evaluation_results"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Coverage", f"{eval_results.get('coverage', 0):.2f}")
            with col2:
                st.metric("Completeness", f"{eval_results.get('completeness', 0):.2f}")
            with col3:
                st.metric("Readability", f"{eval_results.get('readability', 0):.2f}")
            with col4:
                st.metric("Overall", f"{eval_results.get('overall', 0):.2f}")
    else:
        st.info("No reports generated yet. Complete the workflow to generate reports.")


def get_mime_type(format_type: str) -> str:
    """Get MIME type for file format.

    Args:
        format_type: File format name.

    Returns:
        MIME type string.
    """
    mime_types = {
        "pdf": "application/pdf",
        "html": "text/html",
        "word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "markdown": "text/markdown",
    }
    return mime_types.get(format_type.lower(), "application/octet-stream")


if __name__ == "__main__":
    main()
