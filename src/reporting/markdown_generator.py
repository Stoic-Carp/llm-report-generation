"""Markdown report generator."""

from datetime import datetime
from typing import List

from config.settings import Settings
from src.llm.state import ReportState
from src.llm.utils import normalize_markdown
from src.utils.logger import app_logger


class MarkdownGenerator:
    """Generates markdown format reports."""

    def __init__(self, settings: Settings, state: ReportState):
        """Initialize markdown generator.

        Args:
            settings: Application settings.
            state: LangGraph workflow state.
        """
        self.settings = settings
        self.state = state
        self.logger = app_logger

    def generate(self) -> str:
        """Generate markdown report.

        Returns:
            Path to generated markdown file.
        """
        self.logger.info("Generating markdown report")

        markdown_content = self._build_markdown()

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bmw_sales_report_{timestamp}.md"
        output_path = self.settings.report.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        self.logger.info(f"Markdown report saved: {output_path}")

        return str(output_path)

    def _build_markdown(self) -> str:
        """Build markdown content from state."""

        sections: List[str] = []
        sections.extend(self._build_header_lines())
        sections.extend(self._build_table_of_contents_lines())
        sections.extend(self._build_key_metrics_lines())
        sections.extend(self._build_executive_summary_lines())
        sections.extend(self._build_analysis_section_lines())
        sections.extend(self._build_visualizations_lines())
        sections.extend(self._build_recommendations_lines())
        sections.extend(self._build_appendix_lines())

        return "\n".join(sections).strip() + "\n"

    def _build_header_lines(self) -> List[str]:
        """Create document header."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [
            f"# {self.settings.report.company_name}",
            "",
            f"**Generated:** {timestamp}",
            f"**Author:** {self.settings.report.author}",
            "",
            "---",
            "",
        ]

    def _build_table_of_contents_lines(self) -> List[str]:
        """Create table of contents section."""

        lines = [
            "## Table of Contents",
            "",
            "1. [Executive Summary](#executive-summary)",
            "2. [Sales Performance Trends](#sales-performance-trends)",
            "3. [Regional Analysis](#regional-analysis)",
            "4. [Product Performance](#product-performance)",
            "5. [Recommendations](#recommendations)",
        ]

        if self.settings.report.include_appendix:
            lines.append("6. [Appendix](#appendix)")

        lines.extend(["", "---", ""])
        return lines

    def _build_key_metrics_lines(self) -> List[str]:
        """Render key metrics dashboard."""

        key_metrics = self.state.get("key_metrics") or {}
        if not key_metrics:
            return []

        lines = ["## Key Metrics Dashboard", ""]

        total_sales = key_metrics.get("total_sales")
        sales_yoy = key_metrics.get("sales_yoy_pct")
        prev_year = key_metrics.get("previous_sales_year")
        if total_sales is not None:
            yoy_text = (
                f" ({sales_yoy:+.1f}% vs {prev_year})"
                if sales_yoy is not None and prev_year
                else ""
            )
            lines.append(f"- **Total Sales:** {total_sales:,.0f}{yoy_text}")

        avg_price = key_metrics.get("average_price")
        price_yoy = key_metrics.get("price_yoy_pct")
        if avg_price is not None:
            yoy_text = f" ({price_yoy:+.1f}% YoY)" if price_yoy is not None else ""
            lines.append(f"- **Average Price:** ${avg_price:,.2f}{yoy_text}")

        lines.append("")

        top_models = key_metrics.get("top_models") or []
        if top_models:
            lines.append("| Top Model | Sales | Share |")
            lines.append("| --- | ---: | ---: |")
            for entry in top_models:
                lines.append(
                    f"| {entry.get('model','N/A')} | {entry.get('sales',0):,} | {entry.get('share',0):.2f}% |"
                )
            lines.append("")

        region_share = key_metrics.get("market_share_by_region") or {}
        if region_share:
            lines.append("| Region | Market Share |")
            lines.append("| --- | ---: |")
            for region, share in region_share.items():
                lines.append(f"| {region} | {share:.2f}% |")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _build_executive_summary_lines(self) -> List[str]:
        """Render executive summary block."""

        summary = normalize_markdown(self.state.get("executive_summary", ""))
        if not summary:
            return []

        return ["## Executive Summary", "", summary, "", "---", ""]

    def _build_analysis_section_lines(self) -> List[str]:
        """Render detailed analysis sections."""

        sections_state = self.state.get("analysis_sections") or {}
        if not sections_state:
            return []

        order = [
            ("sales_trends", "Sales Performance Trends"),
            ("regional_analysis", "Regional Analysis"),
            ("product_performance", "Product Performance"),
        ]

        lines: List[str] = []
        for key, heading in order:
            content = normalize_markdown(sections_state.get(key, ""))
            if not content:
                continue
            lines.extend([f"## {heading}", "", content, "", "---", ""])

        return lines

    def _build_visualizations_lines(self) -> List[str]:
        """Render visualizations section."""

        plot_paths = self.state.get("plot_paths") or []
        if not plot_paths:
            return []

        lines = ["## Visualizations", ""]
        for i, plot_path in enumerate(plot_paths, 1):
            lines.append(f"![Plot {i}]({plot_path})")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _build_recommendations_lines(self) -> List[str]:
        """Render recommendations list."""

        recommendations = self.state.get("recommendations") or []
        if not recommendations:
            return []

        lines = ["## Recommendations", ""]
        for rec in recommendations:
            lines.append(f"- {rec}")

        lines.extend(["", "---", ""])
        return lines

    def _build_appendix_lines(self) -> List[str]:
        """Render appendix content if enabled."""

        if not self.settings.report.include_appendix:
            return []

        lines = ["## Appendix", "", "### Key Insights", ""]
        if self.state.get("insights"):
            for insight in self.state["insights"][:10]:
                lines.append(f"- **{insight.get('insight', 'N/A')}**")
                lines.append(f"  - Evidence: {insight.get('evidence', 'N/A')}")
                lines.append(f"  - Impact: {insight.get('impact', 'N/A')}")
                lines.append(f"  - Confidence: {insight.get('confidence', 0):.2f}")
                lines.append("")

        return lines

