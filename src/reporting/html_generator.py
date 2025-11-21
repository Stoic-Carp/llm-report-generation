"""HTML report generator."""

from pathlib import Path
from datetime import datetime
from config.settings import Settings
from src.llm.state import ReportState
from src.utils.logger import app_logger


class HTMLGenerator:
    """Generates HTML format reports."""

    def __init__(self, settings: Settings, state: ReportState):
        """Initialize HTML generator.

        Args:
            settings: Application settings.
            state: LangGraph workflow state.
        """
        self.settings = settings
        self.state = state
        self.logger = app_logger

    def generate(self) -> str:
        """Generate HTML report.

        Returns:
            Path to generated HTML file.
        """
        self.logger.info("Generating HTML report")

        html_content = self._build_html()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bmw_sales_report_{timestamp}.html"
        output_path = self.settings.report.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"HTML report saved: {output_path}")

        return str(output_path)

    def _build_html(self) -> str:
        """Build HTML content from state.

        Returns:
            Complete HTML report content.
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{self.settings.report.company_name}</title>",
            self._get_css(),
            "</head>",
            "<body>",
            "<div class='container'>",
        ]

        # Header
        html_parts.append("<header>")
        html_parts.append(f"<h1>{self.settings.report.company_name}</h1>")
        html_parts.append(f"<p class='meta'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html_parts.append(f"<p class='meta'>Author: {self.settings.report.author}</p>")
        html_parts.append("</header>")

        # Executive Summary
        if self.state.get("executive_summary"):
            html_parts.append("<section>")
            html_parts.append("<h2>Executive Summary</h2>")
            html_parts.append(f"<div class='content'>{self._markdown_to_html(self.state['executive_summary'])}</div>")
            html_parts.append("</section>")

        # Analysis Sections
        if self.state.get("analysis_sections"):
            sections = self.state["analysis_sections"]

            if "sales_trends" in sections:
                html_parts.append("<section>")
                html_parts.append("<h2>Sales Performance Trends</h2>")
                html_parts.append(f"<div class='content'>{self._markdown_to_html(sections['sales_trends'])}</div>")
                html_parts.append("</section>")

            if "regional_analysis" in sections:
                html_parts.append("<section>")
                html_parts.append("<h2>Regional Analysis</h2>")
                html_parts.append(f"<div class='content'>{self._markdown_to_html(sections['regional_analysis'])}</div>")
                html_parts.append("</section>")

            if "product_performance" in sections:
                html_parts.append("<section>")
                html_parts.append("<h2>Product Performance</h2>")
                html_parts.append(f"<div class='content'>{self._markdown_to_html(sections['product_performance'])}</div>")
                html_parts.append("</section>")

        # Visualizations
        if self.state.get("plot_paths"):
            html_parts.append("<section>")
            html_parts.append("<h2>Visualizations</h2>")
            html_parts.append("<div class='visualizations'>")
            for plot_path in self.state["plot_paths"]:
                if Path(plot_path).exists():
                    # Convert to relative path for HTML
                    plot_rel_path = Path(plot_path).relative_to(self.settings.report.output_dir.parent)
                    html_parts.append("<div class='plot-container'>")
                    html_parts.append(f"<img src='{plot_rel_path}' alt='Data Visualization' class='plot-image'>")
                    html_parts.append("</div>")
                else:
                    html_parts.append(f"<p class='warning'>Plot not found: {plot_path}</p>")
            html_parts.append("</div>")
            html_parts.append("</section>")

        # Recommendations
        if self.state.get("recommendations"):
            html_parts.append("<section>")
            html_parts.append("<h2>Recommendations</h2>")
            html_parts.append("<ul class='recommendations'>")
            for rec in self.state["recommendations"]:
                html_parts.append(f"<li>{rec}</li>")
            html_parts.append("</ul>")
            html_parts.append("</section>")

        html_parts.extend(["</div>", "</body>", "</html>"])

        return "\n".join(html_parts)

    def _get_css(self) -> str:
        """Get CSS styles for HTML report.

        Returns:
            CSS style string.
        """
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 40px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            header {
                border-bottom: 3px solid #0066cc;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            h1 {
                color: #0066cc;
                margin-bottom: 10px;
            }
            .meta {
                color: #666;
                font-size: 0.9em;
            }
            section {
                margin-bottom: 40px;
            }
            h2 {
                color: #0066cc;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 10px;
            }
            .content {
                margin-top: 20px;
            }
            .visualizations {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .plot-image {
                max-width: 100%;
                height: auto;
                border: 2px solid #E0E0E0;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 20px 0;
                display: block;
                background: white;
                padding: 10px;
            }
            .plot-image:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transition: box-shadow 0.3s ease;
            }
            .plot-container {
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: #FAFAFA;
                border-radius: 8px;
            }
            .recommendations {
                list-style-type: disc;
                padding-left: 30px;
            }
            .recommendations li {
                margin-bottom: 10px;
            }
        </style>
        """

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML (simple conversion).

        Args:
            markdown_text: Markdown text to convert.

        Returns:
            HTML string.
        """
        # Simple markdown to HTML conversion
        html = markdown_text.replace("\n\n", "</p><p>")
        html = html.replace("\n", "<br>")
        html = f"<p>{html}</p>"
        return html

