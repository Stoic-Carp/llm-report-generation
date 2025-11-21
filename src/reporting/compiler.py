"""Report assembly and compilation module."""

from typing import Dict

from config.settings import Settings
from src.llm.state import ReportState
from src.reporting.html_generator import HTMLGenerator
from src.reporting.markdown_generator import MarkdownGenerator
from src.reporting.pdf_generator import PDFGenerator
from src.reporting.word_generator import WordGenerator
from src.utils.logger import app_logger


class ReportCompiler:
    """Compiles reports in multiple formats."""

    def __init__(self, settings: Settings, state: ReportState):
        """Initialize report compiler.

        Args:
            settings: Application settings.
            state: LangGraph workflow state.
        """
        self.settings = settings
        self.state = state
        self.logger = app_logger

    def generate_all_formats(self) -> Dict[str, str]:
        """Generate reports in all requested formats.

        Returns:
            Dictionary mapping format names to file paths.
        """
        self.logger.info("Compiling reports in all formats")

        report_paths = {}

        # Generate markdown first (base format)
        if self.settings.report.generate_markdown:
            markdown_gen = MarkdownGenerator(self.settings, self.state)
            markdown_path = markdown_gen.generate()
            report_paths["markdown"] = markdown_path
            self.state["report_markdown"] = markdown_path

        # Generate other formats
        if self.settings.report.generate_pdf:
            pdf_gen = PDFGenerator(self.settings, self.state)
            pdf_path = pdf_gen.generate()
            report_paths["pdf"] = pdf_path

        if self.settings.report.generate_html:
            html_gen = HTMLGenerator(self.settings, self.state)
            html_path = html_gen.generate()
            report_paths["html"] = html_path

        if self.settings.report.generate_word:
            word_gen = WordGenerator(self.settings, self.state)
            word_path = word_gen.generate()
            report_paths["word"] = word_path

        self.logger.info(f"Reports generated: {list(report_paths.keys())}")

        return report_paths
