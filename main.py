"""CLI entry point for running BMW sales analysis."""

from pathlib import Path
from typing import Optional

import typer

from config.settings import get_settings
from src.llm.graph import ReportGenerator
from src.utils.logger import app_logger

app = typer.Typer(
    add_completion=False,
    help="Run the BMW sales analysis workflow and generate a PDF report.",
)


def _configure_output_formats(settings, pdf_only: bool) -> None:
    """Ensure report formats align with CLI expectations.

    Args:
        settings: Application settings instance.
        pdf_only: Whether to disable non-PDF formats (markdown stays for evaluation).
    """
    settings.report.generate_pdf = True
    settings.report.generate_markdown = True  # Needed for evaluation + PDF content

    if pdf_only:
        settings.report.generate_html = False
        settings.report.generate_word = False
    else:
        # Respect user configuration for other formats
        settings.report.generate_html = settings.report.generate_html
        settings.report.generate_word = settings.report.generate_word


@app.command()
def run(
    data_path: Path = typer.Argument(
        default="data/raw/BMW sales data (2020-2024).xlsx",
        exists=True,
        resolve_path=True,
        readable=True,
        help="Path to the BMW sales data file (CSV/XLSX).",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Override the LLM provider for this run."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the LLM model for this run."
    ),
    pdf_only: bool = typer.Option(
        True,
        "--pdf-only/--all-formats",
        help="Force PDF-only generation (markdown kept for evaluation).",
    ),
) -> None:
    """Execute the end-to-end analysis pipeline and emit a PDF report."""
    settings = get_settings()

    if provider:
        app_logger.info("Overriding LLM provider to %s", provider)
        settings.llm.provider = provider
    if model:
        app_logger.info("Overriding LLM model to %s", model)
        settings.llm.model = model

    _configure_output_formats(settings, pdf_only)

    app_logger.info("Launching report generation for %s", data_path)
    generator = ReportGenerator()
    result = generator.generate_report(str(data_path))

    errors = result.get("errors") or []
    if errors:
        typer.echo("Errors occurred during report generation:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)

    report_paths = result.get("report_paths") or {}
    pdf_path = report_paths.get("pdf")
    if pdf_path:
        typer.echo(f"\nPDF report generated: {pdf_path}")
    else:
        typer.echo(
            "\nWorkflow completed but no PDF path was reported.",
            err=True,
        )

    if not errors and result.get("evaluation_results"):
        typer.echo("\nEvaluation summary:")
        for metric, score in result["evaluation_results"].items():
            typer.echo(f"  - {metric}: {score:.2f}")

    if errors:
        raise typer.Exit(code=1)


def main() -> None:
    """Typer entry point."""
    app()


if __name__ == "__main__":
    main()
