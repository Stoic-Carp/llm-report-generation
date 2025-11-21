"""CLI entry point for running analysis."""

import typer
from pathlib import Path
from config.settings import get_settings
from src.llm.graph import ReportGenerator
from src.utils.logger import app_logger

app = typer.Typer()


@app.command()
def analyze(
    data_path: Path = typer.Argument(..., help="Path to sales data file"),
    provider: str = typer.Option(None, help="LLM provider override"),
    model: str = typer.Option(None, help="LLM model override"),
):
    """Run analysis with CLI arguments.

    Args:
        data_path: Path to the data file.
        provider: Optional LLM provider override.
        model: Optional LLM model override.
    """
    settings = get_settings()

    # Override with CLI arguments
    if provider:
        settings.llm.provider = provider
    if model:
        settings.llm.model = model

    app_logger.info(f"Starting analysis with {settings.llm.provider}/{settings.llm.model}")
    app_logger.debug(f"Settings: {settings.model_dump_safe()}")

    # Run pipeline
    generator = ReportGenerator()
    result = generator.generate_report(str(data_path))

    # Print results
    if result.get("errors"):
        typer.echo("Errors occurred:", err=True)
        for error in result["errors"]:
            typer.echo(f"  - {error}", err=True)

    if result.get("report_paths"):
        typer.echo("\nReports generated:")
        for format_type, path in result["report_paths"].items():
            typer.echo(f"  - {format_type.upper()}: {path}")

    if result.get("evaluation_results"):
        typer.echo("\nEvaluation Results:")
        for metric, score in result["evaluation_results"].items():
            typer.echo(f"  - {metric}: {score:.2f}")


if __name__ == "__main__":
    app()

