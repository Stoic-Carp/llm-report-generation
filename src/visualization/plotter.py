"""Plot generation engine."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import get_settings
from src.utils.logger import app_logger
from src.visualization.themes import setup_plot_style


class VisualizationEngine:
    """Engine for generating visualizations from plot specifications."""

    def __init__(self, data: pd.DataFrame, output_dir: Path | None = None):
        """Initialize visualization engine.

        Args:
            data: DataFrame containing the data to visualize.
            output_dir: Directory to save plots. If None, uses settings default.
        """
        self.data = data.copy()
        self.settings = get_settings()
        self.output_dir = (
            Path(output_dir) if output_dir else self.settings.visualization.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = app_logger
        setup_plot_style()

    def generate_from_specs(self, plot_specs: List[Dict[str, Any]]) -> List[str]:
        """Generate plots from specifications.

        Args:
            plot_specs: List of plot specification dictionaries.

        Returns:
            List of paths to generated plot files.
        """
        self.logger.info(f"Generating {len(plot_specs)} plots")

        plot_paths = []

        for i, spec in enumerate(plot_specs):
            try:
                plot_path = self._generate_plot(spec, i)
                plot_paths.append(plot_path)
                self.logger.info(
                    f"Generated plot {i + 1}/{len(plot_specs)}: {plot_path}"
                )
            except Exception as e:
                self.logger.error(f"Failed to generate plot {i + 1}: {e}")
                continue

        return plot_paths

    def _generate_plot(self, spec: Dict[str, Any], index: int) -> str:
        """Generate a single plot from specification.

        Args:
            spec: Plot specification dictionary.
            index: Plot index for filename.

        Returns:
            Path to generated plot file.

        Raises:
            ValueError: If plot type is unsupported or columns don't exist.
        """
        plot_type = spec.get("plot_type", "line").lower()
        title = spec.get("title", f"Plot {index + 1}")
        x_axis = spec.get("x_axis")
        y_axis = spec.get("y_axis")
        groupby = spec.get("groupby")
        color = spec.get("color")

        # Map column names intelligently (handle LLM suggestions that might use original names)
        available_columns = list(self.data.columns)
        available_lower = [col.lower() for col in available_columns]

        def find_column(col_name: str | None) -> str | None:
            """Find matching column name (case-insensitive, handles variations)."""
            if not col_name:
                return None

            col_clean = col_name.strip()
            col_lower = (
                col_clean.lower().replace("_", "").replace("-", "").replace(" ", "")
            )

            # Try exact match first
            if col_clean in available_columns:
                return col_clean

            # Try case-insensitive match
            for i, avail_lower in enumerate(available_lower):
                avail_clean = (
                    avail_lower.replace("_", "").replace("-", "").replace(" ", "")
                )
                if col_lower == avail_clean:
                    return available_columns[i]

            # Try partial match
            for i, avail_lower in enumerate(available_lower):
                avail_clean = (
                    avail_lower.replace("_", "").replace("-", "").replace(" ", "")
                )
                if col_lower in avail_clean or avail_clean in col_lower:
                    if len(col_lower) >= 3:  # Only if meaningful match
                        return available_columns[i]

            # Common mappings (expanded)
            mapping = {
                "year": "date",
                "salesvolume": "sales",
                "salesshare": "sales",
                "priceusd": "price",
                "enginesizel": "Engine_Size_L",
                "enginesize": "Engine_Size_L",
                "mileagekm": "Mileage_KM",
                "mileage": "Mileage_KM",
                "fueltype": "Fuel_Type",
                "transmission": "Transmission",
                "color": "Color",
            }

            if col_lower in mapping:
                mapped_col = mapping[col_lower]
                if mapped_col in available_columns:
                    return mapped_col

            return None

        # Map columns
        x_axis = find_column(x_axis)
        y_axis = find_column(y_axis)
        groupby = find_column(groupby)
        color = find_column(color)

        if plot_type == "heatmap" and (not x_axis or not y_axis):
            self.logger.info(
                "Heatmap without explicit axes specified - generating correlation matrix."
            )
            x_axis = x_axis or "dummy_x"
            y_axis = y_axis or "dummy_y"
        elif not x_axis or not y_axis:
            self.logger.warning("Column mapping failed. Attempting fallback mapping...")
            if not x_axis:
                for col in ["model", "region", "date"]:
                    if col in available_columns:
                        x_axis = col
                        self.logger.info(f"Using fallback x_axis: {x_axis}")
                        break
            if not y_axis and "sales" in available_columns:
                y_axis = "sales"
                self.logger.info("Using fallback y_axis: sales")

            if not x_axis or not y_axis:
                raise ValueError(
                    f"Required columns (x_axis, y_axis) not found. "
                    f"Spec: {spec}. Available columns: {available_columns}"
                )

        aggregation = (spec.get("aggregation") or "sum").lower()
        sort_order = (spec.get("sort") or "none").lower()
        limit = spec.get("limit")

        skip_aggregation = plot_type == "heatmap" and (
            x_axis == "dummy_x" or y_axis == "dummy_y"
        )

        if skip_aggregation:
            plot_data = self.data.copy()
        else:
            plot_data = self._prepare_data(
                x_axis, y_axis, groupby, aggregation, sort_order, limit
            )

        return self._render_matplotlib(
            plot_type,
            title,
            plot_data,
            x_axis,
            y_axis,
            groupby,
            color,
            sort_order,
            limit,
            index,
        )

    def _prepare_data(
        self,
        x_axis: str,
        y_axis: str,
        groupby: str | None,
        aggregation: str,
        sort_order: str,
        limit: int | None,
    ) -> pd.DataFrame:
        """Prepare and aggregate data for plotting.

        Args:
            x_axis: X-axis column name.
            y_axis: Y-axis column name.
            groupby: Optional grouping column.
            aggregation: Aggregation method (sum, mean, count).
            sort_order: Sort order (desc, asc, none).
            limit: Limit number of rows (for top N).

        Returns:
            Prepared DataFrame.
        """
        data = self.data.copy()

        # Ensure aggregation matches data type
        y_is_numeric = pd.api.types.is_numeric_dtype(data[y_axis])
        if aggregation in {"sum", "mean"} and not y_is_numeric:
            self.logger.warning(
                f"Non-numeric y-axis '{y_axis}' with aggregation '{aggregation}'. "
                "Falling back to 'count'."
            )
            aggregation = "count"

        if y_axis == x_axis and aggregation in {"sum", "mean"}:
            self.logger.info(
                "Aggregation requested on identical x/y axes; using 'count' instead."
            )
            aggregation = "count"

        group_columns: list[str] = []
        if x_axis:
            group_columns.append(x_axis)
        if groupby and groupby not in group_columns:
            group_columns.append(groupby)

        needs_aggregation = aggregation != "none" and bool(group_columns)

        if needs_aggregation:
            agg_func = {"sum": "sum", "mean": "mean", "count": "count"}.get(
                aggregation, "sum"
            )

            plot_data = (
                data.groupby(group_columns)[y_axis]
                .agg(agg_func)
                .reset_index()
                .rename(columns={y_axis: f"{y_axis}_{agg_func}"})
            )

            # Ensure the aggregated column is consistently referenced as y_axis
            agg_column = f"{y_axis}_{agg_func}"
            plot_data = plot_data.rename(columns={agg_column: y_axis})

            # Sort if requested
            if sort_order == "desc":
                plot_data = plot_data.sort_values(y_axis, ascending=False)
            elif sort_order == "asc":
                plot_data = plot_data.sort_values(y_axis, ascending=True)

            if limit and limit > 0:
                plot_data = plot_data.head(limit)
        else:
            plot_data = data[[col for col in {x_axis, y_axis} if col]].copy()

        return plot_data

    def _render_matplotlib(
        self,
        plot_type: str,
        title: str,
        plot_data: pd.DataFrame,
        x_axis: Optional[str],
        y_axis: Optional[str],
        groupby: Optional[str],
        color: Optional[str],
        sort_order: str,
        limit: Optional[int],
        index: int,
    ) -> str:
        supported = {"line", "bar", "scatter", "heatmap", "box"}
        if plot_type not in supported:
            raise ValueError(
                f"Unsupported plot type: {plot_type}. "
                "Switch visualization.engine to 'plotly' for this chart."
            )

        fig, ax = plt.subplots(figsize=self.settings.visualization.figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if plot_type == "line":
            self._create_line_plot(ax, plot_data, x_axis, y_axis, groupby, color)
        elif plot_type == "bar":
            self._create_bar_plot(
                ax, plot_data, x_axis, y_axis, groupby, color, sort_order, limit
            )
        elif plot_type == "scatter":
            self._create_scatter_plot(ax, plot_data, x_axis, y_axis, color)
        elif plot_type == "heatmap":
            self._create_heatmap(ax, plot_data, x_axis, y_axis, groupby)
        elif plot_type == "box":
            self._create_box_plot(ax, plot_data, x_axis, y_axis, groupby)

        ax.set_title(title, pad=20, color="#2C3E50")

        if plot_type == "bar" and x_axis and self._is_horizontal_bar(plot_data, x_axis):
            ax.set_xlabel(self._format_label(y_axis or ""), labelpad=10)
            ax.set_ylabel(self._format_label(x_axis), labelpad=10)
            try:
                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
            except Exception:
                pass
        else:
            ax.set_xlabel(self._format_label(x_axis or ""), labelpad=10)
            ax.set_ylabel(self._format_label(y_axis or ""), labelpad=10)
            try:
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
            except Exception:
                pass

        ax.set_axisbelow(True)

        plot_path = self._build_filename(title, index)
        plt.savefig(
            plot_path,
            format=self.settings.visualization.format,
            dpi=self.settings.visualization.dpi,
        )
        plt.close()

        return str(plot_path)

    def _build_filename(
        self, title: str, index: int, extension: Optional[str] = None
    ) -> Path:
        ext = extension or self.settings.visualization.format
        safe_title = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in title.lower()
        ).strip("_")
        if not safe_title:
            safe_title = "plot"
        filename = f"plot_{index + 1:02d}_{safe_title[:40]}.{ext}"
        return self.output_dir / filename

    def _is_horizontal_bar(self, plot_data: pd.DataFrame, x_axis: str) -> bool:
        """Check if bar chart should be horizontal.

        Args:
            plot_data: Prepared DataFrame.
            x_axis: X-axis column name.

        Returns:
            True if horizontal bar chart should be used.
        """
        n_items = len(plot_data)
        labels = plot_data[x_axis].astype(str)
        long_labels = any(len(label_value) > 10 for label_value in labels)
        return n_items > 8 or long_labels

    def _format_label(self, column_name: str) -> str:
        """Format column name for axis label.

        Args:
            column_name: Column name.

        Returns:
            Formatted label.
        """
        # Replace underscores with spaces and title case
        label = column_name.replace("_", " ").title()
        # Common replacements
        replacements = {
            "Sales": "Sales Volume",
            "Price": "Price (USD)",
            "Date": "Date",
            "Model": "Model",
            "Region": "Region",
        }
        for key, value in replacements.items():
            if label.startswith(key):
                label = label.replace(key, value, 1)
                break
        return label

    def _create_line_plot(
        self,
        ax,
        plot_data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        groupby: str | None,
        color: str | None,
    ):
        """Create a line plot with trend analysis.

        Args:
            ax: Matplotlib axes object.
            plot_data: Prepared DataFrame.
            x_axis: Column name for x-axis.
            y_axis: Column name for y-axis.
            groupby: Optional column to group by.
            color: Optional column for color coding.
        """
        # Handle datetime columns
        if pd.api.types.is_datetime64_any_dtype(plot_data[x_axis]):
            plot_data = plot_data.sort_values(x_axis)

        from src.visualization.themes import PROFESSIONAL_COLORS

        if groupby:
            unique_groups = plot_data[groupby].unique()
            colors = (
                PROFESSIONAL_COLORS[: len(unique_groups)]
                if len(unique_groups) <= len(PROFESSIONAL_COLORS)
                else sns.color_palette("husl", len(unique_groups))
            )

            for idx, group_value in enumerate(unique_groups):
                group_data = plot_data[plot_data[groupby] == group_value]
                if pd.api.types.is_datetime64_any_dtype(plot_data[x_axis]):
                    group_data = group_data.sort_values(x_axis)

                ax.plot(
                    group_data[x_axis],
                    group_data[y_axis],
                    label=str(group_value),
                    marker="o",
                    linewidth=2.5,
                    markersize=6,
                    color=colors[idx % len(colors)],
                    alpha=0.9,
                )

            # Improve legend
            ax.legend(title=self._format_label(groupby), loc="best")
        else:
            plot_data = (
                plot_data.sort_values(x_axis)
                if not pd.api.types.is_datetime64_any_dtype(plot_data[x_axis])
                else plot_data
            )
            ax.plot(
                plot_data[x_axis],
                plot_data[y_axis],
                marker="o",
                linewidth=3,
                markersize=8,
                color=PROFESSIONAL_COLORS[0],
                alpha=0.9,
                zorder=3,
            )

            # Add trend line for time series
            if pd.api.types.is_datetime64_any_dtype(plot_data[x_axis]):
                try:
                    x_numeric = pd.to_numeric(plot_data[x_axis])
                    z = np.polyfit(x_numeric, plot_data[y_axis], 1)
                    p = np.poly1d(z)
                    ax.plot(
                        plot_data[x_axis],
                        p(x_numeric),
                        "--",
                        alpha=0.8,
                        label="Trend",
                        linewidth=2,
                        color=PROFESSIONAL_COLORS[1],
                        zorder=2,
                    )
                    ax.legend()
                except Exception:
                    pass

        # Smart date formatting
        if pd.api.types.is_datetime64_any_dtype(plot_data[x_axis]):
            import matplotlib.dates as mdates

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _create_bar_plot(
        self,
        ax,
        plot_data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        groupby: str | None,
        color: str | None,
        sort_order: str,
        limit: int | None,
    ):
        """Create an informative bar plot.

        Args:
            ax: Matplotlib axes object.
            plot_data: Prepared DataFrame.
            x_axis: Column name for x-axis.
            y_axis: Column name for y-axis.
            groupby: Optional column to group by.
            color: Optional column for color coding.
            sort_order: Sort order.
            limit: Limit number of bars.
        """
        from src.visualization.themes import PROFESSIONAL_COLORS

        if groupby:
            pivot_data = plot_data.pivot_table(
                values=y_axis,
                index=x_axis,
                columns=groupby,
                aggfunc="sum",
            )

            # Ensure we don't have too many groups for colors
            n_groups = len(pivot_data.columns)
            colors = (
                PROFESSIONAL_COLORS[:n_groups]
                if n_groups <= len(PROFESSIONAL_COLORS)
                else sns.color_palette("husl", n_groups)
            )

            pivot_data.plot(
                kind="bar",
                ax=ax,
                stacked=False,
                width=0.8,
                color=colors,
                edgecolor="none",
            )
            ax.legend(title=self._format_label(groupby))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            # Use horizontal bars for better readability when there are many categories
            # OR if the labels are long
            n_items = len(plot_data)
            labels = plot_data[x_axis].astype(str)
            long_labels = any(len(label_value) > 10 for label_value in labels)

            use_horizontal = n_items > 8 or long_labels

            if use_horizontal:
                # Sort for horizontal bar (top value at top)
                if sort_order == "desc":
                    plot_data = plot_data.sort_values(
                        y_axis, ascending=True
                    )  # Invert for horizontal
                elif sort_order == "asc":
                    plot_data = plot_data.sort_values(y_axis, ascending=False)

                # Reset index to ensure clean iteration
                plot_data = plot_data.reset_index(drop=True)

                ax.barh(
                    range(len(plot_data)),
                    plot_data[y_axis],
                    color=PROFESSIONAL_COLORS[0],
                    edgecolor="none",
                    height=0.6,
                )
                ax.set_yticks(range(len(plot_data)))
                ax.set_yticklabels(plot_data[x_axis].values)

                # Add value labels
                max_val = plot_data[y_axis].max()
                for i in range(len(plot_data)):
                    value = plot_data.iloc[i][y_axis]
                    ax.text(
                        value + max_val * 0.01,
                        i,
                        f"{value:,.0f}",
                        va="center",
                        fontsize=9,
                        color="#2C3E50",
                    )
            else:
                # Reset index for clean iteration
                plot_data = plot_data.reset_index(drop=True)

                ax.bar(
                    range(len(plot_data)),
                    plot_data[y_axis],
                    color=PROFESSIONAL_COLORS[0],
                    width=0.6,
                    edgecolor="none",
                )
                ax.set_xticks(range(len(plot_data)))
                ax.set_xticklabels(plot_data[x_axis].values, rotation=45, ha="right")

                # Add value labels
                max_val = plot_data[y_axis].max()
                for i in range(len(plot_data)):
                    value = plot_data.iloc[i][y_axis]
                    ax.text(
                        i,
                        value + max_val * 0.02,
                        f"{value:,.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#2C3E50",
                    )

    def _create_scatter_plot(
        self, ax, plot_data: pd.DataFrame, x_axis: str, y_axis: str, color: str | None
    ):
        """Create a scatter plot with correlation analysis.

        Args:
            ax: Matplotlib axes object.
            plot_data: Prepared DataFrame.
            x_axis: Column name for x-axis.
            y_axis: Column name for y-axis.
            color: Optional column for color coding.
        """
        from src.visualization.themes import PROFESSIONAL_COLORS

        # For large datasets, sample to avoid overplotting
        if len(plot_data) > 5000:
            plot_sample = plot_data.sample(n=5000, random_state=42)
            self.logger.info(
                f"Sampling {len(plot_sample)} points from {len(plot_data)} for scatter plot"
            )
        else:
            plot_sample = plot_data

        # Use hexbin for better visualization of dense data
        if len(plot_sample) > 1000:
            # Create hexbin plot for dense data
            hexbin = ax.hexbin(
                plot_sample[x_axis],
                plot_sample[y_axis],
                gridsize=30,
                cmap="Blues",
                alpha=0.8,
                edgecolors="none",
                mincnt=1,
            )
            cbar = plt.colorbar(hexbin, ax=ax, pad=0.02)
            cbar.set_label("Count", fontsize=10)
        else:
            # Traditional scatter for smaller datasets
            if color and color in plot_sample.columns:
                scatter = ax.scatter(
                    plot_sample[x_axis],
                    plot_sample[y_axis],
                    c=plot_sample[color],
                    cmap="viridis",
                    alpha=0.6,
                    s=60,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                cbar.set_label(self._format_label(color))
            else:
                ax.scatter(
                    plot_sample[x_axis],
                    plot_sample[y_axis],
                    alpha=0.6,
                    s=60,
                    color=PROFESSIONAL_COLORS[0],
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )

        # Add trend line
        try:
            if pd.api.types.is_numeric_dtype(
                plot_data[x_axis]
            ) and pd.api.types.is_numeric_dtype(plot_data[y_axis]):
                # Calculate correlation
                corr = plot_data[x_axis].corr(plot_data[y_axis])

                # Add regression line
                z = np.polyfit(plot_data[x_axis], plot_data[y_axis], 1)
                p = np.poly1d(z)
                x_line = np.linspace(
                    plot_data[x_axis].min(), plot_data[x_axis].max(), 100
                )
                ax.plot(
                    x_line,
                    p(x_line),
                    "--",
                    color=PROFESSIONAL_COLORS[1],
                    linewidth=2.5,
                    alpha=0.8,
                    label=f"Trend (r={corr:.3f})",
                    zorder=4,
                )

                # Add correlation info box
                corr_strength = (
                    "Strong"
                    if abs(corr) > 0.7
                    else "Moderate"
                    if abs(corr) > 0.3
                    else "Weak"
                )
                corr_direction = "positive" if corr > 0 else "negative"
                textstr = f"{corr_strength} {corr_direction}\ncorrelation: {corr:.3f}"
                ax.text(
                    0.05,
                    0.95,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="#bdc3c7",
                        linewidth=1.5,
                    ),
                    verticalalignment="top",
                    fontweight="bold",
                )
                ax.legend(loc="lower right", framealpha=0.9)
        except Exception as e:
            self.logger.warning(f"Could not add correlation analysis: {e}")

    def _create_heatmap(
        self, ax, plot_data: pd.DataFrame, x_axis: str, y_axis: str, groupby: str | None
    ):
        """Create a heatmap.

        Args:
            ax: Matplotlib axes object.
            plot_data: Prepared DataFrame.
            x_axis: Column name for x-axis.
            y_axis: Column name for y-axis.
            groupby: Optional column to group by.
        """
        # Check if this is a correlation heatmap request (dummy axes or no valid columns)
        is_correlation_request = (
            x_axis == "dummy_x"
            or y_axis == "dummy_y"
            or x_axis not in plot_data.columns
            or y_axis not in plot_data.columns
        )

        if is_correlation_request or (
            not groupby and len(plot_data.select_dtypes(include=["number"]).columns) > 1
        ):
            # Create correlation heatmap
            numeric_cols = plot_data.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 1:
                self.logger.info(
                    f"Creating correlation heatmap with columns: {list(numeric_cols)}"
                )
                corr_data = plot_data[numeric_cols].corr()
                sns.heatmap(
                    corr_data,
                    annot=True,
                    fmt=".2f",
                    ax=ax,
                    cmap="RdBu_r",
                    center=0,
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    cbar_kws={"label": "Correlation"},
                )
                return
            else:
                self.logger.warning(
                    "Not enough numeric columns for correlation heatmap, falling back to pivot table"
                )

        # Regular heatmap with pivot table
        if groupby and groupby in plot_data.columns:
            try:
                pivot_data = plot_data.pivot_table(
                    values=y_axis,
                    index=x_axis,
                    columns=groupby,
                    aggfunc="sum",
                )
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt=".0f",
                    ax=ax,
                    cmap="Blues",
                    cbar_kws={"label": self._format_label(y_axis)},
                    linewidths=0.5,
                    linecolor="white",
                )
            except Exception as e:
                self.logger.warning(
                    f"Pivot table heatmap failed: {e}, trying correlation instead"
                )
                # Fallback to correlation
                numeric_cols = plot_data.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 1:
                    corr_data = plot_data[numeric_cols].corr()
                    sns.heatmap(
                        corr_data,
                        annot=True,
                        fmt=".2f",
                        ax=ax,
                        cmap="RdBu_r",
                        center=0,
                        square=True,
                        linewidths=0.5,
                        linecolor="white",
                    )
        else:
            # Simple pivot table or correlation
            try:
                if x_axis in plot_data.columns and y_axis in plot_data.columns:
                    pivot_data = plot_data.pivot_table(
                        values=y_axis,
                        index=x_axis,
                        aggfunc="sum",
                    )
                    sns.heatmap(
                        pivot_data.values.reshape(-1, 1),
                        annot=True,
                        fmt=".0f",
                        ax=ax,
                        cmap="Blues",
                        xticklabels=[y_axis],
                        yticklabels=pivot_data.index,
                        linewidths=0.5,
                        linecolor="white",
                    )
                else:
                    # Last resort: correlation matrix
                    numeric_cols = plot_data.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 1:
                        corr_data = plot_data[numeric_cols].corr()
                        sns.heatmap(
                            corr_data,
                            annot=True,
                            fmt=".2f",
                            ax=ax,
                            cmap="RdBu_r",
                            center=0,
                            square=True,
                            linewidths=0.5,
                            linecolor="white",
                        )
            except Exception as e:
                self.logger.error(f"All heatmap methods failed: {e}")
                raise

    def _create_box_plot(
        self, ax, plot_data: pd.DataFrame, x_axis: str, y_axis: str, groupby: str | None
    ):
        """Create a box plot showing distributions.

        Args:
            ax: Matplotlib axes object.
            plot_data: Prepared DataFrame.
            x_axis: Column name for x-axis.
            y_axis: Column name for y-axis.
            groupby: Optional column to group by.
        """
        from src.visualization.themes import PROFESSIONAL_COLORS

        if groupby:
            unique_groups = plot_data[groupby].unique()
            colors = (
                PROFESSIONAL_COLORS[: len(unique_groups)]
                if len(unique_groups) <= len(PROFESSIONAL_COLORS)
                else sns.color_palette("husl", len(unique_groups))
            )

            sns.boxplot(
                data=plot_data,
                x=x_axis,
                y=y_axis,
                hue=groupby,
                ax=ax,
                palette=colors,
                linewidth=1.0,
                fliersize=3,
            )
            ax.legend(title=self._format_label(groupby))
        else:
            sns.boxplot(
                data=plot_data,
                x=x_axis,
                y=y_axis,
                ax=ax,
                color=PROFESSIONAL_COLORS[0],
                linewidth=1.0,
                fliersize=3,
            )

        # Rotate x-axis labels if needed
        if len(plot_data[x_axis].unique()) > 5:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
