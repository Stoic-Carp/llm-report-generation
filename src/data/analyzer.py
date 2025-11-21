"""Statistical analysis module."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from src.utils.logger import app_logger


class StatisticalAnalyzer:
    """Generates comprehensive statistical analysis."""

    def __init__(self, df: pd.DataFrame):
        """Initialize analyzer.

        Args:
            df: DataFrame containing sales data.
        """
        self.df = df.copy()
        self.logger = app_logger

    def generate_profile(self) -> Dict[str, Any]:
        """Generate full data profile for LLM.

        Returns:
            Dictionary containing comprehensive data profile.
        """
        self.logger.info("Generating data profile")
        return {
            "basic_stats": self._basic_statistics(),
            "temporal_trends": self._temporal_analysis(),
            "regional_breakdown": self._regional_analysis(),
            "model_performance": self._model_analysis(),
            "correlations": self._correlation_analysis(),
            "summary_insights": self._generate_summary(),
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        return self._basic_statistics()

    def _basic_statistics(self) -> Dict[str, Any]:
        """Basic descriptive statistics.

        Returns:
            Dictionary with basic statistics.
        """
        stats = {
            "total_records": len(self.df),
            "date_range": {
                "start": str(self.df["date"].min()),
                "end": str(self.df["date"].max()),
            } if "date" in self.df.columns else None,
            "total_sales": int(self.df["sales"].sum()) if "sales" in self.df.columns else None,
            "avg_price": float(self.df["price"].mean()) if "price" in self.df.columns else None,
        }

        if "sales" in self.df.columns:
            stats["sales_stats"] = {
                "mean": float(self.df["sales"].mean()),
                "median": float(self.df["sales"].median()),
                "std": float(self.df["sales"].std()),
                "min": int(self.df["sales"].min()),
                "max": int(self.df["sales"].max()),
            }

        if "price" in self.df.columns:
            stats["price_stats"] = {
                "mean": float(self.df["price"].mean()),
                "median": float(self.df["price"].median()),
                "std": float(self.df["price"].std()),
                "min": float(self.df["price"].min()),
                "max": float(self.df["price"].max()),
            }

        return stats

    def _temporal_analysis(self) -> Dict[str, Any]:
        """Time series analysis.

        Returns:
            Dictionary with temporal analysis results.
        """
        if "date" not in self.df.columns:
            return {}

        self.df["year"] = pd.to_datetime(self.df["date"]).dt.year
        self.df["month"] = pd.to_datetime(self.df["date"]).dt.month
        self.df["quarter"] = pd.to_datetime(self.df["date"]).dt.quarter

        yearly = (
            self.df.groupby("year")
            .agg({"sales": "sum", "price": "mean"})
            .to_dict("index")
        )

        monthly = (
            self.df.groupby("month")
            .agg({"sales": "sum", "price": "mean"})
            .to_dict("index")
        )

        return {
            "yearly_sales": yearly,
            "monthly_sales": monthly,
            "growth_rates": self._calculate_growth_rates(),
            "seasonality": self._detect_seasonality(),
        }

    def _calculate_growth_rates(self) -> Dict[str, float]:
        """Calculate year-over-year growth rates.

        Returns:
            Dictionary with growth rates.
        """
        if "date" not in self.df.columns or "sales" not in self.df.columns:
            return {}

        self.df["year"] = pd.to_datetime(self.df["date"]).dt.year
        yearly_sales = self.df.groupby("year")["sales"].sum().sort_index()

        growth_rates = {}
        for i in range(1, len(yearly_sales)):
            prev_year = yearly_sales.iloc[i - 1]
            curr_year = yearly_sales.iloc[i]
            if prev_year > 0:
                growth_rate = ((curr_year - prev_year) / prev_year) * 100
                growth_rates[f"{yearly_sales.index[i-1]}-{yearly_sales.index[i]}"] = (
                    float(growth_rate)
                )

        return growth_rates

    def _detect_seasonality(self) -> Dict[str, Any]:
        """Detect seasonal patterns.

        Returns:
            Dictionary with seasonality analysis.
        """
        if "date" not in self.df.columns or "sales" not in self.df.columns:
            return {}

        self.df["month"] = pd.to_datetime(self.df["date"]).dt.month
        monthly_avg = self.df.groupby("month")["sales"].mean()

        return {
            "peak_month": int(monthly_avg.idxmax()),
            "low_month": int(monthly_avg.idxmin()),
            "monthly_averages": monthly_avg.to_dict(),
        }

    def _regional_analysis(self) -> Dict[str, Any]:
        """Regional performance analysis.

        Returns:
            Dictionary with regional analysis.
        """
        if "region" not in self.df.columns:
            return {}

        regional = (
            self.df.groupby("region")
            .agg(
                {
                    "sales": ["sum", "mean", "count"],
                    "price": "mean",
                }
            )
            .to_dict("index")
        )

        # Top region by sales
        top_region = self.df.groupby("region")["sales"].sum().idxmax()

        return {
            "regional_stats": regional,
            "top_region": top_region,
            "regional_sales_share": (
                self.df.groupby("region")["sales"].sum() / self.df["sales"].sum()
            ).to_dict(),
        }

    def _model_analysis(self) -> Dict[str, Any]:
        """Model performance analysis.

        Returns:
            Dictionary with model analysis.
        """
        if "model" not in self.df.columns:
            return {}

        model_stats = (
            self.df.groupby("model")
            .agg(
                {
                    "sales": ["sum", "mean", "count"],
                    "price": "mean",
                }
            )
            .to_dict("index")
        )

        # Top and bottom models
        model_sales = self.df.groupby("model")["sales"].sum().sort_values(ascending=False)
        top_models = model_sales.head(5).to_dict()
        bottom_models = model_sales.tail(5).to_dict()

        return {
            "model_stats": model_stats,
            "top_models": top_models,
            "bottom_models": bottom_models,
        }

    def _correlation_analysis(self) -> Dict[str, float]:
        """Calculate correlations between numeric columns.

        Returns:
            Dictionary with correlation matrix.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {}

        corr_matrix = self.df[numeric_cols].corr()
        return corr_matrix.to_dict()

    def _generate_summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Summary string.
        """
        summary_parts = []

        if "date" in self.df.columns:
            date_range = f"{self.df['date'].min()} to {self.df['date'].max()}"
            summary_parts.append(f"Date range: {date_range}")

        if "sales" in self.df.columns:
            total_sales = self.df["sales"].sum()
            summary_parts.append(f"Total sales: {total_sales:,.0f}")

        if "region" in self.df.columns:
            num_regions = self.df["region"].nunique()
            summary_parts.append(f"Number of regions: {num_regions}")

        if "model" in self.df.columns:
            num_models = self.df["model"].nunique()
            summary_parts.append(f"Number of models: {num_models}")

        return "; ".join(summary_parts)

