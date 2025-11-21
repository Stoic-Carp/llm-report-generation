"""Data introspection tools for LLM agents.

These tools enable agents to discover ground truth about data structure,
quality, and contents, minimizing errors and hallucinations.

All tools are decorated with @tool from langchain_core.tools, making them
compatible with LangChain agents created via langchain.agents.create_agent.
The tools are integrated into the workflow graph at key stages where agents
need to validate data, check statistics, or inspect data structure.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool


def _load_dataframe(file_path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse potential datetime series with heuristics to avoid warnings."""
    if series.dropna().empty:
        return pd.to_datetime(series, errors="coerce", cache=True)

    non_null = series.dropna()

    if pd.api.types.is_integer_dtype(non_null):
        return pd.to_datetime(series, errors="coerce", format="%Y", cache=True)

    if pd.api.types.is_float_dtype(non_null) and (non_null % 1 == 0).all():
        return pd.to_datetime(
            series.astype("Int64"), errors="coerce", format="%Y", cache=True
        )

    str_series = non_null.astype(str)
    patterns = [
        (r"^\d{4}$", "%Y"),
        (r"^\d{4}-\d{2}$", "%Y-%m"),
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
        (r"^\d{2}/\d{2}/\d{4}$", "%m/%d/%Y"),
        (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),
    ]

    for pattern, fmt in patterns:
        if str_series.str.match(pattern).all():
            return pd.to_datetime(
                series.astype(str), format=fmt, errors="coerce", cache=True
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(
            series,
            errors="coerce",
            infer_datetime_format=True,
            cache=True,
        )


@tool
def inspect_data_columns(file_path: str, nrows: int = 5) -> Dict[str, Any]:
    """
    Tool 0: Inspect actual columns and data types from a data file.

    Reads the first few rows to understand data structure without loading
    the entire dataset into memory.

    Args:
        file_path: Path to the data file (CSV or Excel)
        nrows: Number of rows to read for inspection (default: 5)

    Returns:
        Dictionary with column information including names, types, and samples.
    """
    df = _load_dataframe(file_path, nrows=nrows)

    column_info = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_values": {col: df[col].head(3).tolist() for col in df.columns},
        "total_columns": len(df.columns),
        "shape_preview": f"{len(df)} rows × {len(df.columns)} columns (preview only)",
    }

    return column_info


@tool
def check_data_quality(file_path: str) -> Dict[str, Any]:
    """
    Tool 1: Check data quality metrics.

    Analyzes missing values, duplicates, and basic quality indicators
    to help agents understand data limitations and issues.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary with quality metrics including missing values, duplicates,
        and potential outliers.
    """
    df = _load_dataframe(file_path)

    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outlier_count = outlier_mask.sum()
        if outlier_count > 0:
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": round(outlier_count / len(df) * 100, 2),
            }

    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing_counts.to_dict(),
        "missing_percentage": missing_percentages.to_dict(),
        "columns_with_missing": missing_counts[missing_counts > 0].index.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "potential_outliers": outliers,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "quality_summary": {
            "is_clean": missing_counts.sum() == 0 and df.duplicated().sum() == 0,
            "completeness_score": round((1 - missing_counts.sum() / df.size) * 100, 2),
        },
    }

    return quality_report


@tool
def get_statistical_summary(file_path: str) -> Dict[str, Any]:
    """
    Tool 2: Get comprehensive statistical summary.

    Generates descriptive statistics, distributions, and correlations
    for numeric columns to provide evidence-based insights.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary with statistical metrics including mean, median, std,
        correlations, and distribution information.
    """
    df = _load_dataframe(file_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {"error": "No numeric columns found in dataset", "numeric_columns": []}

    desc_stats = df[numeric_cols].describe()

    correlations = df[numeric_cols].corr()

    # Find strongest correlations (excluding self-correlations)
    strong_correlations = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:  # Avoid duplicates and self-correlations
                corr_value = correlations.loc[col1, col2]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append(
                        {
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(corr_value, 3),
                            "strength": "strong positive"
                            if corr_value > 0.7
                            else "strong negative"
                            if corr_value < -0.7
                            else "moderate",
                        }
                    )

    skewness = {col: round(df[col].skew(), 3) for col in numeric_cols}

    result = {
        "numeric_columns": numeric_cols,
        "statistics": {
            col: {
                "count": int(desc_stats.loc["count", col]),
                "mean": round(desc_stats.loc["mean", col], 2),
                "std": round(desc_stats.loc["std", col], 2),
                "min": round(desc_stats.loc["min", col], 2),
                "25%": round(desc_stats.loc["25%", col], 2),
                "median": round(desc_stats.loc["50%", col], 2),
                "75%": round(desc_stats.loc["75%", col], 2),
                "max": round(desc_stats.loc["max", col], 2),
                "skewness": skewness[col],
                "distribution": "right-skewed"
                if skewness[col] > 0.5
                else "left-skewed"
                if skewness[col] < -0.5
                else "approximately normal",
            }
            for col in numeric_cols
        },
        "correlations": correlations.round(3).to_dict(),
        "strong_correlations": strong_correlations,
        "summary": {
            "total_numeric_columns": len(numeric_cols),
            "has_strong_correlations": len(strong_correlations) > 0,
        },
    }

    return result


@tool
def get_categorical_values(
    file_path: str, column: Optional[str] = None, limit: int = 20
) -> Dict[str, Any]:
    """
    Tool 3: Get unique values and frequencies for categorical columns.

    Prevents agents from hallucinating category names by providing
    actual unique values and their frequencies.

    Args:
        file_path: Path to the data file
        column: Specific column to analyze (if None, analyzes all categorical)
        limit: Maximum number of top values to return per column

    Returns:
        Dictionary with unique values, counts, and frequencies for
        categorical columns.
    """
    df = _load_dataframe(file_path)

    if column:
        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found",
                "available_columns": df.columns.tolist(),
            }

        value_counts = df[column].value_counts().head(limit)

        return {
            "column": column,
            "unique_count": int(df[column].nunique()),
            "null_count": int(df[column].isnull().sum()),
            "top_values": value_counts.to_dict(),
            "top_values_percentage": (value_counts / len(df) * 100).round(2).to_dict(),
            "sample_values": df[column].dropna().unique()[:10].tolist(),
            "cardinality": "high"
            if df[column].nunique() > 50
            else "medium"
            if df[column].nunique() > 10
            else "low",
        }

    else:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical_cols:
            return {
                "message": "No categorical columns found",
                "categorical_columns": [],
            }

        result = {"categorical_columns": categorical_cols, "column_details": {}}

        for col in categorical_cols:
            value_counts = df[col].value_counts().head(limit)
            result["column_details"][col] = {
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "top_values": value_counts.to_dict(),
                "sample_values": df[col].dropna().unique()[:5].tolist(),
                "cardinality": "high"
                if df[col].nunique() > 50
                else "medium"
                if df[col].nunique() > 10
                else "low",
            }

        return result


@tool
def detect_date_columns(file_path: str) -> Dict[str, Any]:
    """
    Tool 4: Automatically detect and analyze date/time columns.

    Identifies temporal columns, extracts date ranges, and determines
    granularity to help agents understand time-series data.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary with date column information including ranges,
        granularity, and temporal patterns.
    """
    df = _load_dataframe(file_path)

    date_columns = []
    date_info = {}

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
        else:
            try:
                parsed = _parse_datetime_series(df[col])
                if parsed.notna().sum() / len(df) > 0.5:
                    date_columns.append(col)
                    df[col] = parsed
            except Exception:
                continue

    if not date_columns:
        return {
            "message": "No date columns detected",
            "date_columns": [],
            "recommendation": "Check if date columns need manual parsing",
        }

    # Analyze each date column
    for col in date_columns:
        dates = df[col].dropna()

        if len(dates) == 0:
            continue

        min_date = dates.min()
        max_date = dates.max()
        date_range_days = (max_date - min_date).days

        unique_dates = dates.nunique()
        avg_gap = date_range_days / unique_dates if unique_dates > 1 else 0

        if avg_gap < 1:
            granularity = "hourly or finer"
        elif avg_gap < 2:
            granularity = "daily"
        elif avg_gap < 8:
            granularity = "weekly"
        elif avg_gap < 35:
            granularity = "monthly"
        elif avg_gap < 100:
            granularity = "quarterly"
        else:
            granularity = "yearly"

        year_range = sorted(dates.dt.year.unique().tolist())
        month_dist = dates.dt.month.value_counts().sort_index().to_dict()

        date_info[col] = {
            "min_date": str(min_date),
            "max_date": str(max_date),
            "date_range_days": date_range_days,
            "unique_dates": int(unique_dates),
            "granularity": granularity,
            "years_covered": year_range,
            "year_range": f"{year_range[0]}-{year_range[-1]}" if year_range else "N/A",
            "month_distribution": month_dist,
            "null_count": int(df[col].isnull().sum()),
            "has_time_component": bool(
                dates.dt.hour.sum() > 0 or dates.dt.minute.sum() > 0
            ),
        }

    return {
        "date_columns": date_columns,
        "total_date_columns": len(date_columns),
        "date_column_details": date_info,
        "recommendation": f"Use '{date_columns[0]}' for time-series analysis"
        if date_columns
        else None,
    }


@tool
def analyze_relationships(file_path: str) -> Dict[str, Any]:
    """
    Tool 5: Analyze relationships between columns.

    Detects potential grouping keys, hierarchical relationships,
    and suggests useful aggregations for analysis.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary with relationship information including potential
        keys, hierarchies, and aggregation suggestions.
    """
    df = _load_dataframe(file_path)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    potential_keys = []
    for col in categorical_cols:
        cardinality = df[col].nunique()
        cardinality_ratio = cardinality / len(df)

        if cardinality_ratio < 0.5 and cardinality > 1:  # Not too unique, not constant
            potential_keys.append(
                {
                    "column": col,
                    "unique_values": int(cardinality),
                    "cardinality_ratio": round(cardinality_ratio, 3),
                    "suitable_for": "groupby, aggregation, categorical analysis",
                }
            )

    hierarchies = []
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1 :]:
            # Check if col1 is a parent of col2 (one-to-many relationship)
            grouped = df.groupby(col1)[col2].nunique()
            if (grouped > 1).any() and (grouped.mean() > 1.5):
                # col1 likely contains col2
                hierarchies.append(
                    {
                        "parent": col1,
                        "child": col2,
                        "relationship": "one-to-many",
                        "avg_children_per_parent": round(grouped.mean(), 2),
                    }
                )

    aggregation_suggestions = []
    if potential_keys and numeric_cols:
        for key in potential_keys[:3]:  # Top 3 keys
            for num_col in numeric_cols[:3]:  # Top 3 numeric columns
                aggregation_suggestions.append(
                    {
                        "groupby": key["column"],
                        "aggregate": num_col,
                        "suggested_operations": ["sum", "mean", "count"],
                        "use_case": f"Analyze {num_col} by {key['column']}",
                    }
                )

    numeric_relationships = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.5:
                    numeric_relationships.append(
                        {
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(corr, 3),
                            "relationship_type": "positive" if corr > 0 else "negative",
                            "strength": "strong" if abs(corr) > 0.7 else "moderate",
                        }
                    )

    result = {
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "potential_grouping_keys": potential_keys,
        "hierarchical_relationships": hierarchies,
        "aggregation_suggestions": aggregation_suggestions[:10],  # Limit to top 10
        "numeric_relationships": numeric_relationships,
        "recommendations": {
            "best_groupby_column": potential_keys[0]["column"]
            if potential_keys
            else None,
            "primary_numeric_columns": numeric_cols[:3] if numeric_cols else [],
            "has_hierarchies": len(hierarchies) > 0,
            "suitable_for_pivot": len(potential_keys) >= 2 and len(numeric_cols) >= 1,
        },
    }

    return result
