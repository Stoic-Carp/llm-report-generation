"""Data loading and validation module."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import get_settings
from config.schemas import SalesRecord
from src.data.column_mapper import ColumnMapper
from src.utils.logger import app_logger


class BMWDataLoader:
    """Loads and validates BMW sales data."""

    def __init__(self, file_path: Path):
        """Initialize data loader.

        Args:
            file_path: Path to the Excel file containing sales data.
        """
        self.file_path = Path(file_path)
        self.df: pd.DataFrame | None = None
        self.settings = get_settings()
        self.logger = app_logger
        self.column_mapper = ColumnMapper()
        self.column_mapping: Dict[str, str] = {}

    def load(self, use_llm_mapping: bool = True) -> pd.DataFrame:
        """Load Excel file with intelligent column mapping and validation.

        Args:
            use_llm_mapping: Whether to use LLM for intelligent column mapping.

        Returns:
            Loaded and validated DataFrame with standardized column names.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If data validation fails.
        """
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.file_path}")

            # Load raw data
            self.df = pd.read_excel(self.file_path)
            self.logger.info(f"Loaded {len(self.df)} records from {self.file_path}")
            self.logger.info(f"Original columns: {list(self.df.columns)}")

            # Map columns intelligently
            try:
                self.column_mapping = self.column_mapper.map_columns(self.df, use_llm=use_llm_mapping)
                self.logger.info(f"Column mapping completed: {self.column_mapping}")
            except Exception as e:
                self.logger.warning(f"LLM mapping failed ({e}), falling back to heuristic mapping")
                self.column_mapping = self.column_mapper.map_columns(self.df, use_llm=False)
            
            # Apply mapping - always apply if we have mappings
            if self.column_mapping:
                original_cols = list(self.df.columns)
                self.df = self.column_mapper.apply_mapping(self.df, self.column_mapping)
                self.logger.info(f"Applied mapping: {self.column_mapping}")
                self.logger.info(f"Columns before mapping: {original_cols}")
                self.logger.info(f"Columns after mapping: {list(self.df.columns)}")
            else:
                self.logger.warning(
                    f"No column mapping found. Original columns: {list(self.df.columns)}"
                )

            # Validate and normalize (after mapping is complete)
            self._validate()
            self._normalize_data()
            
            self.logger.info(f"Successfully loaded and validated {len(self.df)} records")
            return self.df
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def _validate(self) -> None:
        """Validate data structure and content.

        Raises:
            ValueError: If validation fails.
        """
        if self.df is None:
            raise ValueError("DataFrame is None, cannot validate")

        # Check for essential columns (at least sales or model should exist)
        essential_columns = ["sales", "model"]
        has_essential = any(col in self.df.columns for col in essential_columns)
        
        if not has_essential:
            available_cols = ", ".join(self.df.columns)
            mapping_info = f"Column mapping attempted: {self.column_mapping}" if self.column_mapping else "No mapping was found"
            raise ValueError(
                f"Data must contain at least one essential column (sales, model). "
                f"Available columns after mapping: {available_cols}. "
                f"{mapping_info}. "
                f"Please ensure your data has columns that can be mapped to 'sales' or 'model'."
            )

        # Check for nulls in critical columns (if they exist)
        if "sales" in self.df.columns:
            if self.df["sales"].isna().any():
                if self.settings.data.handle_missing == "error":
                    raise ValueError("Sales column contains null values")
                elif self.settings.data.handle_missing == "drop":
                    self.df = self.df.dropna(subset=["sales"])
                    self.logger.warning("Dropped rows with null sales values")
                elif self.settings.data.handle_missing == "fill":
                    self.df["sales"] = self.df["sales"].fillna(0)
                    self.logger.warning("Filled null sales values with 0")

            # Check for negative sales
            if (self.df["sales"] < 0).any():
                self.logger.warning("Found negative sales values, converting to absolute")
                self.df["sales"] = self.df["sales"].abs()

        # Check for invalid prices (if price column exists)
        if "price" in self.df.columns:
            if (self.df["price"] <= 0).any():
                self.logger.warning("Found non-positive prices, filtering out")
                self.df = self.df[self.df["price"] > 0]

        self.logger.info("Data validation passed")

    def _normalize_data(self) -> None:
        """Normalize and standardize data formats."""
        if self.df is None:
            return

        # Normalize date column (if exists)
        if "date" in self.df.columns:
            try:
                # Try to convert year column to date if it's numeric
                if self.df["date"].dtype in ["int64", "float64"]:
                    # Assume it's a year
                    self.df["date"] = pd.to_datetime(self.df["date"].astype(str) + "-01-01")
                else:
                    self.df["date"] = pd.to_datetime(self.df["date"])
            except Exception as e:
                self.logger.warning(f"Could not normalize date column: {e}")

        # Normalize region column (if exists) - standardize to uppercase
        if "region" in self.df.columns:
            self.df["region"] = self.df["region"].astype(str).str.upper()

        # Normalize model column (if exists) - ensure string type
        if "model" in self.df.columns:
            self.df["model"] = self.df["model"].astype(str)

        # Ensure numeric columns are numeric
        for col in ["sales", "price"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def get_summary(self) -> Dict[str, Any]:
        """Get basic summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        summary = {
            "total_records": len(self.df),
            "columns": list(self.df.columns),
            "column_mapping": self.column_mapping,
        }

        if "date" in self.df.columns:
            summary["date_range"] = (
                self.df["date"].min(),
                self.df["date"].max(),
            )
        if "sales" in self.df.columns:
            summary["total_sales"] = int(self.df["sales"].sum())
        if "price" in self.df.columns:
            summary["avg_price"] = float(self.df["price"].mean())
        if "model" in self.df.columns:
            summary["unique_models"] = self.df["model"].nunique()
        if "region" in self.df.columns:
            summary["unique_regions"] = self.df["region"].nunique()

        return summary

