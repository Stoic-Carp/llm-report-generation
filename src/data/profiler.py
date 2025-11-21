"""Data profiling utilities."""

import pandas as pd
from typing import Dict, Any
from src.utils.logger import app_logger


class DataProfiler:
    """Utility class for data profiling."""

    def __init__(self, df: pd.DataFrame):
        """Initialize profiler.

        Args:
            df: DataFrame to profile.
        """
        self.df = df
        self.logger = app_logger

    def profile(self) -> Dict[str, Any]:
        """Generate comprehensive data profile.

        Returns:
            Dictionary with profile information.
        """
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "memory_usage": self.df.memory_usage(deep=True).to_dict(),
        }

