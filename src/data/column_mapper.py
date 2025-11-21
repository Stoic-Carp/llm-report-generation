"""Intelligent column mapping using LLM to understand data schemas."""

import pandas as pd
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from config.settings import get_settings
from src.llm.llm_factory import LLMFactory
from src.llm.utils import extract_llm_content, parse_json_response
from src.utils.logger import app_logger


class ColumnMapper:
    """Maps actual column names to expected schema using LLM intelligence."""

    # Expected column mappings
    EXPECTED_COLUMNS = {
        "date": ["date", "year", "time", "timestamp", "period"],
        "region": ["region", "location", "area", "geography", "market"],
        "model": ["model", "product", "car_model", "vehicle", "name"],
        "sales": ["sales", "sales_volume", "units_sold", "quantity", "volume", "salesvolume"],
        "price": ["price", "price_usd", "cost", "amount", "value", "priceusd"],
    }

    def __init__(self):
        """Initialize column mapper."""
        self.settings = get_settings()
        self.logger = app_logger
        self.mapping: Dict[str, str] = {}

    def map_columns(self, df: pd.DataFrame, use_llm: bool = True) -> Dict[str, str]:
        """Map actual columns to expected schema.

        Args:
            df: DataFrame with actual columns.
            use_llm: Whether to use LLM for intelligent mapping.

        Returns:
            Dictionary mapping expected column names to actual column names.
        """
        actual_columns = list(df.columns)
        self.logger.info(f"Mapping columns: {actual_columns}")

        # First try heuristic mapping (always works)
        heuristic_mapping = self._heuristic_map(actual_columns)
        
        # If LLM is requested and heuristic found some mappings, try LLM to improve
        if use_llm and heuristic_mapping:
            try:
                llm_mapping = self._llm_map_columns(actual_columns, df.head(3))
                # Merge LLM mapping with heuristic (LLM takes precedence)
                merged_mapping = {**heuristic_mapping, **llm_mapping}
                self.mapping = merged_mapping
                self.logger.info(f"Using LLM-enhanced mapping: {merged_mapping}")
                return merged_mapping
            except Exception as e:
                self.logger.warning(f"LLM mapping failed: {e}, using heuristic mapping")
                self.mapping = heuristic_mapping
                return heuristic_mapping
        else:
            self.mapping = heuristic_mapping
            return heuristic_mapping

    def _llm_map_columns(self, actual_columns: list[str], sample_data: pd.DataFrame) -> Dict[str, str]:
        """Use LLM to intelligently map columns.

        Args:
            actual_columns: List of actual column names.
            sample_data: Sample DataFrame rows for context.

        Returns:
            Dictionary mapping expected to actual column names.
        """
        try:
            llm = LLMFactory.get_default_llm()
        except Exception as e:
            self.logger.warning(f"Could not create LLM for column mapping: {e}")
            raise

        system_prompt = """You are a data schema expert. Map the provided column names to a standard schema.
Return a JSON object mapping expected column names to actual column names.

Expected columns:
- date: temporal/date information
- region: geographical location
- model: product/model name
- sales: sales volume/quantity
- price: price/cost information

Return ONLY valid JSON with this structure:
{
  "date": "actual_column_name_or_null",
  "region": "actual_column_name_or_null",
  "model": "actual_column_name_or_null",
  "sales": "actual_column_name_or_null",
  "price": "actual_column_name_or_null"
}

Use null if no suitable column exists."""

        user_prompt = f"""Map these column names to the expected schema:

Actual columns: {actual_columns}

Sample data (first 3 rows):
{sample_data.to_string()}

Provide the mapping as JSON."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = llm.invoke(messages)
            content = extract_llm_content(response)
            
            # Try robust JSON parsing
            try:
                mapping = parse_json_response(content)
            except ValueError:
                # Fallback to JsonOutputParser
                parser = JsonOutputParser()
                mapping = parser.parse(content)

            # Clean up mapping (remove null values, convert to lowercase for matching)
            cleaned_mapping = {}
            for expected, actual in mapping.items():
                if actual and actual.lower() != "null":
                    # Find case-insensitive match
                    actual_lower = actual.lower()
                    matched_col = next(
                        (col for col in actual_columns if col.lower() == actual_lower),
                        None,
                    )
                    if matched_col:
                        cleaned_mapping[expected] = matched_col

            self.logger.info(f"LLM mapping result: {cleaned_mapping}")
            return cleaned_mapping

        except Exception as e:
            self.logger.error(f"Error in LLM column mapping: {e}")
            raise

    def _heuristic_map(self, actual_columns: list[str]) -> Dict[str, str]:
        """Heuristic mapping based on column name similarity.

        Args:
            actual_columns: List of actual column names.

        Returns:
            Dictionary mapping expected to actual column names.
        """
        mapping = {}
        actual_lower = [col.lower().replace("_", "").replace("-", "") for col in actual_columns]

        for expected, synonyms in self.EXPECTED_COLUMNS.items():
            # Skip if already mapped
            if expected in mapping:
                continue
                
            for synonym in synonyms:
                synonym_clean = synonym.lower().replace("_", "").replace("-", "")
                
                # Try exact match (case-insensitive, ignoring underscores/hyphens)
                for i, col_lower in enumerate(actual_lower):
                    if synonym_clean == col_lower:
                        mapping[expected] = actual_columns[i]
                        break
                
                if expected in mapping:
                    break
                    
                # Try partial match (substring)
                if expected not in mapping:
                    for i, col_lower in enumerate(actual_lower):
                        if synonym_clean in col_lower or col_lower in synonym_clean:
                            mapping[expected] = actual_columns[i]
                            break
                    
                    if expected in mapping:
                        break

        self.logger.info(f"Heuristic mapping result: {mapping}")
        return mapping

    def apply_mapping(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply column mapping to DataFrame.

        Args:
            df: Original DataFrame.
            mapping: Mapping dictionary (expected -> actual).

        Returns:
            DataFrame with renamed columns.
        """
        rename_dict = {v: k for k, v in mapping.items()}
        df_mapped = df.rename(columns=rename_dict)
        return df_mapped

