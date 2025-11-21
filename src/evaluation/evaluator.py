"""Report quality evaluation module."""

from typing import Dict, Any
import pandas as pd
from pathlib import Path
from config.settings import Settings
from src.utils.logger import app_logger


class ReportEvaluator:
    """Evaluates report quality using various metrics."""

    def __init__(self, settings: Settings):
        """Initialize evaluator.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.logger = app_logger

    def evaluate(
        self,
        report_markdown: str | None = None,
        data: pd.DataFrame | None = None,
        insights: list | None = None,
    ) -> Dict[str, float]:
        """Evaluate report quality.

        Args:
            report_markdown: Path to markdown report file.
            data: Original data DataFrame.
            insights: List of insights from analysis.

        Returns:
            Dictionary with evaluation scores.
        """
        self.logger.info("Evaluating report quality")

        scores = {}

        # Coverage score
        scores["coverage"] = self._calculate_coverage(insights, data)

        # Completeness score
        scores["completeness"] = self._calculate_completeness(report_markdown)

        # Readability score (simple heuristic)
        scores["readability"] = self._calculate_readability(report_markdown)

        # Overall score
        scores["overall"] = (
            scores["coverage"] * 0.4
            + scores["completeness"] * 0.3
            + scores["readability"] * 0.3
        )

        self.logger.info(f"Evaluation scores: {scores}")

        return scores

    def _calculate_coverage(self, insights: list | None, data: pd.DataFrame | None) -> float:
        """Calculate coverage score based on insights and data.

        Args:
            insights: List of insights (dicts with 'insight' field or strings).
            data: Original data DataFrame.

        Returns:
            Coverage score between 0 and 1.
        """
        # Handle empty or None inputs
        if not insights:
            self.logger.warning(f"Coverage: No insights provided (insights: {insights})")
            return 0.0
        
        if data is None or len(data) == 0:
            self.logger.warning(f"Coverage: No data provided")
            return 0.0

        # Filter out None/empty insights
        valid_insights = [insight for insight in insights if insight is not None]
        if not valid_insights:
            self.logger.warning(f"Coverage: All insights are None or empty")
            return 0.0

        # Extract insight text from dict or use string directly
        def _extract_insight_text(insight: Any) -> str:
            """Extract insight text from dict or return string representation."""
            if isinstance(insight, dict):
                # Try common field names - also check evidence and impact fields
                text_parts = []
                if "insight" in insight and insight["insight"]:
                    text_parts.append(str(insight["insight"]))
                if "evidence" in insight and insight["evidence"]:
                    text_parts.append(str(insight["evidence"]))
                if "impact" in insight and insight["impact"]:
                    text_parts.append(str(insight["impact"]))
                if "text" in insight and insight["text"]:
                    text_parts.append(str(insight["text"]))
                
                if text_parts:
                    return " ".join(text_parts)
                # Fallback to string representation
                return str(insight)
            return str(insight) if insight else ""

        # Combine all insight texts for checking
        insight_texts = [_extract_insight_text(insight).lower() for insight in valid_insights]
        insight_texts = [text for text in insight_texts if text.strip()]  # Filter empty strings
        combined_text = " ".join(insight_texts)
        
        if not combined_text.strip():
            self.logger.warning(f"Coverage: No extractable text from insights")
            return 0.0

        # Check if this is the default fallback insight (indicates parsing failure)
        default_fallback_patterns = ["analysis completed", "data analyzed", "n/a"]
        is_fallback = any(
            pattern in combined_text.lower() 
            for pattern in default_fallback_patterns
        ) and len(valid_insights) == 1
        
        if is_fallback:
            self.logger.warning(
                f"Coverage: Detected default fallback insight - structured output parsing likely failed. "
                f"Insight text: {combined_text[:200]}"
            )
            # Return a small score instead of 0 to indicate insights exist but are generic
            return 0.1

        self.logger.info(f"Coverage: Processing {len(insights)} insights")
        self.logger.info(f"Coverage: Combined text (first 500 chars): {combined_text[:500]}")
        if valid_insights:
            self.logger.info(f"Coverage: First insight structure: {valid_insights[0]}")
            self.logger.info(f"Coverage: Extracted text from first insight: {_extract_insight_text(valid_insights[0])}")

        # Check if insights cover key dimensions
        dimensions_covered = 0
        total_dimensions = 4

        # Check temporal coverage - expanded keywords
        temporal_keywords = [
            "trend", "time", "year", "temporal", "over time", "period", 
            "growth", "decline", "increase", "decrease", "change", 
            "2020", "2021", "2022", "2023", "2024", "annual", "monthly"
        ]
        temporal_matched = [kw for kw in temporal_keywords if kw in combined_text]
        if temporal_matched:
            dimensions_covered += 1
            self.logger.info(f"Coverage: Temporal dimension covered (matched: {temporal_matched[:3]})")
        else:
            self.logger.debug(f"Coverage: Temporal dimension NOT covered (checked {len(temporal_keywords)} keywords)")

        # Check regional coverage - expanded keywords
        regional_keywords = [
            "region", "geographic", "location", "area", "country", 
            "territory", "market", "north america", "europe", "asia", 
            "africa", "middle east"
        ]
        regional_matched = [kw for kw in regional_keywords if kw in combined_text]
        if regional_matched:
            dimensions_covered += 1
            self.logger.info(f"Coverage: Regional dimension covered (matched: {regional_matched[:3]})")
        else:
            self.logger.debug(f"Coverage: Regional dimension NOT covered (checked {len(regional_keywords)} keywords)")

        # Check product coverage - expanded keywords
        product_keywords = [
            "model", "product", "vehicle", "car", "brand", "series",
            "bmw", "x3", "x5", "x6", "m3", "m5", "i8", "7 series", "5 series"
        ]
        product_matched = [kw for kw in product_keywords if kw in combined_text]
        if product_matched:
            dimensions_covered += 1
            self.logger.info(f"Coverage: Product dimension covered (matched: {product_matched[:3]})")
        else:
            self.logger.debug(f"Coverage: Product dimension NOT covered (checked {len(product_keywords)} keywords)")

        # Check performance metrics - expanded keywords
        performance_keywords = [
            "sales", "performance", "revenue", "volume", "price", 
            "revenue", "sales volume", "sales_volume", "price_usd"
        ]
        performance_matched = [kw for kw in performance_keywords if kw in combined_text]
        if performance_matched:
            dimensions_covered += 1
            self.logger.info(f"Coverage: Performance dimension covered (matched: {performance_matched[:3]})")
        else:
            self.logger.debug(f"Coverage: Performance dimension NOT covered (checked {len(performance_keywords)} keywords)")

        score = dimensions_covered / total_dimensions
        self.logger.info(f"Coverage: {dimensions_covered}/{total_dimensions} dimensions covered, score: {score}")

        return score

    def _calculate_completeness(self, report_markdown: str | None) -> float:
        """Calculate completeness score.

        Args:
            report_markdown: Path to markdown report.

        Returns:
            Completeness score between 0 and 1.
        """
        if not report_markdown:
            return 0.0

        report_path = Path(report_markdown)
        if not report_path.exists():
            return 0.0

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for required sections
            required_sections = [
                "executive summary",
                "trends",
                "recommendations",
            ]

            sections_found = sum(1 for section in required_sections if section.lower() in content.lower())

            # Check minimum length
            min_length = 1000
            length_score = min(len(content) / min_length, 1.0) if len(content) < min_length else 1.0

            section_score = sections_found / len(required_sections)

            return (section_score * 0.7 + length_score * 0.3)

        except Exception as e:
            self.logger.error(f"Error calculating completeness: {e}")
            return 0.5

    def _calculate_readability(self, report_markdown: str | None) -> float:
        """Calculate readability score using simple heuristics.

        Args:
            report_markdown: Path to markdown report.

        Returns:
            Readability score between 0 and 1.
        """
        if not report_markdown:
            return 0.0

        report_path = Path(report_markdown)
        if not report_path.exists():
            return 0.0

        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple readability metrics
            sentences = content.split(".")
            words = content.split()

            if len(sentences) == 0 or len(words) == 0:
                return 0.0

            # Average sentence length (words per sentence)
            avg_sentence_length = len(words) / len(sentences)

            # Ideal sentence length is 15-20 words
            if 15 <= avg_sentence_length <= 20:
                sentence_score = 1.0
            elif 10 <= avg_sentence_length < 15 or 20 < avg_sentence_length <= 25:
                sentence_score = 0.8
            else:
                sentence_score = 0.6

            # Check for structure (headings, lists)
            structure_score = 0.5
            if "#" in content:  # Has headings
                structure_score += 0.3
            if "-" in content or "*" in content:  # Has lists
                structure_score += 0.2

            return (sentence_score * 0.7 + structure_score * 0.3)

        except Exception as e:
            self.logger.error(f"Error calculating readability: {e}")
            return 0.5

