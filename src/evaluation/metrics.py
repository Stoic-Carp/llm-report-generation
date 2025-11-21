"""Evaluation metrics definitions."""

from typing import Dict, Any


class EvaluationMetrics:
    """Collection of evaluation metrics."""

    @staticmethod
    def calculate_accuracy_score(predictions: list, actuals: list) -> float:
        """Calculate accuracy score.

        Args:
            predictions: List of predicted values.
            actuals: List of actual values.

        Returns:
            Accuracy score between 0 and 1.
        """
        if len(predictions) != len(actuals) or len(predictions) == 0:
            return 0.0

        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return correct / len(predictions)

    @staticmethod
    def calculate_precision_recall(
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> Dict[str, float]:
        """Calculate precision and recall.

        Args:
            true_positives: Number of true positives.
            false_positives: Number of false positives.
            false_negatives: Number of false negatives.

        Returns:
            Dictionary with precision and recall scores.
        """
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall}

