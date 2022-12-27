from typing import Any

from .metric_interface import MetricInterface
from .metrics import MAP, NDCG, F1Score, Precision, Recall


class MetricFactory:
    """
    Factory for creating Metric classes from names.
    """

    METRICS = {
        NDCG.name_(): NDCG,
        MAP.name_(): MAP,
        Recall.name_(): Recall,
        Precision.name_(): Precision,
        F1Score.name_(): F1Score,
    }

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> MetricInterface:
        """
        Creates a metric class from its name.
        """
        if name not in cls.METRICS:
            raise ValueError(
                f"Metric {name} is not supported, available ones are: {list(cls.METRICS.keys())}"
            )
        return cls.METRICS[name](**kwargs)
