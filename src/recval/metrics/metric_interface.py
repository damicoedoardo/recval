from dataclasses import dataclass, field

import pandas
from strenum import LowercaseStrEnum

from recval.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from recval.utils import MetaEnum


@dataclass
class MetricInterface:
    """Interface for an evaluation metric.

    Attributes:
        col_user (str, optional): column name for user. Defaults to user_id.
        col_item (str, optional): column name for item. Defaults to item_id.
    """

    col_user: str = DEFAULT_USER_COL
    col_item: str = DEFAULT_ITEM_COL
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = self.name_()

    @staticmethod
    def name_() -> str:
        """Return the name of the metric"""
        return "Abstract Metric"

    def __call__(
        self, df_hit: pandas.DataFrame, df_hit_count: pandas.DataFrame, cutoff: int
    ) -> float:
        """
        Compute value of the metric

        Attributes:
            df_hit (pandas.DataFrame): dataframe of recommendation hits, sorted by col_user and rank,
            df_hit_count (pandas.DataFrame): dataframe of hit counts vs actual relevant items per user
            cutoff (int): cutoff used to compute the recommendation
        """
        return self.compute_metric(
            df_hit=df_hit, df_hit_count=df_hit_count, cutoff=cutoff
        )

    def compute_metric(
        self, df_hit: pandas.DataFrame, df_hit_count: pandas.DataFrame, cutoff: int
    ) -> float:
        """
        Compute value of the metric

        Attributes:
            df_hit (pandas.DataFrame): dataframe of recommendation hits, sorted by col_user and rank,
            df_hit_count (pandas.DataFrame): dataframe of hit counts vs actual relevant items per user
            cutoff (int): cutoff used to compute the recommendation
        """
        raise NotImplementedError


class MetricsEnum(LowercaseStrEnum, metaclass=MetaEnum):  # type: ignore
    """Enumerator containing available metrics"""

    # Accuracy
    RECALL = "Recall"
    PRECISION = "Precision"
    F1_SCORE = "F1Score"
    # Ranking
    NDCG = "NDCG"
    MAP = "MAP"

    @classmethod
    def available_metrics(cls) -> None:
        """Print available metrics"""
        metrics = [metric.value for metric in cls]
        print("Available Metrics:\n")
        for metric_name in metrics:
            print("- " + metric_name)
