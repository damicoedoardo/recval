import pandas

from recval.metrics.accuracy import f1_score, precision, recall
from recval.metrics.metric_interface import MetricInterface
from recval.metrics.ranking import average_precision, ndcg


class NDCG(MetricInterface):  # pylint: disable=too-few-public-methods
    """Normalized Discounted Cumulative Gain (nDCG).
    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """

    @staticmethod
    def name_() -> str:
        return "ndcg"

    def compute_metric(
        self, df_hit: pandas.DataFrame, df_hit_count: pandas.DataFrame, cutoff: int
    ) -> float:
        user_metric_df = ndcg(
            df_hit=df_hit,
            df_hit_count=df_hit_count,
            cutoff=cutoff,
        )
        return float(user_metric_df[self.name].mean())


class MAP(MetricInterface):  # pylint: disable=too-few-public-methods
    """Mean Average Precision (MAP).
    Info: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    """

    @staticmethod
    def name_() -> str:
        return "map"

    def compute_metric(
        self, df_hit: pandas.DataFrame, df_hit_count: pandas.DataFrame, _: int
    ) -> float:
        user_metric_df = average_precision(
            df_hit=df_hit,
            df_hit_count=df_hit_count,
        )
        return float(user_metric_df["avg_prec"].mean())


class Recall(MetricInterface):  # pylint: disable=too-few-public-methods
    """Recall.
    Info: https://en.wikipedia.org/wiki/Precision_and_recall
    """

    @staticmethod
    def name_() -> str:
        return "recall"

    def compute_metric(
        self, _: pandas.DataFrame, df_hit_count: pandas.DataFrame, __: int
    ) -> float:
        user_metric_df = recall(
            df_hit_count=df_hit_count,
        )
        return float(user_metric_df[self.name].mean())


class Precision(MetricInterface):  # pylint: disable=too-few-public-methods
    """Precision.
    Info: https://en.wikipedia.org/wiki/Precision_and_recall
    """

    @staticmethod
    def name_() -> str:
        return "precision"

    def compute_metric(
        self, _: pandas.DataFrame, df_hit_count: pandas.DataFrame, cutoff: int
    ) -> float:
        user_metric_df = precision(
            df_hit_count=df_hit_count,
            cutoff=cutoff,
        )
        return float(user_metric_df[self.name].mean())


class F1Score(MetricInterface):  # pylint: disable=too-few-public-methods
    """F1 Score
    Info: https://en.wikipedia.org/wiki/F-score
    """

    @staticmethod
    def name_() -> str:
        return "f1_score"

    def compute_metric(
        self, _: pandas.DataFrame, df_hit_count: pandas.DataFrame, cutoff: int
    ) -> float:
        user_metric_df = f1_score(
            df_hit_count=df_hit_count,
            cutoff=cutoff,
        )
        return float(user_metric_df[self.name].mean())
