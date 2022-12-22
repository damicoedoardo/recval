from recval.metrics.accuracy import f1_score, precision, recall
from recval.metrics.metric_interface import MetricInterface
from recval.metrics.ranking import average_precision, ndcg


class NDCG(MetricInterface):  # pylint: disable=too-few-public-methods
    """Normalized Discounted Cumulative Gain (nDCG).
    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """

    def _name(self) -> str:
        return "ndcg"

    def compute_metric(self) -> float:
        user_metric_df = ndcg(
            df_hit=self.df_hit,
            df_hit_count=self.df_hit_count,
            cutoff=self.cutoff,
        )
        return float(user_metric_df[self.name].mean())


class MAP(MetricInterface):  # pylint: disable=too-few-public-methods
    """Mean Average Precision (MAP).
    Info: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    """

    def _name(self) -> str:
        return "MAP"

    def compute_metric(self) -> float:
        user_metric_df = average_precision(
            df_hit=self.df_hit,
            df_hit_count=self.df_hit_count,
        )
        return float(user_metric_df["avg_prec"].mean())


class Recall(MetricInterface):  # pylint: disable=too-few-public-methods
    """Recall.
    Info: https://en.wikipedia.org/wiki/Precision_and_recall
    """

    def _name(self) -> str:
        return "recall"

    def compute_metric(self) -> float:
        user_metric_df = recall(
            df_hit_count=self.df_hit_count,
        )
        return float(user_metric_df[self.name].mean())


class Precision(MetricInterface):  # pylint: disable=too-few-public-methods
    """Precision.
    Info: https://en.wikipedia.org/wiki/Precision_and_recall
    """

    def _name(self) -> str:
        return "precision"

    def compute_metric(self) -> float:
        user_metric_df = precision(
            df_hit_count=self.df_hit_count,
            cutoff=self.cutoff,
        )
        return float(user_metric_df[self.name].mean())


class F1Score(MetricInterface):  # pylint: disable=too-few-public-methods
    """F1 Score
    Info: https://en.wikipedia.org/wiki/F-score
    """

    def _name(self) -> str:
        return "f1_score"

    def compute_metric(self) -> float:
        user_metric_df = f1_score(
            df_hit_count=self.df_hit_count,
            cutoff=self.cutoff,
        )
        return float(user_metric_df[self.name].mean())
