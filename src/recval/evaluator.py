from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy
import numpy.typing as npt
import pandas

from recval.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from recval.metrics.metric_interface import MetricInterface
from recval.metrics.metrics_utils import get_hit_rank
from recval.utils import get_topk

from .decorators import timeit
from .metrics import MetricFactory


@dataclass
class RecEvaluator:
    """Main class used to evaluate recommender system algorithm

    Attributes:
        metrics (list[str]): metrics used for the evaluation.
        cutoffs (list[int] | npt.NDArray[numpy.int_]): list of cutoffs used to evaluate the recommendations.
    """

    metrics: list[str]
    cutoffs: list[int] | npt.NDArray[numpy.int_]
    max_cutoff: int = field(init=False)
    metrics_objs: list[MetricInterface] = field(init=False, default_factory=lambda: [])

    def __post_init__(self) -> None:
        self.max_cutoff = max(self.cutoffs)
        # convert metrics name in metrics objects
        for metric_name in self.metrics:
            self.metrics_objs.append(MetricFactory.from_name(metric_name))

    def recs_from_scores(
        self,
        scores: npt.NDArray[numpy.float_],
        user_ids: npt.NDArray[numpy.int_] | list[int] | None = None,
    ) -> pandas.DataFrame:
        """Compute recommendations from estimated scores
        Args:
            scores (npt.NDArray[numpy.float_]): estiamted scores matrix, row containing users and columns containing items
            user_ids (npt.NDArray[numpy.int_] | list[int] | None, optional): user ids associated to each row of the estimated score matrix. Defaults to None. #pylint: disable=line-too-long

        Returns:
            pandas.DataFrame: recommendations dataframe
        """
        if user_ids is not None:
            # check number of user_ids match with scores shape
            num_users = len(user_ids)
            row_scores = len(scores)
            if num_users != row_scores:
                raise ValueError(
                    f"Number of user ids do not match with scores shape. # user ids: {num_users}, \
            # rows in scores: {row_scores}"
                )
        else:
            # no userids meaning we create consecutive user_ids with the same shape as scores passed
            logging.warning(
                "user_ids have not been passed as input, creating consecutive user_ids starting from 0"
            )
            user_ids = numpy.arange(scores.shape[0])

        logging.debug("Retrieving topk items")
        top_items, _ = get_topk(scores=scores, k=self.max_cutoff)

        # repreat max_cutoff times the user ids
        users_rep = numpy.repeat(user_ids, self.max_cutoff)
        recs_df = pandas.DataFrame(
            zip(users_rep, top_items.flatten()),
            columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
        )
        return recs_df

    @timeit
    def eval_from_scores(  # pylint: disable=[too-many-arguments,too-many-locals]
        self,
        scores: npt.NDArray[numpy.float_],
        holdout_data: pandas.DataFrame,
        user_ids: npt.NDArray[numpy.int_] | list[int] | None = None,
        verbose: bool = True,
        decimal_precision: int = 4,
    ) -> pandas.DataFrame:
        """Evaluate recommender system from estimated scores

        Args:
            scores (npt.NDArray[numpy.float_]): estiamted scores matrix, row containing users and columns containing items
            holdout_data (pandas.DataFrame): ground truth data against which perform evaluation
            user_ids (npt.NDArray[numpy.int_] | list[int] | None, optional): user ids associated to each row of the estimated score matrix. Defaults to None. #pylint: disable=line-too-long
            verbose (bool, optional): Wheter or not print metric results. Defaults to True.
            decimal_precision (int, optional): precision with which compute evaluation metrics. Defaults to 4.

        Returns:
            pandas.DataFrame: dataframe containing the result metrics for each cutoff.
        """
        recs_df = self.recs_from_scores(scores=scores, user_ids=user_ids)

        metrics_df = self.eval_from_recs(
            recs_df=recs_df,
            holdout_data=holdout_data,
            verbose=verbose,
            decimal_precision=decimal_precision,
        )
        return metrics_df

    @timeit
    def eval_from_recs(
        self,
        recs_df: pandas.DataFrame,
        holdout_data: pandas.DataFrame,
        verbose: bool = True,
        decimal_precision: int = 4,
    ) -> pandas.DataFrame:
        """Evaluate recommender system from recommendations

        Args:
            recs_df (pandas.DataFrame): recommendations df.
            holdout_data (pandas.DataFrame): ground truth data against which perform evaluation.
            verbose (bool, optional): Wheter or not print metric results. Defaults to True.
            decimal_precision (int, optional): precision with which compute evaluation metrics. Defaults to 4.

        Returns:
            pandas.DataFrame: dataframe containing the result metrics for each cutoff.
        """

        if "rank" not in recs_df.columns:
            # adding rank column on recs dataframe
            recs_df["rank"] = numpy.tile(
                numpy.arange(1, self.max_cutoff + 1),
                recs_df[DEFAULT_USER_COL].nunique(),
            )

        metric_name_list = []
        cutoff_list = []
        metric_res_list = []
        for cutoff in self.cutoffs:
            # filter recs on current cutoff
            recs_df_cutoff = recs_df[recs_df["rank"] <= cutoff]
            df_hit, df_hit_count = get_hit_rank(
                ground_truth_df=holdout_data, pred_df=recs_df_cutoff
            )
            for metric in self.metrics_objs:
                res = metric(df_hit, df_hit_count, cutoff)

                # append results
                metric_name_list.append(metric.name_())
                cutoff_list.append(cutoff)
                metric_res_list.append(res)

                if verbose:
                    print(f"{metric.name_()}@{cutoff}: {round(res, decimal_precision)}")
        return pandas.DataFrame(
            zip(metric_name_list, cutoff_list, metric_res_list),
            columns=["metric", "cutoff", "value"],
        )
