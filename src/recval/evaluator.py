from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from pytictoc import TicToc

from recval.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from recval.metrics.metric_interface import MetricsEnum
from recval.metrics.metrics_utils import get_hit_rank
from recval.utils import get_topk


@dataclass
class RecEvaluator:
    """Main class used to evaluate recommender system algorithm

    Attributes:
        metrics_list (list[str]): metrics used for the evaluation.
        cutoffs_list (list[int] | npt.NDArray[np.int_]): list of cutoffs used to evaluate the recommendations.
    """

    metrics_list: list[str]
    cutoffs_list: list[int] | npt.NDArray[np.int_]
    max_cutoff: int = field(init=False)

    def __post_init__(self) -> None:
        self._check_metrics()
        self.max_cutoff = max(self.cutoffs_list)

    def _check_metrics(self) -> None:
        """Check if metrics passed as input are valid"""
        for metric_class in self.metrics_list:
            assert (
                metric_class in MetricsEnum
            ), f"Metric: {metric_class} is not a valid metric. {MetricsEnum.available_metrics()}"

    def eval_from_scores(  # pylint: disable=[too-many-arguments,too-many-locals]
        self,
        scores: npt.NDArray[np.float_],
        holdout_data: pd.DataFrame,
        user_ids: npt.NDArray[np.int_] | list[int] | None = None,
        verbose: bool = True,
        decimal_precision: int = 4,
    ) -> pd.DataFrame:
        """Evaluate recommender system from estimated scores

        Args:
            scores (npt.NDArray[np.float_]): estiamted scores matrix, row containing users and columns containing items
            holdout_data (pd.DataFrame): ground truth data against which perform evaluation
            user_ids (npt.NDArray[np.int_] | list[int] | None, optional): user ids associated to each row of the estimated score matrix. Defaults to None. #pylint: disable=line-too-long
            verbose (bool, optional): Wheter or not print metric results. Defaults to True.
            decimal_precision (int, optional): precision with which compute evaluation metrics. Defaults to 4.

        Returns:
            pd.DataFrame: dataframe containing the result metrics for each cutoff.
        """
        timer = TicToc()
        if user_ids is not None:
            # check number of user_ids match with scores shape
            assert (num_users := len(user_ids)) == (
                row_scores := scores.shape[0]
            ), f"Number of user ids do not match with scores shape. # user ids: {num_users}, \
            # rows in scores: {row_scores}"
        else:
            # no userids meaning we create consecutive user_ids with the same shape as scores passed
            logging.warning(
                "user_ids have not been passed as input, creating consecutive user_ids starting from 0"
            )
            user_ids = np.arange(scores.shape[0])

        # retrieving recommendations with the highest cutoff passed
        logging.debug("Retrieving topk items")
        timer.tic()
        top_items, _ = get_topk(scores=scores, k=self.max_cutoff)

        # repreat max_cutoff times the user ids
        users_rep = np.repeat(user_ids, self.max_cutoff)
        recs_df = pd.DataFrame(
            zip(users_rep, top_items), columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )

        metric_name_list = []
        cutoff_list = []
        metric_res_list = []
        for cutoff in self.cutoffs_list:
            df_hit, df_hit_count = get_hit_rank(
                ground_truth_df=holdout_data, pred_df=recs_df, cutoff=cutoff
            )
            for metric_clss in self.metrics_list:
                metric = eval(metric_clss)(  # pylint: disable=W0123
                    df_hit=df_hit, df_hit_count=df_hit_count, cutoff=cutoff
                )
                res = metric.compute_metric()

                # append results
                metric_name_list.append(metric_clss)
                cutoff_list.append(cutoff)
                metric_res_list.append(res)

                if verbose:
                    print(f"{metric_clss}@{cutoff}: {round(res, decimal_precision)}")
        res_df = pd.DataFrame(
            zip(metric_name_list, cutoff_list, metric_res_list),
            columns=["metric", "cutoff", "value"],
        )
        logging.debug(timer.toc())
        return res_df
