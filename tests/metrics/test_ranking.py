import numpy as np
import pytest

from recval.metrics.metrics_utils import get_hit_rank
from recval.metrics.ranking import average_precision, ndcg


def test_ndcg(dummy_recs_gt_cutoff):
    recs_df, gt_df, cutoff = dummy_recs_gt_cutoff
    df_hit, df_hit_count = get_hit_rank(ground_truth_df=gt_df, pred_df=recs_df)
    ndcg_df = ndcg(df_hit=df_hit, df_hit_count=df_hit_count, cutoff=cutoff)
    assert ndcg_df["ndcg"].values == pytest.approx(
        np.array([1 / (1 + (1 / np.log2(3))), 0.0, 1.0])
    )


def test_average_precision(dummy_recs_gt_cutoff):
    recs_df, gt_df, _ = dummy_recs_gt_cutoff
    df_hit, df_hit_count = get_hit_rank(ground_truth_df=gt_df, pred_df=recs_df)
    ap_df = average_precision(df_hit=df_hit, df_hit_count=df_hit_count)
    assert ap_df["avg_prec"].values == pytest.approx(np.array([0.5, 0.0, 1.0]))
