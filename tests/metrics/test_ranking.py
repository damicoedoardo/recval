import numpy as np
import pytest

from recval.metrics.metrics_utils import get_hit_rank
from recval.metrics.ranking import ndcg


def test_ndcg(dummy_recs_gt_cutoff):
    recs_df, gt_df, cutoff = dummy_recs_gt_cutoff
    df_hit, df_hit_count = get_hit_rank(
        ground_truth_df=gt_df, pred_df=recs_df, cutoff=cutoff
    )
    ndcg_df = ndcg(df_hit=df_hit, df_hit_count=df_hit_count, cutoff=cutoff)
    assert ndcg_df["ndcg"].values == pytest.approx(
        np.array([1 / (1 + (1 / np.log2(3))), 0.0, 1.0])
    )
