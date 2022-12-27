# get_hit_rank
import numpy as np

from recval.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL
from recval.metrics.metrics_utils import get_hit_rank


def test_get_hit_rank(dummy_recs_gt_cutoff):
    recs_df, gt_df, _ = dummy_recs_gt_cutoff
    df_hit, df_hit_count = get_hit_rank(ground_truth_df=gt_df, pred_df=recs_df)
    expected_user_id = np.array([1, 3, 3, 3])
    expected_rank = np.array([1, 1, 2, 3])
    expected_item_id = np.array([1, 7, 8, 9])

    assert (df_hit[DEFAULT_USER_COL].values == expected_user_id).all()
    assert (df_hit["rank"].values == expected_rank).all()
    assert (df_hit[DEFAULT_ITEM_COL].values == expected_item_id).all()

    expected_user_id = np.array([1, 2, 3])
    expected_hit = np.array([1, 0, 3])
    expected_actual = np.array([2, 2, 3])

    assert (df_hit_count[DEFAULT_USER_COL].values == expected_user_id).all()
    assert (df_hit_count["hit"].values == expected_hit).all()
    assert (df_hit_count["actual"].values == expected_actual).all()
