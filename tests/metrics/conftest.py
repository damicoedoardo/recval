import pandas as pd
import pytest

from recval.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL


@pytest.fixture()
def dummy_recs_gt_cutoff():
    users = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    recs_df = pd.DataFrame(
        zip(users, items), columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
    )

    users_gt = [1, 1, 2, 2, 3, 3, 3]
    items_gt = [1, 5, 2, 3, 7, 8, 9]
    gt_df = pd.DataFrame(
        zip(users_gt, items_gt), columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
    )

    # cutoff=3, 3 recs for each user
    # user 1 -> 1 hit, recall = 0.5, precision=1/3,
    # user 2 -> 0 hit, recall = 0.0, precision=0.0
    # user 3 -> 3 hit, recall = 1.0, precision=1.0

    # user 1 ndcg = 0.5454, map=,
    # user 2 ndcg = 0.0, map=,
    # user 3 ndcg = 1.0, map=,

    return recs_df, gt_df, 3
