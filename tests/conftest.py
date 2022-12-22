import numpy as np
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

    # user 1 ndcg = 0.6131, map=1.0 ,
    # user 2 ndcg = 0.0, map=0.0,
    # user 3 ndcg = 1.0, map=1.0,

    return recs_df, gt_df, 3


@pytest.fixture()
def dummy_userids_scores_holdout():
    # create dummy scores for 3 users over 5 items with ids -> [0,1,2,3,4]
    users = [1, 2, 3]
    scores = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
        ]
    )

    # user 1, holdout_1 = [0]
    # user 2, holdout_2 = [1,2]
    # user 3, holdout_3 = [2,3,4]
    holdout = [
        0,
        1,
        2,
        2,
        3,
        4,
    ]
    u_holdout = [1, 2, 2, 3, 3, 3]
    holdout_df = pd.DataFrame(
        zip(u_holdout, holdout), columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
    )

    return users, scores, holdout_df
