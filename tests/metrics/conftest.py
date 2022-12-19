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
    # user 1 -> 1 hit
    # user 2 -> 0 hit
    # user 3 -> 3 hit

    return recs_df, gt_df, 3
