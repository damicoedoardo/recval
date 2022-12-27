import pandas

from recval.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL


def get_hit_rank(
    ground_truth_df: pandas.DataFrame,
    pred_df: pandas.DataFrame,
    col_user: str = DEFAULT_USER_COL,
    col_item: str = DEFAULT_ITEM_COL,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Compute hit and hit ranks for each user

    Args:
        ground_truth_df (pandas.DataFrame): Ground Truth DataFrame
        pred_df (pandas.DataFrame): Prediction DataFrame
        col_user (str, optional): column name for user. Defaults to user_id.
        col_item (str, optional): column name for item. Defaults to item_id.
    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:DataFrame of recommendation hits, sorted by col_user and rank,
        DataFrame of hit counts vs actual relevant items per user,
    """

    # Make sure the prediction and true data frames have the same set of users
    common_users = set(ground_truth_df[col_user]).intersection(set(pred_df[col_user]))
    # TODO: This can be made a warning
    assert (
        len(common_users)
        == ground_truth_df[col_user].nunique()
        == pred_df[col_user].nunique()
    ), f"Missing predictions for some users: ground truth users: {ground_truth_df[col_user].nunique()}, \
    predicted users: {pred_df[col_user].nunique()}, common_users: {len(common_users)}"

    df_hit = pandas.merge(pred_df, ground_truth_df, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    if len(df_hit) == 0:
        raise ValueError("No hits found in prediction data.")

    # count the number of hits vs actual relevant items per user
    df_hit_count = pandas.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        ground_truth_df.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
        how="right",
    )

    df_hit_count["hit"] = df_hit_count["hit"].fillna(0)

    return df_hit, df_hit_count
