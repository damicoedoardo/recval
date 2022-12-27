import numpy as np
import pandas as pd

from recval.constants import DEFAULT_USER_COL


def ndcg(
    df_hit: pd.DataFrame,
    df_hit_count: pd.DataFrame,
    cutoff: int,
    col_user: str = DEFAULT_USER_COL,
) -> pd.DataFrame:
    """Normalized Discounted Cumulative Gain (nDCG).
    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    Args:
        df_hit (pd.DataFrame): df containing number of hit and number of ground truth item for each user.
        df_hit_count (pd.DataFrame): df containing number of hit and number of ground truth item for each user.
        cutoff (int): cutoff used to retrieve recommendations
        col_user (str, optional): column containing user_ids. Defaults to `user_id`.

    Returns:
        pd.DataFrame: nDCG for each user.
    """
    df_dcg = df_hit.copy()
    # compute DCG, relevance in this case is always 1
    df_dcg["dcg"] = 1 / np.log2(df_dcg["rank"] + 1)
    # sum up DCG
    df_dcg = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})
    # calculate ideal discounted cumulative gain
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=[col_user], how="right")
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log2(range(2, min(x, cutoff) + 2)))
    )
    df_ndcg["ndcg"] = (df_ndcg["dcg"] / df_ndcg["idcg"]).fillna(0)

    return df_ndcg[[DEFAULT_USER_COL, "ndcg"]]


def average_precision(
    df_hit: pd.DataFrame,
    df_hit_count: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
) -> pd.DataFrame:
    """Average Precision (AP)
    Info: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    Args:
        df_hit (pd.DataFrame): df containing number of hit and number of ground truth item for each user.
        df_hit_count (pd.DataFrame): df containing number of hit and number of ground truth item for each user.
        col_user (str, optional): column containing user_ids. Defaults to `user_id`.

    Returns:
        pd.DataFrame: AP for each user.
    """
    # calculate reciprocal rank of items for each user and sum them up
    df_ap = df_hit.copy()
    df_ap["rr"] = (df_ap.groupby(col_user).cumcount() + 1) / df_ap["rank"]
    df_ap = df_ap.groupby(col_user).agg({"rr": "sum"}).reset_index()

    df_ap = pd.merge(df_ap, df_hit_count, on=col_user, how="right")
    df_ap["rr"] = df_ap["rr"].fillna(0)

    df_ap["avg_prec"] = df_ap["rr"] / (df_ap["actual"] + np.finfo(float).eps)

    return df_ap[[DEFAULT_USER_COL, "avg_prec"]]
