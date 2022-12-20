import numpy as np
import pandas as pd

from recval.constants import DEFAULT_USER_COL


def recall(
    df_hit_count: pd.DataFrame,
    col_user: str = DEFAULT_USER_COL,
) -> pd.DataFrame:
    """Compute Recall for each user.

    Args:
        df_hit_count (pd.DataFrame): df containing number of hit and number of ground truth item for each user.
        col_user (str, optional): column containing user_ids. Defaults to `user_id`.

    Returns:
        pd.DataFrame: dataframa containing recall for each user.
    """
    rec = df_hit_count["hit"] / df_hit_count["actual"]
    users = df_hit_count[col_user]
    return pd.DataFrame(zip(users, rec), columns=[col_user, "recall"])


def precision(
    df_hit_count: pd.DataFrame,
    cutoff: int,
    col_user: str = DEFAULT_USER_COL,
) -> pd.DataFrame:
    """Compute Precision for each user.

    Args:
        df_hit_count (pd.DataFrame): df containing number of hit and number of ground truth item for each user.
        cutoff (int): cutoff used to retrieve recommendations
        col_user (str, optional): column containing user_id. Defaults to user_id.

    Returns:
        pd.DataFrame: dataframe containing precision for each user.
    """
    prec = df_hit_count["hit"] / cutoff
    users = df_hit_count[col_user]
    return pd.DataFrame(zip(users, prec), columns=[col_user, "precision"])


def f1_score(
    df_hit_count: pd.DataFrame, cutoff: int, col_user: str = DEFAULT_USER_COL
) -> pd.DataFrame:
    """compute F1_score for each user

    Args:
        df_hit_count (pd.DataFrame): df containing number of hit and ground truth item for each user.
        cutoff (int): cutoff used to retrieve recommendations
        col_user (str, optional): column containing user_id. Defaults to user_id.

    Returns:
        pd.DataFrame: dataframe containing precision for each user.
    """
    rec = recall(df_hit_count, col_user=col_user)
    prec = precision(df_hit_count, cutoff, col_user=col_user)
    prec_rec_df = pd.merge(prec, rec, on=DEFAULT_USER_COL)
    prec_rec_df["f1_score"] = (
        2
        * (prec_rec_df["precision"] * prec_rec_df["recall"])
        / ((prec_rec_df["precision"] + prec_rec_df["recall"]) + np.finfo(float).eps)
    )
    return prec_rec_df[[DEFAULT_USER_COL, "f1_score"]]
