import numpy as np
import numpy.typing as npt


def get_topk(
    scores: npt.NDArray[np.float_],
    k: int,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
    """Retrieve the top-k items and scores from a score matrix

    Args:
        scores (np.ndarray): user-item scores matix
        k (int): number of top items and scores to retrieve

    Returns:
        tuple[np.ndarray, np.ndarray]: top_items and top_scores
    """
    # check scores is a two dimensional array
    assert (
        adim := scores.ndim
    ) == 2, f"scores has to be a 2-dimensional array, passed is {adim}-dimensional"

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    # top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    test_user_idx = np.arange(scores.shape[0])[:, None]
    top_scores = scores[test_user_idx, top_items]

    # sort top k items
    sort_ind = np.argsort(-top_scores)
    top_items = top_items[test_user_idx, sort_ind]
    top_scores = top_scores[test_user_idx, sort_ind]

    return top_items, top_scores
