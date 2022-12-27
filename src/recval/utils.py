import heapq
import logging

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .decorators import timeit


@timeit
def get_topk(
    scores: npt.NDArray[np.float_],
    k: int,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
    """Retrieve the top-k items and scores from a score matrix

    Args:
        scores (npt.NDArray[np.float_]): user-item scores matix
        k (int): number of top items and scores to retrieve

    Returns:
        tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]: top_items and top_scores
    """
    # check scores is a two dimensional array
    if score_dimensions := scores.ndim != 2:
        raise ValueError(
            f"scores has to be a 2-dimensional array, passed is {score_dimensions}-dimensional"
        )

    logging.debug("Retrieving Top-K items")
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_items, axis=1)

    # sort top k items
    sort_ind = np.argsort(-top_scores)

    top_items = np.take_along_axis(top_items, sort_ind, axis=1)
    top_scores = np.take_along_axis(top_scores, sort_ind, axis=1)

    return top_items, top_scores


@njit(cache=True, parallel=True)  # type: ignore # pragma: no cover
def numba_get_topk(
    scores: npt.NDArray[np.float_], k: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Retrieve the top-k items and scores from a score matrix

    Args:
        scores (npt.NDArray[np.float_]): user-item scores matix
        k (int): number of top items and scores to retrieve

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]: top_items and top_scores
    """
    n_users, n_items = scores.shape
    # convert scores to float32
    scores = scores.astype(np.float32)

    top_scores, top_items = np.zeros((n_users, k), dtype=np.float32), np.zeros(
        (n_users, k), dtype=np.int64
    )
    # user cycle
    for i in prange(n_users):  # pylint: disable=not-an-iterable
        # item cycle
        # declare type of the lists
        user_scores = [np.float32(x) for x in range(0)]
        item_idx = [np.int64(x) for x in range(0)]
        score_idx_list = list(zip(user_scores, item_idx))
        heapq.heapify(score_idx_list)
        for j in range(n_items):
            candidate_score = scores[i][j]
            if j < k:
                heapq.heappush(score_idx_list, (candidate_score, j))  # type: ignore
            elif candidate_score > score_idx_list[0][0]:
                # pop smallest value and push new tup
                heapq.heapreplace(score_idx_list, (candidate_score, j))  # type: ignore
        topk = heapq.nlargest(k, score_idx_list)

        for idx in range(k):
            top_scores[i][idx] = topk[idx][0]
            top_items[i][idx] = topk[idx][1]

    return top_items, top_scores
