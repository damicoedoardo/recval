import numpy as np

from recval.utils import get_topk, numba_get_topk


def test_get_topk_arange():
    arange = np.arange(90, 100, dtype=np.float32)
    dummy_scores = np.tile(arange, 10).reshape(-1, 10)
    top_items, top_scores = get_topk(scores=dummy_scores, k=1)
    assert all(i == 9 for i in top_items)
    assert all(s == 99 for s in top_scores)


def test_nb_get_topk_arange():
    arange = np.arange(90, 100, dtype=np.float32)
    dummy_scores = np.tile(arange, 10).reshape(-1, 10)
    top_items, top_scores = numba_get_topk(scores=dummy_scores, k=1)
    assert all(i == 9 for i in top_items)
    assert all(s == 99 for s in top_scores)
