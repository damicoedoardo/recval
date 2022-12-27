import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays

from recval.utils import get_topk, numba_get_topk


@given(
    scores=arrays(np.float_, elements=st.floats(-10.0, 10.0), shape=(2, 5)),
    k=st.integers(1, 3),
)
def test_get_topk(scores, k):
    top_items, top_scores = get_topk(scores, k)

    # Check that the top-k items and scores are returned
    assert top_items.shape == top_scores.shape == (scores.shape[0], k)

    # Check that the top-k items and scores are sorted
    for i in range(scores.shape[0]):
        assert np.all(np.diff(top_scores[i]) <= 0)

    # Check that the top-k scores are the highest scores in the scores matrix
    sorted_scores = np.sort(scores)[:, :k][:, ::-1]
    for i in range(scores.shape[0]):
        assert np.all(top_scores[i] >= sorted_scores[i])


def test_nb_get_topk_arange():
    arange = np.arange(90, 100, dtype=np.float32)
    dummy_scores = np.tile(arange, 10).reshape(-1, 10)
    top_items, top_scores = numba_get_topk(scores=dummy_scores, k=1)
    assert all(i == 9 for i in top_items)
    assert all(s == 99 for s in top_scores)


def test_get_topk_scores_wrong_shape():
    scores = np.random.rand(10, 10, 10)
    with pytest.raises(ValueError):
        get_topk(scores, k=1)
