import numpy as np
import pytest

from recval.metrics.accuracy import f1_score, precision, recall
from recval.metrics.metrics_utils import get_hit_rank


def test_recall(dummy_recs_gt_cutoff):
    recs_df, gt_df, cutoff = dummy_recs_gt_cutoff
    _, df_hit_count = get_hit_rank(
        ground_truth_df=gt_df, pred_df=recs_df, cutoff=cutoff
    )
    rec_df = recall(df_hit_count=df_hit_count)
    assert (rec_df["recall"].values == np.array([1 / 2, 0.0, 1.0])).all()


def test_precision(dummy_recs_gt_cutoff):
    recs_df, gt_df, cutoff = dummy_recs_gt_cutoff
    _, df_hit_count = get_hit_rank(
        ground_truth_df=gt_df, pred_df=recs_df, cutoff=cutoff
    )
    rec_df = precision(df_hit_count=df_hit_count, cutoff=cutoff)
    assert (rec_df["precision"].values == np.array([1 / 3, 0.0, 1.0])).all()


def test_f1_score(dummy_recs_gt_cutoff):
    recs_df, gt_df, cutoff = dummy_recs_gt_cutoff
    _, df_hit_count = get_hit_rank(
        ground_truth_df=gt_df, pred_df=recs_df, cutoff=cutoff
    )
    rec_df = f1_score(df_hit_count=df_hit_count, cutoff=cutoff)
    assert rec_df["f1_score"].values == pytest.approx(
        np.array([2 * (1 / 3 * 1 / 2) / (1 / 3 + 1 / 2), 0.0, 1.0])
    )
