import pytest

from recval.evaluator import RecEvaluator


def test_receval_invalid_metrics():
    cutoffs = [5, 10]
    metrics = ["Recall", "None"]

    with pytest.raises(ValueError):
        _ = RecEvaluator(cutoffs=cutoffs, metrics=metrics)


def test_receval_invalid_users(dummy_userids_scores_holdout):
    cutoffs = [5, 10]
    metrics = ["recall", "map"]
    # create a non-matching number of user ids with scores
    invalid_user_ids = [1, 2]
    _, scores, holdout_df = dummy_userids_scores_holdout
    recval = RecEvaluator(cutoffs=cutoffs, metrics=metrics)
    with pytest.raises(ValueError):
        recval.eval_from_scores(
            scores=scores, holdout_data=holdout_df, user_ids=invalid_user_ids
        )


def test_receval_eval_from_recs(dummy_recs_gt_cutoff):
    metric_list = ["recall", "precision", "f1_score", "ndcg", "map"]
    cutoff_list = [1, 2, 3]
    evaluator = RecEvaluator(metrics=metric_list, cutoffs=cutoff_list)
    recs_df, gt_df, _ = dummy_recs_gt_cutoff
    result_df = evaluator.eval_from_recs(recs_df=recs_df, holdout_data=gt_df)
    assert len(result_df) == 15


# TODO test_eval_from_scores
