import pytest

from recval.evaluator import RecEvaluator


def test_receval_check_metrics():
    cutoffs = [5, 10]
    metrics = ["Recall", "None"]

    with pytest.raises(ValueError):
        _ = RecEvaluator(cutoffs_list=cutoffs, metrics_list=metrics)


def test_receval_invalid_users(dummy_userids_scores_holdout):
    cutoffs = [5, 10]
    metrics = ["Recall", "MAP"]
    # create a non-matching number of user ids with scores
    invalid_user_ids = [1, 2]
    _, scores, holdout_df = dummy_userids_scores_holdout
    recval = RecEvaluator(cutoffs_list=cutoffs, metrics_list=metrics)
    with pytest.raises(ValueError):
        recval.eval_from_scores(
            scores=scores, holdout_data=holdout_df, user_ids=invalid_user_ids
        )
