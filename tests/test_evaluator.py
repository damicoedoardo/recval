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

    recall_at3 = result_df[
        (result_df["cutoff"] == 3) & (result_df["metric"] == "recall")
    ].value.values[0]
    assert recall_at3 == 0.5

    precision_at3 = result_df[
        (result_df["cutoff"] == 3) & (result_df["metric"] == "precision")
    ].value.values[0]
    assert precision_at3 == pytest.approx((1 / 3 + 0.0 + 1.0) / 3)

    ndcg_at3 = result_df[
        (result_df["cutoff"] == 3) & (result_df["metric"] == "ndcg")
    ].value.values[0]
    assert ndcg_at3 == pytest.approx((0.6131 + 0.0 + 1.0) / 3, rel=1e-4)

    map_at3 = result_df[
        (result_df["cutoff"] == 3) & (result_df["metric"] == "map")
    ].value.values[0]
    assert map_at3 == pytest.approx((0.5 + 0.0 + 1.0) / 3)


def test_receval_eval_from_scores(dummy_userids_scores_holdout):
    metric_list = ["recall", "precision"]
    cutoff_list = [3]
    evaluator = RecEvaluator(metrics=metric_list, cutoffs=cutoff_list)
    users, scores, holdout_df = dummy_userids_scores_holdout
    res_df = evaluator.eval_from_scores(
        scores=scores, user_ids=users, holdout_data=holdout_df
    )
    assert res_df[res_df["metric"] == "recall"].value.values[0] == pytest.approx(0.5)
    assert res_df[res_df["metric"] == "precision"].value.values[0] == pytest.approx(
        0.4444, rel=1e-3
    )


def test_receval_eval_from_scores_missing_users_ids(dummy_userids_scores_holdout):
    metric_list = ["recall", "precision"]
    cutoff_list = [3]
    evaluator = RecEvaluator(metrics=metric_list, cutoffs=cutoff_list)
    _, scores, holdout_df = dummy_userids_scores_holdout
    res_df = evaluator.eval_from_scores(scores=scores, holdout_data=holdout_df)
    assert res_df[res_df["metric"] == "recall"].value.values[0] == pytest.approx(0.5)
    assert res_df[res_df["metric"] == "precision"].value.values[0] == pytest.approx(
        0.4444, rel=1e-3
    )
