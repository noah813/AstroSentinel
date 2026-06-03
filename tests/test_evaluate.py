import os
import json
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_LABELED = os.path.join(FIXTURES_DIR, "sample_labeled.csv")


def _make_mini_labeled(tmp_path, n=20):
    import config
    rows = []
    for i in range(n):
        row = {"author_id": f"user_{i:03d}", "is_bot": i % 2}
        for col in config.FEATURE_COLS:
            row[col] = float(i % 10) / 10.0
        rows.append(row)
    df = pd.DataFrame(rows)
    p = str(tmp_path / "labeled_data.csv")
    df.to_csv(p, index=False)
    return df, p


def test_load_test_split_matches_stage1_indices(tmp_path):
    """load_test_split 以相同索引取出的測試集應與手動 iloc 結果一致"""
    import config
    from evaluate import load_test_split

    df, labeled_path = _make_mini_labeled(tmp_path)
    test_indices = [2, 5, 7, 11, 15]
    report = {"data_split_indices": {"test_indices": test_indices}}
    report_path = str(tmp_path / "stage1_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f)

    X_test, y_test = load_test_split(labeled_path, report_path, config.FEATURE_COLS)

    expected_y = df.iloc[test_indices]["is_bot"].values
    assert len(X_test) == len(test_indices)
    assert len(y_test) == len(test_indices)
    assert list(y_test.values) == list(expected_y), "y_test 應與 stage1 的 test_indices 一致"
    assert list(X_test.columns) == config.FEATURE_COLS


def test_evaluate_bert_model_returns_none_if_dir_missing(tmp_path):
    """BERT 目錄不存在時，evaluate_bert_model 應回傳 None，不拋例外"""
    from evaluate import evaluate_bert_model

    nonexistent = str(tmp_path / "no_such_bert_dir")
    y_test = pd.Series([0, 1, 0, 1])
    result = evaluate_bert_model(nonexistent, ["a", "b", "c", "d"], str(tmp_path), y_test)
    assert result is None, "目錄不存在時應回傳 None"


def test_evaluate_sklearn_model_returns_required_keys():
    """evaluate_sklearn_model 應回傳含所有必要指標的 dict"""
    from evaluate import evaluate_sklearn_model

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0, 1, 1])
    mock_model.predict_proba.return_value = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.3, 0.7], [0.1, 0.9]
    ])

    X_test = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_test = pd.Series([0, 1, 0, 1, 1])

    result = evaluate_sklearn_model(mock_model, X_test, y_test, "test_model")

    for key in ["precision", "recall", "f1", "auc_roc", "confusion_matrix"]:
        assert key in result, f"結果缺少 '{key}' 指標"
    assert isinstance(result["confusion_matrix"], list)
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["auc_roc"] <= 1.0


def test_evaluate_bert_model_aggregates_user_scores(tmp_path):
    """evaluate_bert_model 應正確聚合留言分數至用戶層級"""
    from evaluate import evaluate_bert_model

    # Create fake bert model dir
    bert_dir = str(tmp_path / "bert")
    os.makedirs(bert_dir)

    # Create bert_comment_scores.csv
    scores_data = [
        {"comment_id": "c1", "author_id": "user_000", "spam_prob": 0.8},
        {"comment_id": "c2", "author_id": "user_000", "spam_prob": 0.9},
        {"comment_id": "c3", "author_id": "user_001", "spam_prob": 0.1},
        {"comment_id": "c4", "author_id": "user_002", "spam_prob": 0.2},
    ]
    scores_df = pd.DataFrame(scores_data)
    scores_df.to_csv(str(tmp_path / "bert_comment_scores.csv"), index=False)

    test_author_ids = ["user_000", "user_001", "user_002"]
    y_test = pd.Series([1, 0, 0])

    result = evaluate_bert_model(bert_dir, test_author_ids, str(tmp_path), y_test)
    assert result is not None, "BERT 目錄存在時不應回傳 None"
    for key in ["precision", "recall", "f1", "auc_roc", "confusion_matrix"]:
        assert key in result


def test_comparison_report_four_decimal_places(tmp_path):
    """comparison_report.csv 數值應保留 4 位小數"""
    import config
    from evaluate import run_evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df, labeled_path = _make_mini_labeled(tmp_path, n=50)

    test_indices = list(range(40, 50))
    results_dir = str(tmp_path / "results")
    os.makedirs(results_dir, exist_ok=True)
    report = {"data_split_indices": {"test_indices": test_indices}}
    with open(os.path.join(results_dir, "stage1_report.json"), "w") as f:
        json.dump(report, f)

    X_train = df.iloc[:40][config.FEATURE_COLS]
    y_train = df.iloc[:40]["is_bot"]
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    lr.fit(X_train, y_train)

    model_dir = str(tmp_path / "models")
    os.makedirs(os.path.join(model_dir, "stage1"), exist_ok=True)
    joblib.dump(lr, os.path.join(model_dir, "stage1", "lr_model.pkl"))
    joblib.dump(lr, os.path.join(model_dir, "stage1", "xgb_model.pkl"))

    run_evaluation(
        labeled_path=labeled_path,
        model_dir=model_dir,
        raw_dir=str(tmp_path),
        results_dir=results_dir,
    )

    comp_path = os.path.join(results_dir, "comparison_report.csv")
    assert os.path.exists(comp_path), "comparison_report.csv 應存在"

    comp_df = pd.read_csv(comp_path)
    for col in ["precision", "recall", "f1", "auc_roc"]:
        assert col in comp_df.columns, f"缺少 '{col}' 欄位"
        for val in comp_df[col]:
            val_str = f"{val:.10f}"
            parts = val_str.rstrip("0").split(".")
            decimal_part = parts[1] if len(parts) == 2 else ""
            assert len(decimal_part) <= 4, f"值 {val} 在 '{col}' 欄超過 4 位小數"
