import os
import json
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_mini_env(tmp_path, n=30):
    import config
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for i in range(n):
        row = {"author_id": f"user_{i:03d}", "is_bot": i % 2}
        for col in config.FEATURE_COLS:
            row[col] = float(i % 10) / 10.0
        rows.append(row)
    labeled_df = pd.DataFrame(rows)
    labeled_path = str(tmp_path / "labeled_data.csv")
    labeled_df.to_csv(labeled_path, index=False)

    X = labeled_df[config.FEATURE_COLS]
    y = labeled_df["is_bot"]
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    lr.fit(X, y)

    model_dir = str(tmp_path / "models")
    os.makedirs(os.path.join(model_dir, "stage1"), exist_ok=True)
    joblib.dump(lr, os.path.join(model_dir, "stage1", "lr_model.pkl"))
    joblib.dump(lr, os.path.join(model_dir, "stage1", "xgb_model.pkl"))

    scores_data = [
        {"comment_id": f"c{i}", "author_id": f"user_{i:03d}", "spam_prob": float(i % 2)}
        for i in range(n)
    ]
    scores_df = pd.DataFrame(scores_data)
    scores_path = str(tmp_path / "bert_comment_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    return labeled_path, model_dir, scores_path, labeled_df


def test_build_fusion_features_returns_five_columns(tmp_path):
    """build_fusion_features 輸出恰好 5 個特徵欄位"""
    import config
    from fusion import build_fusion_features, FUSION_FEATURE_COLS

    labeled_path, model_dir, scores_path, _ = _make_mini_env(tmp_path)

    result = build_fusion_features(labeled_path, model_dir, scores_path, config.FEATURE_COLS)

    assert len(result.columns) == 5, f"期望 5 個特徵欄位，實際 {len(result.columns)}"
    for col in FUSION_FEATURE_COLS:
        assert col in result.columns, f"缺少特徵欄位：{col}"


def test_build_fusion_features_handles_missing_bert_scores(tmp_path):
    """build_fusion_features 對沒有 BERT 分數的用戶應填入預設值"""
    import config
    from fusion import build_fusion_features

    labeled_path, model_dir, _, labeled_df = _make_mini_env(tmp_path)

    # Create empty bert scores (no users)
    empty_scores = pd.DataFrame(columns=["comment_id", "author_id", "spam_prob"])
    scores_path = str(tmp_path / "empty_scores.csv")
    empty_scores.to_csv(scores_path, index=False)

    result = build_fusion_features(labeled_path, model_dir, scores_path, config.FEATURE_COLS)

    # bert_user_prob and bert_max_prob should be 0.5 (default)
    assert (result["bert_user_prob"] == 0.5).all(), "缺少分數的用戶 bert_user_prob 應為 0.5"
    assert (result["bert_comment_count"] == 0).all(), "缺少分數的用戶 bert_comment_count 應為 0"


def test_train_fusion_excludes_test_indices(tmp_path):
    """train_fusion 的訓練資料不應包含 test_indices 中的樣本"""
    from fusion import build_fusion_features, train_fusion
    import config

    labeled_path, model_dir, scores_path, labeled_df = _make_mini_env(tmp_path, n=30)
    y = labeled_df["is_bot"].reset_index(drop=True)
    X_fusion = build_fusion_features(labeled_path, model_dir, scores_path, config.FEATURE_COLS)

    test_indices = list(range(0, 6))  # 前 6 個為 test
    model, cv_results = train_fusion(X_fusion, y, test_indices)

    expected_train_size = len(labeled_df) - len(test_indices)
    assert cv_results["train_size"] == expected_train_size, (
        f"訓練集大小應為 {expected_train_size}，實際 {cv_results['train_size']}"
    )


def test_train_fusion_returns_fitted_model(tmp_path):
    """train_fusion 應回傳可預測的模型"""
    from fusion import build_fusion_features, train_fusion
    import config

    labeled_path, model_dir, scores_path, labeled_df = _make_mini_env(tmp_path)
    y = labeled_df["is_bot"].reset_index(drop=True)
    X_fusion = build_fusion_features(labeled_path, model_dir, scores_path, config.FEATURE_COLS)

    test_indices = list(range(0, 6))
    model, cv_results = train_fusion(X_fusion, y, test_indices)

    assert model is not None
    X_test = X_fusion.iloc[test_indices].values
    preds = model.predict(X_test)
    assert len(preds) == len(test_indices)
    assert set(preds).issubset({0, 1}), "預測值應為 0 或 1"
    assert "cv_f1_mean" in cv_results
    assert "cv_f1_std" in cv_results


def test_run_fusion_creates_output_files(tmp_path):
    """run_fusion 應產生 fusion_model.pkl 與 stage3_report.json"""
    from fusion import run_fusion
    import config

    labeled_path, model_dir, scores_path, labeled_df = _make_mini_env(tmp_path, n=30)

    results_dir = str(tmp_path / "results")
    os.makedirs(results_dir, exist_ok=True)
    test_indices = list(range(24, 30))
    stage1_report = {"data_split_indices": {"test_indices": test_indices}}
    with open(os.path.join(results_dir, "stage1_report.json"), "w") as f:
        json.dump(stage1_report, f)

    run_fusion(
        labeled_path=labeled_path,
        model_dir=model_dir,
        bert_scores_path=scores_path,
        results_dir=results_dir,
    )

    model_path = os.path.join(model_dir, "stage3", "fusion_model.pkl")
    report_path = os.path.join(results_dir, "stage3_report.json")

    assert os.path.exists(model_path), "fusion_model.pkl 應存在"
    assert os.path.exists(report_path), "stage3_report.json 應存在"

    with open(report_path) as f:
        report = json.load(f)
    for key in ["precision", "recall", "f1", "auc_roc"]:
        assert key in report, f"stage3_report.json 缺少 '{key}'"
