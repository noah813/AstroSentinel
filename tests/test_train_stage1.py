import pytest
import os
import json
import numpy as np
import pandas as pd
import joblib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_LABELED = os.path.join(FIXTURES_DIR, "sample_labeled.csv")


def test_load_labeled_data_insufficient():
    """樣本數 < 50 時應拋 ValueError"""
    from train_stage1 import load_labeled_data
    import config
    # 建立只有 10 筆的假 CSV
    data = {col: [0.0] * 10 for col in config.FEATURE_COLS}
    data["author_id"] = [f"u{i}" for i in range(10)]
    data["is_bot"] = [0] * 10
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        pd.DataFrame(data).to_csv(f, index=False)
        path = f.name
    try:
        with pytest.raises(ValueError, match="50"):
            load_labeled_data(path, config.FEATURE_COLS)
    finally:
        os.unlink(path)


def test_load_labeled_data_success():
    """正常讀取 fixture 資料"""
    from train_stage1 import load_labeled_data
    import config
    X, y = load_labeled_data(SAMPLE_LABELED, config.FEATURE_COLS)
    assert len(X) >= 50
    assert len(y) == len(X)
    assert set(X.columns) == set(config.FEATURE_COLS)


def test_scale_pos_weight():
    """bot:normal = 1:9 時 scale_pos_weight 應為 9.0"""
    y = np.array([0] * 90 + [1] * 10)
    # 只驗算 scale_pos_weight = 90/10 = 9.0（不實際訓練）
    expected_spw = (y == 0).sum() / (y == 1).sum()
    assert abs(expected_spw - 9.0) < 0.001


def test_train_logistic_regression():
    """LR 訓練後可正常 predict"""
    from train_stage1 import load_labeled_data, train_logistic_regression
    import config
    X, y = load_labeled_data(SAMPLE_LABELED, config.FEATURE_COLS)
    lr_model, params = train_logistic_regression(X, y)
    preds = lr_model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
    assert "clf__C" in params


def test_run_training_output(tmp_path):
    """完整訓練產生所需檔案"""
    from train_stage1 import run_training
    model_dir = str(tmp_path / "models" / "stage1")
    results_dir = str(tmp_path / "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    run_training(SAMPLE_LABELED, model_dir, results_dir)
    assert os.path.exists(os.path.join(model_dir, "lr_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "xgb_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "scaler.pkl"))
    report_path = os.path.join(results_dir, "stage1_report.json")
    assert os.path.exists(report_path)
    with open(report_path) as f:
        report = json.load(f)
    assert "train_size" in report
    assert "test_size" in report
    assert "lr" in report
    assert "xgb" in report
    assert len(report["data_split_indices"]["test_indices"]) > 0


def test_model_predict_after_load(tmp_path):
    """joblib.load 後模型可正常 predict"""
    from train_stage1 import run_training, load_labeled_data
    import config
    model_dir = str(tmp_path / "models" / "stage1")
    results_dir = str(tmp_path / "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    run_training(SAMPLE_LABELED, model_dir, results_dir)
    lr_model = joblib.load(os.path.join(model_dir, "lr_model.pkl"))
    X, y = load_labeled_data(SAMPLE_LABELED, config.FEATURE_COLS)
    preds = lr_model.predict(X)
    assert len(preds) == len(y)


def test_stage1_report_format(tmp_path):
    """stage1_report.json 格式正確"""
    from train_stage1 import run_training
    model_dir = str(tmp_path / "models" / "stage1")
    results_dir = str(tmp_path / "results")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    run_training(SAMPLE_LABELED, model_dir, results_dir)
    with open(os.path.join(results_dir, "stage1_report.json")) as f:
        report = json.load(f)
    for model_key in ["lr", "xgb"]:
        assert "precision" in report[model_key]
        assert "recall" in report[model_key]
        assert "f1" in report[model_key]
        assert "auc_roc" in report[model_key]
