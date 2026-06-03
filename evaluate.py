import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import config


def load_test_split(
    labeled_path: str,
    stage1_report_path: str,
    feature_cols: list,
) -> tuple:
    """以 stage1_report.json 記錄的索引還原 test split，回傳 (X_test, y_test)。"""
    labeled_df = pd.read_csv(labeled_path)
    with open(stage1_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    test_indices = report["data_split_indices"]["test_indices"]
    test_df = labeled_df.iloc[test_indices]
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df["is_bot"].reset_index(drop=True)
    return X_test, y_test


def evaluate_sklearn_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """評估 sklearn/XGBoost 模型，回傳指標 dict。"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
    }


def evaluate_bert_model(
    bert_model_dir: str,
    test_author_ids: list,
    raw_dir: str,
    y_test: pd.Series,
) -> dict | None:
    """聚合 BERT 留言分數至用戶層級，回傳指標 dict。BERT 目錄不存在時回傳 None。"""
    if not os.path.isdir(bert_model_dir):
        return None

    scores_path = os.path.join(raw_dir, "bert_comment_scores.csv")
    if not os.path.exists(scores_path):
        return None

    scores_df = pd.read_csv(scores_path)
    user_scores = scores_df.groupby("author_id")["spam_prob"].mean().to_dict()

    # Default 0.5 for users without comment scores (neutral / uncertain)
    y_pred_proba = [user_scores.get(aid, 0.5) for aid in test_author_ids]
    y_pred = [int(p >= 0.5) for p in y_pred_proba]

    y_vals = y_test.values if hasattr(y_test, "values") else list(y_test)

    precision = float(precision_score(y_vals, y_pred, zero_division=0))
    recall = float(recall_score(y_vals, y_pred, zero_division=0))
    f1 = float(f1_score(y_vals, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_vals, y_pred_proba))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(y_vals, y_pred).tolist()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
    }


def run_evaluation(
    labeled_path: str = "data/labeled_data.csv",
    model_dir: str = "data/models",
    raw_dir: str = "data",
    results_dir: str = "results",
) -> None:
    stage1_report_path = os.path.join(results_dir, "stage1_report.json")

    labeled_df = pd.read_csv(labeled_path)
    with open(stage1_report_path, "r", encoding="utf-8") as f:
        stage1_report = json.load(f)
    test_indices = stage1_report["data_split_indices"]["test_indices"]
    test_df = labeled_df.iloc[test_indices]

    X_test = test_df[config.FEATURE_COLS].reset_index(drop=True)
    y_test = test_df["is_bot"].reset_index(drop=True)
    test_author_ids = test_df["author_id"].tolist()

    print(f"[evaluate] Test set size: {len(y_test)} "
          f"(bot={int(y_test.sum())}, normal={int((y_test == 0).sum())})")

    lr_model = joblib.load(os.path.join(model_dir, "stage1", "lr_model.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "stage1", "xgb_model.pkl"))

    lr_metrics = evaluate_sklearn_model(lr_model, X_test, y_test, "logistic_regression")
    print(f"[evaluate] LR  : F1={lr_metrics['f1']:.4f}, "
          f"Precision={lr_metrics['precision']:.4f}, "
          f"Recall={lr_metrics['recall']:.4f}, "
          f"AUC-ROC={lr_metrics['auc_roc']:.4f}")

    xgb_metrics = evaluate_sklearn_model(xgb_model, X_test, y_test, "xgboost")
    print(f"[evaluate] XGB : F1={xgb_metrics['f1']:.4f}, "
          f"Precision={xgb_metrics['precision']:.4f}, "
          f"Recall={xgb_metrics['recall']:.4f}, "
          f"AUC-ROC={xgb_metrics['auc_roc']:.4f}")

    bert_dir = os.path.join(model_dir, "stage2", "bert")
    bert_metrics = evaluate_bert_model(bert_dir, test_author_ids, raw_dir, y_test)
    if bert_metrics:
        print(f"[evaluate] BERT: F1={bert_metrics['f1']:.4f}, "
              f"Precision={bert_metrics['precision']:.4f}, "
              f"Recall={bert_metrics['recall']:.4f}, "
              f"AUC-ROC={bert_metrics['auc_roc']:.4f}")
    else:
        print("[evaluate] BERT: model not found or no scores, skipped")

    report_data = {
        "test_size": int(len(y_test)),
        "models": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics,
            "bert_aggregated": bert_metrics,
        },
    }

    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"[evaluate] Saved evaluation report -> {report_path}")

    metric_names = ["precision", "recall", "f1", "auc_roc"]
    rows = []
    for model_name, metrics in report_data["models"].items():
        if metrics is None:
            continue
        row = {"model": model_name}
        for m in metric_names:
            row[m] = round(metrics[m], 4)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(results_dir, "comparison_report.csv")
    comparison_df.to_csv(comparison_path, index=False, float_format="%.4f")
    print(f"[evaluate] Saved comparison report -> {comparison_path}")


if __name__ == "__main__":
    run_evaluation()
    print("Evaluation complete.")
