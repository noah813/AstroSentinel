import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import config

FUSION_FEATURE_COLS = ["lr_prob", "xgb_prob", "bert_user_prob", "bert_max_prob", "bert_comment_count"]


def build_fusion_features(
    labeled_path: str,
    model_dir: str,
    bert_scores_path: str,
    feature_cols: list,
) -> pd.DataFrame:
    """構建 5 維融合特徵 DataFrame，index 為 author_id。"""
    labeled_df = pd.read_csv(labeled_path)
    X = labeled_df[feature_cols]
    author_ids = labeled_df["author_id"].values

    lr_model = joblib.load(os.path.join(model_dir, "stage1", "lr_model.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "stage1", "xgb_model.pkl"))

    lr_probs = lr_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    scores_df = pd.read_csv(bert_scores_path)
    bert_agg = scores_df.groupby("author_id").agg(
        bert_user_prob=("spam_prob", "mean"),
        bert_max_prob=("spam_prob", "max"),
        bert_comment_count=("spam_prob", "count"),
    )

    bert_user_probs = [
        bert_agg.at[aid, "bert_user_prob"] if aid in bert_agg.index else 0.5
        for aid in author_ids
    ]
    bert_max_probs = [
        bert_agg.at[aid, "bert_max_prob"] if aid in bert_agg.index else 0.5
        for aid in author_ids
    ]
    bert_comment_counts = [
        int(bert_agg.at[aid, "bert_comment_count"]) if aid in bert_agg.index else 0
        for aid in author_ids
    ]

    fusion_df = pd.DataFrame(
        {
            "lr_prob": lr_probs,
            "xgb_prob": xgb_probs,
            "bert_user_prob": bert_user_probs,
            "bert_max_prob": bert_max_probs,
            "bert_comment_count": bert_comment_counts,
        },
        index=author_ids,
    )
    return fusion_df


def train_fusion(
    X_fusion: pd.DataFrame,
    y: pd.Series,
    test_indices: list,
) -> tuple:
    """在非 test_indices 的樣本上 5-fold CV 訓練融合 LR，回傳 (model, cv_results)。"""
    test_idx_set = set(test_indices)
    train_idx = [i for i in range(len(X_fusion)) if i not in test_idx_set]

    X_train = X_fusion.iloc[train_idx].values
    y_train = y.iloc[train_idx].values

    clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")

    clf.fit(X_train, y_train)

    cv_results = {
        "train_size": len(train_idx),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
    }
    return clf, cv_results


def run_fusion(
    labeled_path: str = "data/labeled_data.csv",
    model_dir: str = "data/models",
    bert_scores_path: str = "data/bert_comment_scores.csv",
    results_dir: str = "results",
) -> None:
    stage1_report_path = os.path.join(results_dir, "stage1_report.json")
    with open(stage1_report_path, "r", encoding="utf-8") as f:
        stage1_report = json.load(f)
    test_indices = stage1_report["data_split_indices"]["test_indices"]

    labeled_df = pd.read_csv(labeled_path)
    y = labeled_df["is_bot"].reset_index(drop=True)

    print("[fusion] Building fusion features...")
    X_fusion = build_fusion_features(labeled_path, model_dir, bert_scores_path, config.FEATURE_COLS)

    print(f"[fusion] Training fusion model (train size = {len(y) - len(test_indices)})...")
    fusion_model, cv_results = train_fusion(X_fusion, y, test_indices)

    # Evaluate on hold-out test set (same as stage1 test split)
    X_test = X_fusion.iloc[test_indices].values
    y_test = y.iloc[test_indices].values
    y_pred = fusion_model.predict(X_test)
    y_proba = fusion_model.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(
        f"[fusion] Test results: F1={f1:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"AUC-ROC={auc:.4f}"
    )

    stage3_dir = os.path.join(model_dir, "stage3")
    os.makedirs(stage3_dir, exist_ok=True)
    model_path = os.path.join(stage3_dir, "fusion_model.pkl")
    joblib.dump(fusion_model, model_path)
    print(f"[fusion] Saved fusion model -> {model_path}")

    report = {
        "fusion_features": FUSION_FEATURE_COLS,
        "test_size": int(len(test_indices)),
        **cv_results,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
    }
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "stage3_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[fusion] Saved stage3 report -> {report_path}")


if __name__ == "__main__":
    run_fusion()
    print("Fusion complete.")
