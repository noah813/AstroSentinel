import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import config


def load_labeled_data(labeled_path: str, feature_cols: list) -> tuple:
    """
    讀取 CSV，驗證欄位，樣本數 < 50 時拋 ValueError。
    回傳 (X: pd.DataFrame, y: pd.Series)
    """
    df = pd.read_csv(labeled_path)
    missing = set(feature_cols + ["is_bot"]) - set(df.columns)
    if missing:
        raise ValueError(f"缺少欄位: {missing}")
    if len(df) < 50:
        raise ValueError(f"樣本數不足（{len(df)} < 50）")
    X = df[feature_cols]
    y = df["is_bot"].astype(int)
    return X, y


def train_logistic_regression(X_train, y_train) -> tuple:
    """
    Pipeline(StandardScaler + LogisticRegression)，GridSearchCV 搜尋 C。
    回傳 (best_pipeline, best_params_dict)
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ))
    ])
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=5)
    gs = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_xgboost(X_train, y_train) -> tuple:
    """
    自動計算 scale_pos_weight，GridSearchCV 搜尋超參數。
    回傳 (best_xgb, best_params_dict)
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=5)
    gs = GridSearchCV(xgb, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def run_training(
    labeled_path: str = "data/labeled_data.csv",
    model_dir: str = "data/models/stage1",
    results_dir: str = "results",
) -> None:
    X, y = load_labeled_data(labeled_path, config.FEATURE_COLS)

    # StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.STAGE1_TEST_SIZE,
        random_state=config.STAGE1_RANDOM_STATE,
    )
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 訓練
    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    xgb_model, xgb_params = train_xgboost(X_train, y_train)

    # 評估
    def eval_model(model, params):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "best_params": {
                k: float(v) if isinstance(v, (np.floating,)) else v
                for k, v in params.items()
            },
        }

    # 儲存模型
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(model_dir, "lr_model.pkl"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_model.pkl"))
    # 獨立儲存 scaler（取自 LR pipeline）
    joblib.dump(lr_model.named_steps["scaler"], os.path.join(model_dir, "scaler.pkl"))

    # 輸出報告
    report = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "lr": eval_model(lr_model, lr_params),
        "xgb": eval_model(xgb_model, xgb_params),
        "data_split_indices": {
            "test_indices": test_idx.tolist(),
            "train_indices": train_idx.tolist(),
        },
    }
    with open(os.path.join(results_dir, "stage1_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_training()
    print("Stage1 training complete.")
