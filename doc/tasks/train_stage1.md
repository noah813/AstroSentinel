# train_stage1.py — 傳統 ML 訓練

## 前置條件

- `config.py` 完成（C4）
- `data/labeled_data.csv` 已存在且 ≥ 50 筆
- `data/models/stage1/` 目錄已存在

## 子任務

- [ ] **TS1-1** 實作 `load_labeled_data(labeled_path, feature_cols)`：
  - 讀取 CSV，驗證 `feature_cols` 與 `is_bot` 欄位皆存在
  - 樣本數 < 50 時以清楚訊息拋 `ValueError`（不輸出模型）
  - 回傳 `(X: pd.DataFrame, y: pd.Series)`，`y` 為 `is_bot`

- [ ] **TS1-2** 實作 `train_logistic_regression(X_train, y_train)`：
  - `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"))])`
  - `GridSearchCV`（`StratifiedKFold(n_splits=5)`）搜尋 `clf__C in [0.01, 0.1, 1.0, 10.0]`，scoring=`"f1"`
  - 回傳 `(best_pipeline, best_params_dict)`

- [ ] **TS1-3** 實作 `train_xgboost(X_train, y_train)`：
  - 自動計算 `scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()`
  - `GridSearchCV` 搜尋：`n_estimators in [100, 200, 300]`、`max_depth in [3, 4, 6]`、`learning_rate in [0.05, 0.1, 0.2]`，scoring=`"f1"`
  - 回傳 `(best_xgb, best_params_dict)`

- [ ] **TS1-4** 實作 `run_training(labeled_path, model_dir, results_dir)`：
  - 以 `StratifiedShuffleSplit(test_size=0.2, random_state=42)` 切分 train/test，記錄 test 索引
  - 在 train 集上訓練 LR 與 XGBoost
  - 用 `joblib.dump` 儲存：
    - `{model_dir}/lr_model.pkl`（整個 Pipeline）
    - `{model_dir}/xgb_model.pkl`
    - `{model_dir}/scaler.pkl`（Pipeline 中的 scaler 獨立存放）
  - 輸出 `{results_dir}/stage1_report.json`（含 `train_size`、`test_size`、`lr`、`xgb`、`data_split_indices`）

- [ ] **TS1-5** 寫測試 `tests/test_train_stage1.py`：
  - 樣本數 < 50 時 `load_labeled_data` 拋 `ValueError`
  - `scale_pos_weight` 在 bot:normal = 1:9 時正確計算為 9.0
  - `joblib.load(lr_model.pkl)` 後模型可正常呼叫 `predict`
  - `stage1_report.json` 含必要欄位且格式正確
  - 使用 `tests/fixtures/sample_labeled.csv`（≥ 50 筆，或在測試中生成假資料）

## 驗收條件

- `python train_stage1.py` 執行後產生 `lr_model.pkl`、`xgb_model.pkl`、`scaler.pkl`、`stage1_report.json`
- `stage1_report.json` 的 `data_split_indices.test_indices` 非空
- `tests/test_train_stage1.py` 全部通過
