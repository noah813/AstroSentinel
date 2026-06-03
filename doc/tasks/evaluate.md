# evaluate.py — 模型評估

## 前置條件

- `data/labeled_data.csv` 已存在
- `results/stage1_report.json` 已存在（含 `data_split_indices.test_indices`）
- `data/models/stage1/lr_model.pkl` 與 `xgb_model.pkl` 已存在
- （可選）`data/models/stage2/bert/` 已存在；不存在時跳過 BERT 評估

## 子任務

- [ ] **EV1** 實作 `load_test_split(labeled_path, stage1_report_path, feature_cols)`：
  - 讀取 `stage1_report.json` 的 `data_split_indices.test_indices`
  - 以該索引從 `labeled_data.csv` 取出測試集
  - 回傳 `(X_test: pd.DataFrame, y_test: pd.Series)`

- [ ] **EV2** 實作 `evaluate_sklearn_model(model, X_test, y_test, model_name)`：
  - 計算以下指標（`sklearn`）：
    - `precision`（`precision_score(average='binary')`）
    - `recall`（`recall_score(average='binary')`）
    - `f1`（`f1_score(average='binary')`）
    - `auc_roc`（`roc_auc_score`，需 `predict_proba`）
    - `confusion_matrix`（`[[TN, FP], [FN, TP]]`）
  - 回傳指標 dict

- [ ] **EV3** 實作 `evaluate_bert_model(bert_model_dir, test_author_ids, raw_dir, y_test)`：
  - `bert_model_dir` 不存在時回傳 `None`（不拋例外）
  - 讀取 `data/bert_comment_scores.csv`，篩選屬於 `test_author_ids` 的留言
  - 聚合：每位用戶取 `spam_prob` 均值 → `user_bot_prob`
  - `threshold=0.5`：`is_bot_pred = int(user_bot_prob >= 0.5)`
  - 計算同 EV2 的 5 個指標並回傳

- [ ] **EV4** 實作 `run_evaluation(labeled_path, model_dir, raw_dir, results_dir)`：
  - 載入 LR 與 XGBoost 模型，呼叫 EV1~EV3 進行評估
  - 輸出 `{results_dir}/evaluation_report.json`（格式見詳設 §10）
  - 輸出 `{results_dir}/comparison_report.csv`：行為模型、列為指標，數值保留 4 位小數
  - BERT 不存在時在 report 標示 `"bert_aggregated": null`

- [ ] **EV5** 寫測試 `tests/test_evaluate.py`：
  - `load_test_split` 以相同索引取出的測試集與 Stage1 訓練時完全一致
  - BERT 模型目錄不存在時 `evaluate_bert_model` 回傳 `None`，不拋例外
  - `comparison_report.csv` 數值小數點位數為 4

## 驗收條件

- `python evaluate.py` 執行後產生 `results/evaluation_report.json` 與 `results/comparison_report.csv`
- 三個模型在 **完全相同的** test set 上被評估（test_indices 一致）
- `tests/test_evaluate.py` 全部通過
