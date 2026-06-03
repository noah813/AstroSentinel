# fusion.py — 多模態融合（選配）

> **狀態：選配（Advanced）**  
> 需先完成 `train_stage1.py` 與 `train_stage2.py` 全部任務後方可執行。

## 前置條件

- `data/models/stage1/lr_model.pkl` 與 `xgb_model.pkl` 已存在
- `data/models/stage2/bert/` 已存在
- `data/bert_comment_scores.csv` 已存在
- `data/labeled_data.csv` 已存在
- `results/stage1_report.json` 已存在（供重現相同 test split）

## 子任務

- [ ] **FU1** 實作融合特徵構建函式 `build_fusion_features(labeled_path, model_dir, bert_scores_path, feature_cols)`：
  - 讀取 `labeled_data.csv` 的 `FEATURE_COLS` 特徵
  - 以 LR `predict_proba` 計算每位用戶的 `lr_prob`
  - 以 XGBoost `predict_proba` 計算每位用戶的 `xgb_prob`
  - 讀取 `bert_comment_scores.csv`，按 `author_id` 聚合：
    - `bert_user_prob`（spam_prob 的均值）
    - `bert_max_prob`（spam_prob 的最大值）
    - `bert_comment_count`（留言數）
  - 回傳含以上 5 維特徵的 DataFrame（index 為 `author_id`）

- [ ] **FU2** 實作融合模型訓練 `train_fusion(X_fusion, y, test_indices)`：
  - 以 `test_indices` 切出 hold-out test set（與 Stage1 相同）
  - 在剩餘資料上做 5-fold CV 訓練融合 LR：`LogisticRegression(class_weight="balanced", max_iter=500)`
  - 防止 data leakage：融合模型的訓練資料不含 test set
  - 回傳 `(best_model, cv_results_dict)`

- [ ] **FU3** 實作 `run_fusion(labeled_path, model_dir, bert_scores_path, results_dir)`（主執行函式）：
  - 呼叫 FU1、FU2
  - 儲存 `data/models/stage3/fusion_model.pkl`
  - 評估融合模型在 hold-out test set 的指標（同 evaluate.py 的 5 個指標）
  - 輸出 `results/stage3_report.json`

- [ ] **FU4** （可選）寫測試 `tests/test_fusion.py`：
  - `build_fusion_features` 輸出恰好 5 個特徵欄位
  - `train_fusion` 不使用 test_indices 中的樣本訓練（防 data leakage）

## 驗收條件

- `python fusion.py` 執行後產生 `data/models/stage3/fusion_model.pkl` 與 `results/stage3_report.json`
- `stage3_report.json` 包含 `precision`、`recall`、`f1`、`auc_roc` 四個指標
- 融合模型的評估 test set 與 Stage1/Stage2 完全相同
