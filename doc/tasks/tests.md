# tests/ — 測試基礎設施

## 前置條件

- `requirements.txt` 中已列入 `pytest`

## 子任務

- [ ] **T1** 建立 `tests/__init__.py`（空檔，讓 pytest 正確發現測試）
- [ ] **T2** 建立 `tests/fixtures/sample_raw.csv`（30 筆假留言）：
  - 欄位：`record_type`、`comment_id`、`video_id`、`author_id`、`author_name`、`content`、`like_count`、`reply_count`、`published_at`（ISO8601 UTC）、`parent_id`
  - 涵蓋以下邊界情境：
    - 單一留言用戶（author A，1 則留言）
    - 凌晨 0~5 點發文的用戶（author B，UTC 時間）
    - 含廣告關鍵字的留言（「訂閱」、「加LINE」）
    - 含 `https://` 連結的留言
    - 同一用戶同一小時內 3+ 則留言（測試 max_burst）
    - reply 型別的留言（parent_id 非空）
    - `like_count=0` 的留言（測試 zero_like_ratio）
- [ ] **T3** 建立 `tests/fixtures/sample_labeled.csv`（≥ 50 位用戶）：
  - 欄位：`author_id` + 13 個特徵 + `is_bot`
  - 包含確定水軍（is_bot=1）、確定正常（is_bot=0）的樣本（各 ≥ 5 筆）
- [ ] **T4** 建立 `tests/test_pipeline.py`（整合測試）：
  - 以 fixture 資料完整執行 `feature_engineering` → `auto_labeler` → `merge_labels` → `train_stage1` → `evaluate`（不包含 collector，不需 API）
  - 驗證每個模組的輸出檔案存在且欄位格式正確
  - 驗證 `evaluate.py` 能正確計算指標（不要求達到目標值，只驗格式）

## 驗收條件

- `pytest tests/` 全部通過（含所有模組的單元測試）
- `tests/test_pipeline.py` 在不需要網路或 GPU 的情況下通過
