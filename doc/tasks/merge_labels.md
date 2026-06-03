# merge_labels.py — 人工複審合併

## 前置條件

- `data/labeled_data.csv` 已存在（由 `auto_labeler.py` 產生）
- 研究者已在 `data/pending_review.csv` 的 `is_bot` 欄位填入 0 或 1（可部分填寫）

## 子任務

- [ ] **ML1** 實作 `merge_reviewed_labels(labeled_path, pending_path, output_path)`：
  - 讀取 `pending_review.csv`，只保留 `is_bot` 為 `"0"` 或 `"1"` 的列（轉為 int）
  - 跳過 `is_bot` 為空字串或 NaN 的列（計入 `skipped_empty`）
  - `is_bot` 為其他值（`2`、`"yes"` 等）時拋 `ValueError` 並終止（附清楚訊息說明哪一列）
  - 以 `author_id` 檢查重複：已存在於 `labeled_data.csv` 的 `author_id` 跳過並輸出 stderr 警告（計入 `skipped_duplicate`）
  - 合併後以 `author_id` 去重，覆寫 `output_path`
  - 回傳 `{"merged": N, "skipped_empty": N, "skipped_duplicate": N}`

- [ ] **ML2** 寫測試 `tests/test_merge_labels.py`：
  - `is_bot` 為空字串的列不納入合併
  - `is_bot` 為 `"yes"` 時拋 `ValueError`
  - 重複 `author_id` 保留 `labeled_data.csv` 的既有版本，stderr 有警告輸出
  - 部分填寫的 `pending_review.csv`（只有部分列有值）正確只合併有值的列

## 驗收條件

- `python merge_labels.py` 執行後 `data/labeled_data.csv` 筆數增加（等於 `merged` 數量）
- 非法 `is_bot` 值終止執行並輸出清楚錯誤訊息
- `tests/test_merge_labels.py` 全部通過
