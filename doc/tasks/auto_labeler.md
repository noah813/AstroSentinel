# auto_labeler.py — 自動標注

## 前置條件

- `config.py` 完成（C4）
- `data/user_features.csv` 已存在（由 `feature_engineering.py` 產生）

## 子任務

- [ ] **AL1** 建立測試夾具 `tests/fixtures/sample_labeled.csv`：
  - 20 位用戶，包含：
    - 明確水軍用戶（同時滿足 4 個 bot 條件，含邊界值如 `concentration=5.01`）
    - 邊界外用戶（`concentration=5.0`，剛好不算水軍）
    - 明確正常用戶（同時滿足 3 個 normal 條件）
    - 不滿足任一組條件的 pending 用戶
  - 欄位：`author_id` + 13 個特徵 + `is_bot`（0 或 1）

- [ ] **AL2** 實作 `classify_user(row)`（純函式）：
  - 輸入：一列 `pd.Series`（含所有特徵欄位）
  - 水軍條件：`concentration > 5.0` **且** `self_similarity > 0.7` **且** `zero_like_ratio > 0.8` **且** `interval_std < 60.0`
  - 正常條件：`unique_videos >= 3` **且** `self_similarity < 0.3` **且** `zero_like_ratio < 0.5`
  - 同時滿足兩組條件時（理論不可能，仍需處理）優先回傳 `'bot'`
  - 兩組條件均不滿足時回傳 `'pending'`

- [ ] **AL3** 實作 `run_labeling(features_path, labeled_output, pending_output)`：
  - 讀取 `user_features.csv`，驗證必要欄位存在（不存在時拋清楚錯誤）
  - 對每列呼叫 `classify_user()`
  - `is_bot=1`（bot）與 `is_bot=0`（normal）的列寫入 `labeled_output`
  - pending 列的 `is_bot` 填空字串（非 NaN）寫入 `pending_output`
  - 終端輸出統計：`[auto_labeler] bot=N, normal=N, pending=N`
  - 回傳統計 dict `{"bot": N, "normal": N, "pending": N}`

- [ ] **AL4** 寫測試 `tests/test_auto_labeler.py`：
  - `classify_user` 對 `concentration=5.0` 回傳非 `'bot'`（邊界值，不含等號）
  - `classify_user` 對 `concentration=5.01` 且其他 bot 條件滿足時回傳 `'bot'`
  - `pending_review.csv` 的 `is_bot` 欄位為空字串（非 NaN）
  - `labeled_data.csv` 無任何空 `is_bot` 值

## 驗收條件

- `python auto_labeler.py` 執行後同時產生 `data/labeled_data.csv` 與 `data/pending_review.csv`
- `labeled_data.csv` 的 `is_bot` 欄位只含 0 或 1
- `pending_review.csv` 的 `is_bot` 欄位全為空字串
- `tests/test_auto_labeler.py` 全部通過
