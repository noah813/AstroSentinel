# feature_engineering.py — 特徵提取

## 前置條件

- `config.py` 完成（C4）
- `data/collected/` 下至少有一個 `youtube_raw_*.csv`（由 `collector.py` 產生）

## 子任務

- [ ] **FE1** 建立測試夾具 `tests/fixtures/sample_raw.csv`：
  - 30 筆假留言，涵蓋邊界值：
    - 單一留言用戶（自身相似度 = 0.0）
    - 凌晨 0-5 點發文的用戶
    - 含廣告關鍵字的留言（例如「訂閱」、「加LINE」）
    - 含 `http://` 連結的留言
    - 同一用戶在同一小時內留多則言（測試 max_burst）
  - 包含欄位：`record_type`、`comment_id`、`video_id`、`author_id`、`content`、`like_count`、`reply_count`、`published_at`、`parent_id`

- [ ] **FE2** 實作 `load_raw_comments(data_dir, lookback_days)`：
  - 讀取 `data_dir/collected/youtube_raw_*.csv` 所有檔案
  - 篩選 `record_type in ("comment", "reply")`
  - 將 `published_at` 解析為 datetime（UTC aware）
  - 過濾至今日起算 `lookback_days` 天內的留言
  - 依 `comment_id` 去重（保留第一筆）後回傳 DataFrame

- [ ] **FE3** 實作 `compute_behavioral_features(df)`：
  - 以 `author_id` 為 groupby 鍵
  - 計算 `total_comments`（count）
  - 計算 `unique_videos`（nunique video_id）
  - 計算 `concentration`（`total_comments / unique_videos`，`unique_videos=0` 時填 0）
  - 計算 `avg_likes`（mean like_count）
  - 計算 `zero_like_ratio`（like_count==0 的比例）
  - 計算 `max_burst`（同作者同小時最大留言數，小時以 `published_at` 的 floor hour）
  - 回傳以 `author_id` 為 index 的 DataFrame

- [ ] **FE4** 實作 `compute_content_features(df)`：
  - 計算 `avg_length`（content 字數平均，`len()` 計）
  - 計算 `unique_content_ratio`（不重複 content 數 / total_comments）
  - 計算 `self_similarity`：
    - 留言數 < 2 時回傳 0.0
    - 用 `TfidfVectorizer(analyzer='char', ngram_range=(1,2))` 向量化
    - 取 `cosine_similarity` 上三角（排除對角線）的均值
  - 計算 `has_ad_keywords`（任一留言含 `config.AD_KEYWORDS` 中的關鍵字則為 1）
  - 計算 `url_ratio`（含 `http://`、`https://`、`www.` 的留言比例）
  - 回傳以 `author_id` 為 index 的 DataFrame

- [ ] **FE5** 實作 `compute_temporal_features(df)`：
  - 計算 `night_ratio`（`published_at` 小時在 `[0, 6)` 的比例，以 UTC 計算）
  - 計算 `interval_std`（同作者留言時間間隔秒數的標準差，留言數 < 2 時填 0.0）
  - 回傳以 `author_id` 為 index 的 DataFrame

- [ ] **FE6** 實作 `run_feature_engineering(data_dir, output_path, lookback_days)`：
  - 呼叫 FE2~FE5，以 `author_id` 為鍵合併三組特徵
  - 最終欄位順序：`author_id` + 13 個特徵（依 `config.FEATURE_COLS` 排列）
  - 寫入 `output_path`（預設 `data/user_features.csv`）

- [ ] **FE7** 寫測試 `tests/test_feature_engineering.py`：
  - `load_raw_comments` 相同 `comment_id` 只保留一筆
  - 單一留言用戶的 `self_similarity` 輸出 0.0
  - 單一留言用戶的 `interval_std` 輸出 0.0
  - `concentration` 在 `unique_videos=0`（防禦案例）填 0
  - `night_ratio` 以 UTC 時區計算正確
  - 使用 `tests/fixtures/sample_raw.csv` 做整合驗證

## 驗收條件

- `python feature_engineering.py` 執行後產生 `data/user_features.csv`
- CSV 包含 `author_id` + 恰好 13 個特徵欄位，無遺漏、無重複
- `tests/test_feature_engineering.py` 全部通過
