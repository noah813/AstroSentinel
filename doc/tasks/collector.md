# collector.py — 資料收集

## 前置條件

- `config.py` 完成（C4）
- `YOUTUBE_API_KEY` 已填入 `.env`
- `data/collected/` 目錄已存在

## 子任務

- [ ] **COL1** 實作 `load_collected_ids(path)`：
  - 檔案不存在時回傳 `{"video_ids": set(), "comment_ids": set()}`（不拋例外）
  - 正常讀取時將 JSON list 轉為 set
- [ ] **COL2** 實作 `save_collected_ids(ids, path)`：
  - 將 `set` 轉為 `list` 後寫入 JSON（確保可序列化）
- [ ] **COL3** 定義 `QuotaExceededError(Exception)`：包含 `used` 與 `limit` 屬性
- [ ] **COL4** 實作 `check_quota(used, limit)`：`used >= limit` 時拋 `QuotaExceededError`
- [ ] **COL5** 實作 `fetch_channel_uploads_playlist(service, channel_id)`：
  - 呼叫 `channels.list`（`part="contentDetails"`），回傳 uploads playlist ID
- [ ] **COL6** 實作 `fetch_recent_videos(service, playlist_id, since, collected_video_ids)`：
  - 逐頁呼叫 `playlistItems.list`（`maxResults=50`）
  - 跳過 `published_at < since` 的影片
  - 跳過已在 `collected_video_ids` 的影片
  - 回傳含 `video_id`、`video_title`、`channel_id`、`published_at` 的 list[dict]
- [ ] **COL7** 實作 `fetch_video_stats(service, video_ids)`：
  - 每批最多 50 個呼叫 `videos.list`（`part="statistics"`）
  - 回傳 `{video_id: {"view_count": int, "comment_count": int}}`
- [ ] **COL8** 實作 `fetch_comments(service, video_id, collected_comment_ids)`：
  - 逐頁呼叫 `commentThreads.list`（`maxResults=100`）
  - 對每個頂層留言若 `replyCount > 0` 呼叫 `comments.list` 取回覆
  - 跳過已在 `collected_comment_ids` 的留言
  - 回覆填入 `parent_id`，頂層留言 `parent_id` 為空字串
  - `record_type`：頂層為 `"comment"`，回覆為 `"reply"`
- [ ] **COL9** 實作 `run_collection(channel_ids, output_dir, collected_ids_path, quota_limit)`：
  - 主流程：遍歷頻道 → 取影片 → 取統計 → 取留言
  - 影片列（`record_type="video"`）與留言列合存於同一 CSV
  - 輸出命名：`youtube_raw_YYYYMMDD.csv`；當日已有同名檔案則加時間戳後綴
  - 捕捉 `QuotaExceededError`：寫入已收集資料後結束，stderr 輸出未完成頻道
  - 單一影片留言失敗時記錄 warning 並繼續（不終止）
- [ ] **COL10** 寫測試 `tests/test_collector.py`：
  - `load_collected_ids` 在檔案不存在時回傳空集合
  - `fetch_recent_videos` 正確過濾 `since` 時間
  - `fetch_comments` 展開回覆且 `parent_id` 正確填入
  - `check_quota` 在達限前拋例外
  - 所有 API 呼叫以 `unittest.mock.patch` mock（不需真實金鑰）

## 驗收條件

- `python collector.py` 執行後於 `data/collected/` 產生 `youtube_raw_YYYYMMDD.csv`
- CSV 同時包含 `record_type="video"` 與 `record_type="comment"` 的列
- `tests/test_collector.py` 全部通過
