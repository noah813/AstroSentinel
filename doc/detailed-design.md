# 網路水軍評論偵測系統 — 詳細設計文檔

**版本：** v1.0  
**日期：** 2026-06-03  
**對應提案：** proposal.md v1.0  

---

## 目錄

1. [總體架構](#1-總體架構)
2. [目錄結構](#2-目錄結構)
3. [全域設定 config.py](#3-全域設定-configpy)
4. [模組 1：collector.py — 資料收集](#4-模組-1collectpy--資料收集)
5. [模組 2：feature_engineering.py — 特徵提取](#5-模組-2feature_engineeringpy--特徵提取)
6. [模組 3：auto_labeler.py — 自動標注](#6-模組-3auto_labelerpy--自動標注)
7. [模組 4：merge_labels.py — 人工複審合併](#7-模組-4merge_labelspy--人工複審合併)
8. [模組 5：train_stage1.py — 傳統 ML 訓練](#8-模組-5train_stage1py--傳統-ml-訓練)
9. [模組 6：train_stage2.py — BERT 微調](#9-模組-6train_stage2py--bert-微調)
10. [模組 7：evaluate.py — 模型評估](#10-模組-7evaluatepy--模型評估)
11. [模組 8：fusion.py — 多模態融合（選配）](#11-模組-8fusionpy--多模態融合選配)
12. [模組間資料流](#12-模組間資料流)
13. [錯誤處理策略](#13-錯誤處理策略)
14. [測試策略](#14-測試策略)

---

## 1. 總體架構

本系統由 8 個獨立可執行的 Python 腳本組成，每個腳本負責單一職責，透過檔案系統（CSV / JSON / PKL）傳遞資料。模組之間無直接 import 相依，任何模組均可單獨執行與測試。

```
collector.py
    │  youtube_raw_YYYYMMDD.csv
    ▼
feature_engineering.py
    │  user_features.csv
    ▼
auto_labeler.py
    ├── labeled_data.csv
    └── pending_review.csv
           │ （人工填寫 is_bot 欄位後）
           ▼
    merge_labels.py
           │  labeled_data.csv（更新）
           ▼
    ┌──────────────────────┐
    │                      │
train_stage1.py      train_stage2.py
    │                      │
lr_model.pkl         bert_model/
xgb_model.pkl
    │                      │
    └──────────┬───────────┘
               ▼
          evaluate.py
               │  evaluation_report.json
               ▼
          fusion.py（選配）
```

---

## 2. 目錄結構

```
AstroSentinel/
├── .env                            # YOUTUBE_API_KEY=...（不上傳 Git）
├── .gitignore                      # 排除 .env, data/, results/
├── config.py                       # 全域設定
├── collector.py
├── feature_engineering.py
├── auto_labeler.py
├── merge_labels.py
├── train_stage1.py
├── train_stage2.py
├── evaluate.py
├── fusion.py                       # 選配
├── requirements.txt
├── data/
│   ├── collected/
│   │   ├── youtube_raw_20260603.csv
│   │   ├── youtube_raw_20260604.csv   # 增量追加，每次執行新增一個
│   │   └── collected_ids.json         # 已收集的 video_id / comment_id 集合
│   ├── user_features.csv
│   ├── labeled_data.csv
│   ├── pending_review.csv
│   └── models/
│       ├── stage1/
│       │   ├── lr_model.pkl
│       │   ├── xgb_model.pkl
│       │   └── scaler.pkl
│       ├── stage2/
│       │   └── bert/               # Hugging Face 格式
│       └── stage3/                 # 選配
│           └── fusion_model.pkl
├── results/
│   ├── stage1_report.json
│   ├── stage2_report.json
│   └── comparison_report.csv
└── doc/
    ├── proposal.md
    └── detailed-design.md
```

---

## 3. 全域設定 config.py

### 職責

集中管理所有可調整的超參數、路徑與常數，避免硬編碼散落各模組。

### 介面

```python
# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
YOUTUBE_API_KEY: str = os.environ["YOUTUBE_API_KEY"]  # 必填，否則啟動時 KeyError
CHANNEL_IDS: list[str] = [
    "UC5l1Yto5oOIgRXlI4p4VKbw",  # 中天新聞
    "UC5nwNW4KdC0SzrhF9BXEYOQ",  # TVBS
    "UCIU8ha-NHmLjtUwU7dFiXUA",  # 三立新聞
    "UC2VmWn8dAqkzlQqvy02E1PA",  # 民視新聞
]
QUOTA_DAILY_LIMIT: int = 9_000   # 留 1,000 作安全緩衝

# ── 路徑 ─────────────────────────────────────────────
DATA_DIR: str     = "data"
COLLECTED_DIR: str = "data/collected"
MODEL_DIR: str    = "data/models"
RESULTS_DIR: str  = "results"
COLLECTED_IDS_PATH: str = "data/collected/collected_ids.json"

# ── 資料收集 ──────────────────────────────────────────
LOOKBACK_DAYS: int = 7

# ── 特徵工程 ──────────────────────────────────────────
AD_KEYWORDS: list[str] = [
    "訂閱", "點擊連結", "私訊我", "加LINE", "賺錢",
    "免費領取", "點我", "私我", "合作", "業配",
]
NIGHT_HOURS: tuple[int, int] = (0, 6)   # [0, 6) 即 0~5 點

# ── 自動標注閾值 ───────────────────────────────────────
BOT_CONCENTRATION_GT: float   = 5.0
BOT_SELF_SIMILARITY_GT: float = 0.7
BOT_ZERO_LIKE_RATIO_GT: float = 0.8
BOT_INTERVAL_STD_LT: float    = 60.0

NORMAL_UNIQUE_VIDEOS_GE: int  = 3
NORMAL_SELF_SIMILARITY_LT: float = 0.3
NORMAL_ZERO_LIKE_RATIO_LT: float = 0.5

# ── 第一階段訓練 ───────────────────────────────────────
STAGE1_TEST_SIZE: float  = 0.2
STAGE1_RANDOM_STATE: int = 42
STAGE1_CV_FOLDS: int     = 5

FEATURE_COLS: list[str] = [
    "total_comments", "unique_videos", "concentration",
    "avg_likes", "zero_like_ratio", "max_burst",
    "avg_length", "unique_content_ratio", "self_similarity",
    "has_ad_keywords", "url_ratio",
    "night_ratio", "interval_std",
]

# ── 第二階段訓練 ───────────────────────────────────────
BERT_MODEL_NAME: str     = "hfl/chinese-macbert-base"
BERT_MAX_LEN: int        = 128
BERT_BATCH_SIZE: int     = 16
BERT_EPOCHS: int         = 3
BERT_LR: float           = 2e-5
BERT_WARMUP_RATIO: float = 0.1
```

### 測試要點

- `YOUTUBE_API_KEY` 在 `.env` 缺失時，import config 應拋出 `KeyError`（啟動失敗優於靜默失敗）。

---

## 4. 模組 1：collector.py — 資料收集

### 職責

從 YouTube Data API v3 收集指定頻道過去 `LOOKBACK_DAYS` 天內的影片及其所有留言（含回覆）。採**增量追加**策略：每次執行以當日日期命名輸出一個新的 CSV，並透過 `collected_ids.json` 跳過已收集的影片與留言。

### 輸入

| 來源 | 說明 |
|------|------|
| `config.CHANNEL_IDS` | 目標頻道清單 |
| `config.YOUTUBE_API_KEY` | API 金鑰 |
| `data/collected/collected_ids.json` | 已收集 ID（可不存在，首次執行時自動建立） |

### 輸出

| 檔案 | 說明 |
|------|------|
| `data/collected/youtube_raw_YYYYMMDD.csv` | 當日收集的影片 + 留言資料（兩個 sheet 合為一檔，見下方 schema） |
| `data/collected/collected_ids.json` | 更新後的已收集 ID 集合 |

### 輸出 Schema

**videos 欄位（type=video 的列）：**

| 欄位 | 類型 | 說明 |
|------|------|------|
| `record_type` | str | 固定值 `"video"` |
| `video_id` | str | 影片唯一識別碼 |
| `video_title` | str | 影片標題 |
| `channel_id` | str | 頻道 ID |
| `published_at` | ISO8601 str | 影片發佈時間（UTC） |
| `view_count` | int | 觀看數 |
| `comment_count` | int | 留言總數（API 提供值） |

**comments 欄位（type=comment 的列）：**

| 欄位 | 類型 | 說明 |
|------|------|------|
| `record_type` | str | `"comment"` 或 `"reply"` |
| `comment_id` | str | 留言唯一識別碼 |
| `video_id` | str | 所屬影片 |
| `author_id` | str | 作者頻道 ID |
| `author_name` | str | 作者顯示名稱 |
| `content` | str | 留言文字內容 |
| `like_count` | int | 按讚數 |
| `reply_count` | int | 回覆數（僅頂層留言有值，回覆為 0） |
| `published_at` | ISO8601 str | 留言時間（UTC） |
| `parent_id` | str | 父留言 ID（僅 reply 有值，否則空字串） |

> **設計說明：** 影片資料與留言資料合存於同一 CSV，以 `record_type` 欄位區分，避免管理兩份檔案。後續模組讀取時以 `df[df.record_type == "comment"]` 篩選。

### collected_ids.json 格式

```json
{
  "video_ids": ["abc123", "def456"],
  "comment_ids": ["Ugxxx", "Ugyyy"]
}
```

### API 配額估算

| API 呼叫 | 單位數/次 | 說明 |
|----------|-----------|------|
| `channels.list` | 1 | 取得 uploads playlist ID |
| `playlistItems.list` | 1 | 每頁 50 支影片 |
| `videos.list` | 1 | 每批 50 支影片取得詳細資訊 |
| `commentThreads.list` | 1 | 每頁 100 則頂層留言 |
| `comments.list` | 1 | 每頁 100 則回覆 |

**保守估算：** 4 個頻道 × 7 天 × 平均 3 支影片 × 每影片 300 則留言 ≈ 3,600 單位。安全邊際充足。

### 公開 API

```python
def load_collected_ids(path: str) -> dict[str, set[str]]:
    """
    回傳 {"video_ids": set, "comment_ids": set}。
    檔案不存在時回傳空集合，不拋例外。
    """

def save_collected_ids(ids: dict[str, set[str]], path: str) -> None:
    """將 set 序列化為 list 後存入 JSON。"""

def fetch_channel_uploads_playlist(service, channel_id: str) -> str:
    """回傳該頻道的 uploads playlist ID。"""

def fetch_recent_videos(
    service,
    playlist_id: str,
    since: datetime,
    collected_video_ids: set[str],
) -> list[dict]:
    """
    取得 since 之後的影片，跳過已收集 ID。
    回傳欄位：video_id, video_title, channel_id, published_at。
    """

def fetch_video_stats(service, video_ids: list[str]) -> dict[str, dict]:
    """批次取得 view_count, comment_count，每批最多 50 個。"""

def fetch_comments(
    service,
    video_id: str,
    collected_comment_ids: set[str],
) -> list[dict]:
    """
    取得頂層留言及其回覆，跳過已收集 ID。
    回傳符合 comment schema 的 list[dict]。
    """

def check_quota(used: int, limit: int) -> None:
    """used + 預估本輪消耗 >= limit 時拋 QuotaExceededError。"""

def run_collection(
    channel_ids: list[str],
    output_dir: str,
    collected_ids_path: str,
    quota_limit: int = 9_000,
) -> str:
    """
    主執行函式。
    回傳本次輸出的 CSV 路徑（data/collected/youtube_raw_YYYYMMDD.csv）。
    若當日已存在同名檔案則以時間戳後綴命名（youtube_raw_20260603_143022.csv）。
    """
```

### 自訂例外

```python
class QuotaExceededError(Exception):
    """API 配額耗盡，包含 used / limit 資訊。"""
```

### 執行方式

```bash
python collector.py
```

### 測試要點

- `load_collected_ids` 在檔案不存在時回傳空集合，不拋例外。
- `fetch_recent_videos` 正確過濾 `since` 時間，跳過已收集影片。
- `fetch_comments` 正確展開回覆（reply），parent_id 正確填入。
- 模擬 API 回應（mock `googleapiclient`），不需真實 API 金鑰。
- `check_quota` 在達到上限前拋出例外，已收集資料仍寫入磁碟。

---

## 5. 模組 2：feature_engineering.py — 特徵提取

### 職責

載入 `data/collected/` 下所有 `youtube_raw_*.csv`，合併去重後，以過去 `LOOKBACK_DAYS` 天的留言資料，為每位用戶（`author_id`）計算 13 個特徵。

### 輸入

| 來源 | 說明 |
|------|------|
| `data/collected/youtube_raw_*.csv` | 所有收集的原始資料 |
| `config.LOOKBACK_DAYS` | 回溯天數（預設 7） |

### 輸出

| 檔案 | Schema |
|------|--------|
| `data/user_features.csv` | `author_id` + 13 個特徵欄位（見下表） |

### 特徵定義

| 特徵名稱 | 類型 | 計算方式 |
|----------|------|----------|
| `total_comments` | int | 7 天內總留言數 |
| `unique_videos` | int | 留言涉及的不同 video_id 數 |
| `concentration` | float | `total_comments / unique_videos`（unique_videos=0 時填 0） |
| `avg_likes` | float | 留言 like_count 的平均值 |
| `zero_like_ratio` | float | like_count == 0 的留言比例 |
| `max_burst` | int | 任一小時內的最大留言數（以 published_at 的小時為單位） |
| `avg_length` | float | 留言 content 字數（以 `len()` 計）的平均值 |
| `unique_content_ratio` | float | `不重複留言數 / total_comments` |
| `self_similarity` | float | TF-IDF 向量兩兩 cosine similarity 的均值；只有 1 則留言時填 0.0 |
| `has_ad_keywords` | int | 任一留言含廣告關鍵字（見 config）則為 1，否則為 0 |
| `url_ratio` | float | 含 URL（`http://`、`https://`、`www.`）的留言比例 |
| `night_ratio` | float | published_at 小時在 \[0, 6\) 的留言比例 |
| `interval_std` | float | 留言時間間隔（秒）的標準差；只有 1 則留言時填 0.0 |

### `self_similarity` 計算細節

1. 取出該用戶所有留言的 `content` 列表。
2. 若 `len(contents) < 2`：回傳 0.0。
3. 以 `TfidfVectorizer(analyzer='char', ngram_range=(1,2))` 對所有留言做向量化（避免詞語切分問題，以字符 n-gram 處理中文）。
4. 計算向量矩陣的 `cosine_similarity`，取上三角（排除對角線）的平均值。

### `max_burst` 計算細節

```
floor(published_at to hour) → group by (author_id, hour_bucket) → count → max per author
```

### 公開 API

```python
def load_raw_comments(data_dir: str, lookback_days: int) -> pd.DataFrame:
    """
    讀取所有 youtube_raw_*.csv，篩選 record_type in ("comment","reply")，
    過濾至今日起算 lookback_days 天內的留言，
    依 comment_id 去重後回傳。
    """

def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """輸入：留言 DataFrame（含 author_id, video_id, like_count, published_at）。
    輸出：author_id 為 index，含 total_comments, unique_videos, concentration,
          avg_likes, zero_like_ratio, max_burst 的 DataFrame。"""

def compute_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """輸入同上，另需 content 欄位。
    輸出：含 avg_length, unique_content_ratio, self_similarity,
          has_ad_keywords, url_ratio 的 DataFrame。"""

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """輸出：含 night_ratio, interval_std 的 DataFrame。"""

def run_feature_engineering(
    data_dir: str,
    output_path: str,
    lookback_days: int = 7,
) -> None:
    """主執行函式，合併三組特徵後寫入 output_path。"""
```

### 執行方式

```bash
python feature_engineering.py
```

### 測試要點

- `load_raw_comments` 正確去重（相同 comment_id 只保留一筆）。
- `compute_content_features` 對只有 1 則留言的用戶，`self_similarity` 輸出 0.0。
- `compute_temporal_features` 對只有 1 則留言的用戶，`interval_std` 輸出 0.0。
- 對 `unique_videos = 0` 的極端案例（不應出現，但需防守），`concentration` 填 0。
- `night_ratio` 以 UTC 計算（不轉時區），與收集時時區一致。

---

## 6. 模組 3：auto_labeler.py — 自動標注

### 職責

對 `user_features.csv` 中每位用戶套用啟發式規則，輸出：
- `labeled_data.csv`：確定為水軍（`is_bot=1`）或確定為正常（`is_bot=0`）的樣本。
- `pending_review.csv`：不明確樣本，供人工填寫。

### 輸入

| 來源 | 說明 |
|------|------|
| `data/user_features.csv` | 所有用戶的 13 個特徵 |

### 輸出

| 檔案 | Schema |
|------|--------|
| `data/labeled_data.csv` | user_features 所有欄位 + `is_bot`（0 或 1） |
| `data/pending_review.csv` | user_features 所有欄位 + `is_bot`（空字串，待人工填寫） |

### 標注規則

**水軍（is_bot = 1）**：以下四個條件**全部**滿足：

```
concentration    > BOT_CONCENTRATION_GT    (5.0)
self_similarity  > BOT_SELF_SIMILARITY_GT  (0.7)
zero_like_ratio  > BOT_ZERO_LIKE_RATIO_GT  (0.8)
interval_std     < BOT_INTERVAL_STD_LT     (60.0)
```

**正常用戶（is_bot = 0）**：以下三個條件**全部**滿足：

```
unique_videos   >= NORMAL_UNIQUE_VIDEOS_GE        (3)
self_similarity <  NORMAL_SELF_SIMILARITY_LT      (0.3)
zero_like_ratio <  NORMAL_ZERO_LIKE_RATIO_LT      (0.5)
```

**待人工複審（pending）**：不滿足上述任一群條件的用戶。

> **注意：** 若同時滿足水軍與正常條件（理論上不可能，因閾值方向相反），優先標記為 bot。

### 公開 API

```python
def classify_user(row: pd.Series) -> str:
    """
    回傳 'bot'、'normal' 或 'pending'。
    純函式，不依賴全域狀態，易於單元測試。
    """

def run_labeling(
    features_path: str,
    labeled_output: str,
    pending_output: str,
) -> dict[str, int]:
    """
    主執行函式。
    回傳統計 {"bot": N, "normal": N, "pending": N}。
    """
```

### 執行方式

```bash
python auto_labeler.py
```

執行後輸出統計：
```
[auto_labeler] bot=82, normal=341, pending=127
已寫入 data/labeled_data.csv (423 筆)
已寫入 data/pending_review.csv (127 筆)
```

### 測試要點

- `classify_user` 對邊界值（例如 `concentration=5.0` 不算水軍，`>5.0` 才算）行為正確。
- 輸出 `pending_review.csv` 的 `is_bot` 欄位為空字串（非 NaN），便於 Excel 編輯。
- `labeled_data.csv` 不含任何空 `is_bot` 值。

---

## 7. 模組 4：merge_labels.py — 人工複審合併

### 職責

人工複審完成後，將 `pending_review.csv` 中已填寫 `is_bot` 的列合併入 `labeled_data.csv`，跳過仍為空的列。

### 前置條件

研究者在 `pending_review.csv` 的 `is_bot` 欄位填入 `0` 或 `1`（可部分填寫）。

### 輸入

| 來源 | 說明 |
|------|------|
| `data/labeled_data.csv` | 現有標注資料 |
| `data/pending_review.csv` | 人工複審後的待複審檔案 |

### 輸出

| 檔案 | 說明 |
|------|------|
| `data/labeled_data.csv` | 合併後的完整標注集（原地更新） |

### 合併邏輯

1. 讀取 `pending_review.csv`，過濾 `is_bot` 為 `0` 或 `1` 的列（轉換為 int）。
2. 以 `author_id` 為鍵，檢查是否與 `labeled_data.csv` 重複（若已存在則跳過並警告）。
3. 合併後去重（以 `author_id` 為準）。
4. 覆寫 `data/labeled_data.csv`。

### 公開 API

```python
def merge_reviewed_labels(
    labeled_path: str,
    pending_path: str,
    output_path: str,
) -> dict[str, int]:
    """
    回傳 {"merged": N, "skipped_empty": N, "skipped_duplicate": N}。
    """
```

### 執行方式

```bash
python merge_labels.py
```

### 測試要點

- `is_bot` 為空字串或 NaN 的列不納入合併。
- `is_bot` 非 0/1 的值（例如 `2`、`yes`）拋出 `ValueError`，終止執行。
- 重複 `author_id` 只保留 `labeled_data.csv` 中既有的版本，並在 stderr 輸出警告。

---

## 8. 模組 5：train_stage1.py — 傳統 ML 訓練

### 職責

以 `labeled_data.csv` 訓練 Logistic Regression 和 XGBoost 分類器，輸出模型檔與 5-fold 交叉驗證報告。

### 輸入

| 來源 | 說明 |
|------|------|
| `data/labeled_data.csv` | 標注資料（需 ≥ 50 筆，否則終止） |
| `config.FEATURE_COLS` | 13 個特徵欄位名稱 |

### 輸出

| 檔案 | 說明 |
|------|------|
| `data/models/stage1/lr_model.pkl` | Logistic Regression Pipeline（含 StandardScaler） |
| `data/models/stage1/xgb_model.pkl` | XGBoost 模型 |
| `data/models/stage1/scaler.pkl` | StandardScaler（獨立存放，供 evaluate.py 使用） |
| `results/stage1_report.json` | 交叉驗證指標 |

### 模型設計

#### Logistic Regression

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,        # GridSearchCV 搜尋範圍：[0.01, 0.1, 1.0, 10.0]
        solver="lbfgs",
    ))
])
```

#### XGBoost

```python
XGBClassifier(
    n_estimators=200,       # GridSearchCV：[100, 200, 300]
    max_depth=4,            # GridSearchCV：[3, 4, 6]
    learning_rate=0.1,      # GridSearchCV：[0.05, 0.1, 0.2]
    scale_pos_weight=neg/pos,  # 根據訓練集比例自動計算
    eval_metric="logloss",
    random_state=42,
)
```

### 訓練流程

1. 讀取資料，以 `is_bot` 為標籤，`FEATURE_COLS` 為特徵。
2. `StratifiedShuffleSplit(test_size=0.2)` 切分 train/test（保存索引供 evaluate.py 重現）。
3. 對訓練集進行 5-fold `StratifiedKFold` GridSearchCV，選出最佳超參數。
4. 以最佳超參數在完整訓練集訓練最終模型。
5. 輸出模型與交叉驗證結果。

### `stage1_report.json` 格式

```json
{
  "train_size": 338,
  "test_size": 85,
  "lr": {
    "best_params": {"clf__C": 1.0},
    "cv_f1_mean": 0.83,
    "cv_f1_std": 0.04
  },
  "xgb": {
    "best_params": {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.1},
    "cv_f1_mean": 0.87,
    "cv_f1_std": 0.03
  },
  "data_split_indices": {
    "test_indices": [12, 45, 67, "..."]
  }
}
```

> `data_split_indices` 存放測試集索引，供 `evaluate.py` 使用相同的 test split 評估。

### 公開 API

```python
def load_labeled_data(
    labeled_path: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """回傳 (X, y)，y 為 is_bot。"""

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict]:
    """回傳 (最佳模型, best_params)。"""

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[XGBClassifier, dict]:
    """回傳 (最佳模型, best_params)。"""

def run_training(
    labeled_path: str,
    model_dir: str,
    results_dir: str,
) -> None:
    """主執行函式。"""
```

### 執行方式

```bash
python train_stage1.py
```

### 測試要點

- 訓練集樣本數 < 50 時，以清楚訊息終止並不輸出模型。
- 模型能正確序列化（`joblib.dump`）與還原（`joblib.load`）。
- `scale_pos_weight` 在正負樣本比不同時正確計算。

---

## 9. 模組 6：train_stage2.py — BERT 微調

### 職責

以用戶層級標籤（`labeled_data.csv`）推衍出留言層級標籤，微調中文 BERT 模型，輸出留言層級水軍分數。

### 標籤推衍原則

**假設：** 水軍用戶（`is_bot=1`）的所有留言均視為 `is_spam=1`；正常用戶（`is_bot=0`）的所有留言視為 `is_spam=0`。  
此假設在論文中需明確說明其侷限性。

### 輸入

| 來源 | 說明 |
|------|------|
| `data/labeled_data.csv` | 用戶層級標籤 |
| `data/collected/youtube_raw_*.csv` | 留言文字內容 |

### 輸出

| 檔案 | 說明 |
|------|------|
| `data/models/stage2/bert/` | 微調後的 Hugging Face 模型（`save_pretrained` 格式） |
| `data/bert_comment_scores.csv` | `comment_id, author_id, spam_prob`（全資料集推論結果） |
| `results/stage2_report.json` | 驗證集指標 |

### 資料集設計

```python
class CommentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],     # is_spam: 0 or 1
        tokenizer,
        max_len: int = 128,
    ):
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...
    # 回傳 {"input_ids", "attention_mask", "labels"}，皆為 torch.Tensor
```

### 訓練流程

1. 讀取 `labeled_data.csv` 取得 `{author_id: is_bot}` 映射。
2. 讀取所有原始留言，以 `author_id` 關聯取得 `is_spam`（user is_bot → comment is_spam）。
3. 移除 `author_id` 不在 `labeled_data.csv` 的留言（unlabeled）。
4. Stratified 80/10/10 分割（train / val / test）。
5. 計算正負樣本比，作為 `pos_weight` 傳入 `BCEWithLogitsLoss`。
6. 載入 `BERT_MODEL_NAME` 的 tokenizer 與 `AutoModelForSequenceClassification(num_labels=1)`（二元分類）。
7. AdamW optimizer + 線性 warmup scheduler（warmup = `BERT_WARMUP_RATIO × total_steps`）。
8. 每個 epoch 後在 val 集計算 F1，若連續 2 個 epoch 無改善則 early stop。
9. 以最佳 val F1 的 checkpoint 儲存模型。

### `stage2_report.json` 格式

```json
{
  "model_name": "hfl/chinese-macbert-base",
  "train_samples": 2800,
  "val_samples": 350,
  "test_samples": 350,
  "best_epoch": 3,
  "val_f1": 0.81,
  "val_precision": 0.83,
  "val_recall": 0.79
}
```

### 公開 API

```python
def build_comment_label_df(
    labeled_path: str,
    raw_dir: str,
) -> pd.DataFrame:
    """
    合併用戶標籤與留言內容，回傳含 comment_id, author_id, content, is_spam 的 DataFrame。
    只包含有標籤的用戶留言。
    """

def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    epochs: int = 3,
) -> dict:
    """回傳最佳驗證指標 dict。"""

def run_inference(
    model_dir: str,
    texts: list[str],
    batch_size: int = 32,
) -> list[float]:
    """對文字列表推論，回傳 spam_prob 列表（0.0~1.0）。"""

def run_training(
    labeled_path: str,
    raw_dir: str,
    model_dir: str,
    results_dir: str,
) -> None:
    """主執行函式。"""
```

### 執行方式

```bash
python train_stage2.py
# GPU 環境（建議）：
CUDA_VISIBLE_DEVICES=0 python train_stage2.py
```

### 測試要點

- `build_comment_label_df` 正確排除 `author_id` 不在 `labeled_data.csv` 的留言。
- `CommentDataset.__getitem__` 回傳 tensor shape 正確（`input_ids`: `[max_len]`）。
- 當 GPU 不可用時自動 fallback 到 CPU（`device = torch.device("cuda" if available else "cpu")`）。
- `run_inference` 輸出機率值在 \[0, 1\] 範圍內（過 sigmoid）。

---

## 10. 模組 7：evaluate.py — 模型評估

### 職責

載入所有已訓練的模型，對相同的測試集進行評估，輸出各模型的量化比較報告。

### 輸入

| 來源 | 說明 |
|------|------|
| `data/labeled_data.csv` | 標注資料（含 FEATURE_COLS + is_bot） |
| `results/stage1_report.json` | 讀取 test_indices 以重現相同 test split |
| `data/models/stage1/lr_model.pkl` | LR 模型 |
| `data/models/stage1/xgb_model.pkl` | XGBoost 模型 |
| `data/models/stage2/bert/` | BERT 模型（可選，不存在時跳過） |
| `data/collected/youtube_raw_*.csv` | 供 BERT 推論使用 |

### 輸出

| 檔案 | 說明 |
|------|------|
| `results/evaluation_report.json` | 各模型完整指標 |
| `results/comparison_report.csv` | 橫向比較表（模型 × 指標） |

### 計算指標

對每個模型計算以下指標（均以測試集為準）：

| 指標 | 說明 |
|------|------|
| `precision` | sklearn `precision_score(average='binary')` |
| `recall` | sklearn `recall_score(average='binary')` |
| `f1` | sklearn `f1_score(average='binary')` |
| `auc_roc` | sklearn `roc_auc_score` |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]` |

### BERT 評估邏輯

BERT 輸出留言層級分數（`spam_prob`）。評估時需聚合為用戶層級：

```
user_bot_prob = mean(spam_prob for all comments by that user in test set)
threshold = 0.5 → is_bot_pred = int(user_bot_prob >= 0.5)
```

### 評估報告格式

```json
{
  "test_size": 85,
  "models": {
    "logistic_regression": {
      "precision": 0.79,
      "recall": 0.76,
      "f1": 0.77,
      "auc_roc": 0.88,
      "confusion_matrix": [[70, 5], [8, 2]]
    },
    "xgboost": {
      "precision": 0.85,
      "recall": 0.80,
      "f1": 0.82,
      "auc_roc": 0.91,
      "confusion_matrix": [[...]]]
    },
    "bert_aggregated": {
      "precision": 0.83,
      "recall": 0.82,
      "f1": 0.82,
      "auc_roc": 0.90,
      "confusion_matrix": [[...]]
    }
  }
}
```

### 公開 API

```python
def load_test_split(
    labeled_path: str,
    stage1_report_path: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """以 stage1_report.json 記錄的索引還原 test split，回傳 (X_test, y_test)。"""

def evaluate_sklearn_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """評估 sklearn/XGBoost 模型，回傳指標 dict。"""

def evaluate_bert_model(
    bert_model_dir: str,
    test_author_ids: list[str],
    raw_dir: str,
    y_test: pd.Series,
) -> dict:
    """聚合 BERT 留言分數至用戶層級，回傳指標 dict。"""

def run_evaluation(
    labeled_path: str,
    model_dir: str,
    raw_dir: str,
    results_dir: str,
) -> None:
    """主執行函式。"""
```

### 執行方式

```bash
python evaluate.py
```

### 測試要點

- 使用 `stage1_report.json` 的索引，確保 LR / XGBoost / BERT 三個模型評估在**完全相同的 test set** 上進行。
- BERT 模型目錄不存在時，跳過 BERT 評估並在報告中標示 `"bert_aggregated": null`，不拋例外。
- `comparison_report.csv` 行為模型、列為指標，數值保留 4 位小數。

---

## 11. 模組 8：fusion.py — 多模態融合（選配）

> **狀態：選配（Advanced）** — 前兩階段模型訓練完成後方可執行。

### 職責

將第一階段（帳號行為特徵）與第二階段（BERT 留言語意）的輸出合併，訓練一個融合分類器，以提升協同操控偵測能力。

### 前置條件

- `data/models/stage1/lr_model.pkl` 與 `xgb_model.pkl` 已存在。
- `data/models/stage2/bert/` 已存在。
- `data/bert_comment_scores.csv` 已存在（`train_stage2.py` 的推論輸出）。

### 融合特徵設計

對每個用戶，構建以下融合特徵向量（共 5 維）：

| 特徵 | 來源 |
|------|------|
| `lr_prob` | LR 對該用戶的預測機率（`predict_proba`） |
| `xgb_prob` | XGBoost 對該用戶的預測機率 |
| `bert_user_prob` | 該用戶所有留言 spam_prob 的平均值 |
| `bert_max_prob` | 該用戶留言 spam_prob 的最大值 |
| `bert_comment_count` | 該用戶留言數（校正樣本量） |

### 融合模型

```python
# 元學習器（Stacking）
LogisticRegression(class_weight="balanced", max_iter=500)
```

以 Stage 1 的相同 test split 作為 hold-out 評估集；在其餘資料上以 5-fold CV 訓練融合模型，防止 data leakage。

### 輸出

| 檔案 | 說明 |
|------|------|
| `data/models/stage3/fusion_model.pkl` | 融合 LR 模型 |
| `results/stage3_report.json` | 融合模型評估指標 |

### GNN 擴展方向（備忘）

若後續要加入 GNN：
- **圖構建：** 節點為 author_id，邊為「在同一影片、同一 24 小時窗口內留言」的用戶對。
- **節點特徵：** `FEATURE_COLS` 中的 13 個行為特徵。
- **GNN 類型：** GraphSAGE（適合 inductive setting）。
- **函式庫：** `torch_geometric`。

---

## 12. 模組間資料流

```
.env
  │
  └─ config.py ──────────────────────────────────────┐
                                                     │ (全域常數)
collector.py ──→ data/collected/youtube_raw_*.csv    │
                         │                           │
                         ▼                           │
feature_engineering.py ──→ data/user_features.csv   │
                                    │                │
                                    ▼                │
auto_labeler.py ─┬─→ data/labeled_data.csv           │
                 └─→ data/pending_review.csv          │
                              │ (人工編輯後)           │
                              ▼                      │
merge_labels.py ──→ data/labeled_data.csv (更新)     │
                              │                      │
               ┌──────────────┼──────────────┐       │
               ▼              ▼              ▼       │
train_stage1.py      train_stage2.py    (labeled_data)
       │                    │
data/models/stage1/   data/models/stage2/bert/
       │               data/bert_comment_scores.csv
       └──────────────────┬──────────────────┘
                          ▼
                    evaluate.py
                          │
                results/evaluation_report.json
                results/comparison_report.csv
                          │
                          ▼ (選配)
                    fusion.py
                          │
                results/stage3_report.json
```

---

## 13. 錯誤處理策略

### 系統邊界（需驗證的輸入）

| 邊界 | 驗證方式 |
|------|---------|
| `YOUTUBE_API_KEY` 環境變數 | `config.py` import 時 `os.environ["YOUTUBE_API_KEY"]` 強制取值 |
| CSV 欄位完整性 | 每個模組讀取後立即 `assert set(REQUIRED_COLS).issubset(df.columns)` |
| 最小樣本數 | `train_stage1.py` / `train_stage2.py` 在訓練前檢查 len(df) >= 50 |
| 人工複審填寫值 | `merge_labels.py` 驗證 is_bot ∈ {0, 1, ""} |

### 不可恢復錯誤

直接 `raise` 並附上清楚訊息，終止執行：
- API 金鑰遺失 / 無效
- 必要輸入檔案不存在（例如 `user_features.csv` 不存在時執行 `auto_labeler.py`）
- 訓練資料不足

### 可恢復錯誤

記錄 warning 並繼續：
- 單一影片的留言 API 失敗（記錄後跳過，不終止整個 collection）
- `evaluate.py` 發現 BERT 模型不存在（跳過 BERT 評估）
- `merge_labels.py` 發現重複 `author_id`（保留原有標籤，輸出警告）

### API 配額耗盡

`collector.py` 捕捉 `QuotaExceededError`，將已收集資料寫入 CSV 後正常結束，並在 stderr 輸出剩餘未收集的頻道清單。

---

## 14. 測試策略

每個模組的測試均放在 `tests/test_<module>.py`，使用 `pytest`，**不依賴真實 API、GPU 或大型模型**。

### 測試原則

1. 純函式優先：`classify_user`、`compute_behavioral_features` 等純函式以小型 DataFrame 單元測試。
2. API 呼叫全部 mock：使用 `unittest.mock.patch` 替換 `googleapiclient`。
3. BERT 測試使用 `distilbert-base-multilingual-cased`（小型、快速）替換正式模型。
4. 不測試 GridSearchCV 的超參數搜尋結果（隨機性高），只測試訓練流程能跑完且輸出正確格式。

### 測試資料

- `tests/fixtures/sample_raw.csv`：30 筆假留言，涵蓋邊界值（單一留言用戶、夜間發文、含廣告關鍵字）。
- `tests/fixtures/sample_labeled.csv`：20 位用戶，bot/normal/pending 各佔一部分。

### 整合測試

`tests/test_pipeline.py`：以 fixture 資料完整執行 collector 以外的所有模組，驗證：
- 每個模組的輸出檔案存在且格式正確。
- `evaluate.py` 能正確計算各指標（不要求達到目標值，只驗證格式）。
