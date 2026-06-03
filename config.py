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
QUOTA_DAILY_LIMIT: int = 9_000

# ── 路徑 ─────────────────────────────────────────────
DATA_DIR: str       = "data"
COLLECTED_DIR: str  = "data/collected"
MODEL_DIR: str      = "data/models"
RESULTS_DIR: str    = "results"
COLLECTED_IDS_PATH: str = "data/collected/collected_ids.json"

# ── 資料收集 ──────────────────────────────────────────
LOOKBACK_DAYS: int = 7

# ── 特徵工程 ──────────────────────────────────────────
AD_KEYWORDS: list[str] = [
    "訂閱", "點擊連結", "私訊我", "加LINE", "賺錢",
    "免費領取", "點我", "私我", "合作", "業配",
]
NIGHT_HOURS: tuple[int, int] = (0, 6)

# ── 自動標注閾值 ───────────────────────────────────────
BOT_CONCENTRATION_GT: float   = 5.0
BOT_SELF_SIMILARITY_GT: float = 0.7
BOT_ZERO_LIKE_RATIO_GT: float = 0.8
BOT_INTERVAL_STD_LT: float    = 60.0

NORMAL_UNIQUE_VIDEOS_GE: int     = 3
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
