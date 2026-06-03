# config.py — 全域設定

## 前置條件

無（此模組最先實作）

## 子任務

- [ ] **C1** 建立 `.env.example`：內含 `YOUTUBE_API_KEY=your_key_here` 一行，作為 API 金鑰範本
- [ ] **C2** 建立 `.gitignore`：排除 `.env`、`data/`、`results/`、`__pycache__/`、`*.pkl`、`*.pyc`
- [ ] **C3** 建立目錄骨架：`data/collected/`、`data/models/stage1/`、`data/models/stage2/`、`data/models/stage3/`、`results/`、`tests/`（各含空的 `.gitkeep`）
- [ ] **C4** 實作 `config.py`：
  - 用 `python-dotenv` 讀取 `.env`
  - 定義 `YOUTUBE_API_KEY`（缺失時 `os.environ["YOUTUBE_API_KEY"]` 拋 `KeyError`）
  - 定義 `CHANNEL_IDS`（4 個頻道）
  - 定義所有路徑常數（`DATA_DIR`、`COLLECTED_DIR`、`MODEL_DIR`、`RESULTS_DIR`、`COLLECTED_IDS_PATH`）
  - 定義特徵工程常數（`AD_KEYWORDS`、`NIGHT_HOURS`、`LOOKBACK_DAYS`）
  - 定義標注閾值（`BOT_*`、`NORMAL_*`）
  - 定義 Stage1 訓練參數（`STAGE1_TEST_SIZE`、`STAGE1_RANDOM_STATE`、`STAGE1_CV_FOLDS`、`FEATURE_COLS`）
  - 定義 Stage2 訓練參數（`BERT_MODEL_NAME`、`BERT_MAX_LEN`、`BERT_BATCH_SIZE`、`BERT_EPOCHS`、`BERT_LR`、`BERT_WARMUP_RATIO`）
- [ ] **C5** 建立 `requirements.txt`：列出 `google-api-python-client`、`python-dotenv`、`pandas`、`numpy`、`scikit-learn`、`xgboost`、`transformers`、`torch`、`pytest`

## 驗收條件

- 執行 `python -c "import config"` 在 `.env` 存在時成功
- 刪除 `.env` 後執行同指令拋出 `KeyError: 'YOUTUBE_API_KEY'`
