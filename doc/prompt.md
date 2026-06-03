# AstroSentinel — Vibe Coding 主 Agent 起始 Prompt

> **角色：主 Agent（進度協調員）**  
> 你是本次 Vibe Coding 的主 Agent。你的職責是按照依賴順序生成子 Agent，讓每個子 Agent 完整實作指定模組（含單元測試），再確認完成後繼續下一模組。整個過程**不需要任何人工介入**。

---

## 一、專案概覽

**專案名稱：** AstroSentinel  
**性質：** 學術研究系統，偵測 YouTube 留言區的網路水軍行為  
**工作目錄：** `c:\Users\noahl\Documents\AstroSentinel`  
**平台：** Windows 11，使用 PowerShell

### 系統組成（8 個模組 + 測試基礎設施）

```
config.py           ← 全域設定、路徑常數、環境變數
collector.py        ← YouTube Data API v3 資料收集（增量）
feature_engineering.py  ← 13 個用戶行為/內容/時間特徵
auto_labeler.py     ← 啟發式規則自動標注
merge_labels.py     ← 人工複審標籤合併
train_stage1.py     ← Logistic Regression + XGBoost 訓練
train_stage2.py     ← 中文 BERT 微調（hfl/chinese-macbert-base）
evaluate.py         ← 多模型統一評估報告
fusion.py           ← 多模態融合（選配，本次需實作）
tests/              ← pytest 測試套件
```

### 資料流向

```
.env (YOUTUBE_API_KEY)
  │
  └─ config.py ─────────────────────────────────────┐
                                                    │
collector.py ──→ data/collected/youtube_raw_*.csv   │
                          │                         │
feature_engineering.py ───┘→ data/user_features.csv│
                                      │             │
auto_labeler.py ──→ data/labeled_data.csv           │
                 └→ data/pending_review.csv          │
                              │                     │
merge_labels.py ──→ data/labeled_data.csv (更新)    │
                              │                     │
          ┌───────────────────┤                     │
          ▼                   ▼                     │
  train_stage1.py      train_stage2.py  ────────────┘
  lr_model.pkl         bert_model/
  xgb_model.pkl        bert_comment_scores.csv
          │                   │
          └────────┬──────────┘
                   ▼
             evaluate.py
                   │
             evaluation_report.json
             comparison_report.csv
                   │
                   ▼ (選配，本次需實作)
             fusion.py
                   │
             stage3_report.json
```

---

## 二、技術環境

| 項目 | 規格 |
|------|------|
| Python | 3.10+，使用 `uv` 管理環境 |
| 套件管理 | `uv`（執行指令用 `uv run python xxx.py`） |
| 測試框架 | `pytest`（執行 `uv run pytest tests/`） |
| 主要依賴 | pandas, numpy, scikit-learn, xgboost, transformers, torch, google-api-python-client, python-dotenv |
| API 金鑰 | `.env` 已有真實 `YOUTUBE_API_KEY` |
| GPU | 若可用自動使用，否則 CPU fallback |

---

## 三、設計文件位置

執行前請先閱讀以下文件，它們包含完整的介面定義、Schema 與測試要點：

| 文件 | 說明 |
|------|------|
| `doc/proposal.md` | 研究背景、特徵定義、ML 方法 |
| `doc/detailed-design.md` | 每個模組的完整介面、Schema、錯誤處理、測試策略 |
| `doc/tasks/config.md` | config 模組子任務清單 |
| `doc/tasks/collector.md` | collector 模組子任務清單 |
| `doc/tasks/feature_engineering.md` | feature_engineering 模組子任務清單 |
| `doc/tasks/auto_labeler.md` | auto_labeler 模組子任務清單 |
| `doc/tasks/merge_labels.md` | merge_labels 模組子任務清單 |
| `doc/tasks/train_stage1.md` | train_stage1 模組子任務清單 |
| `doc/tasks/train_stage2.md` | train_stage2 模組子任務清單 |
| `doc/tasks/evaluate.md` | evaluate 模組子任務清單 |
| `doc/tasks/fusion.md` | fusion 模組子任務清單（選配，本次需實作） |
| `doc/tasks/tests.md` | 測試基礎設施子任務清單 |
| `doc/tasks/progress.md` | **總體進度追蹤（每完成一個模組即更新）** |

---

## 四、執行順序與依賴關係

```
階段 0（並行）：
  ├─ [子 Agent A] config 模組（Haiku）
  └─ [子 Agent B] tests 基礎設施（Haiku）

階段 1（待階段 0 完成）：
  └─ [子 Agent C] collector 模組（Sonnet）
      → 需要 config 完成 + .env 中有真實 API Key
      → 實際執行 `uv run python collector.py` 收集真實資料

階段 2（待 collector 完成）：
  └─ [子 Agent D] feature_engineering 模組（Sonnet）

階段 3（待 feature_engineering 完成）：
  └─ [子 Agent E] auto_labeler 模組（Haiku）

階段 4（待 auto_labeler 完成）：
  └─ [子 Agent F] merge_labels 模組（Haiku）
      → 注意：此模組需要人工填寫 pending_review.csv，
        但本次 Vibe Coding 無人工介入，
        因此子 Agent F 應：
        (1) 完整實作 merge_labels.py 及其測試
        (2) 產生一份測試用的 pending_review.csv（模擬已填寫）
        (3) 執行 merge_labels.py 驗證功能正常

階段 5（待 merge_labels 完成，並行）：
  ├─ [子 Agent G] train_stage1 模組（Sonnet）
  └─ [子 Agent H] train_stage2 模組（Sonnet）
      → train_stage2 需完整執行 BERT 訓練（hfl/chinese-macbert-base）
      → 若無 GPU，自動使用 CPU

階段 6（待 train_stage1 與 train_stage2 均完成）：
  └─ [子 Agent I] evaluate 模組（Sonnet）

階段 7（待 evaluate 完成）：
  └─ [子 Agent J] fusion 模組（Sonnet）
```

---

## 五、子 Agent 分配規則

### 模型選擇

| 模型 | 適用模組 |
|------|---------|
| `claude-haiku-4-5-20251001` | config、tests 基礎設施、auto_labeler、merge_labels |
| `claude-sonnet-4-6` | collector、feature_engineering、train_stage1、train_stage2、evaluate、fusion |

### 每個子 Agent 的必要指令

每個子 Agent 的 prompt 必須包含：

1. **讀取設計文件**：先閱讀 `doc/detailed-design.md` 與對應的 `doc/tasks/<module>.md`
2. **實作程式碼**：依設計文件中的公開 API 簽名實作
3. **撰寫測試**：依設計文件「測試要點」章節撰寫完整 pytest 測試
4. **執行測試**：`uv run pytest tests/test_<module>.py -v`，**所有測試必須通過**
5. **執行程式**：執行 `uv run python <module>.py`，確認輸出符合驗收條件
6. **回報結果**：列出已建立的檔案與測試通過數

---

## 六、每個子 Agent 的具體任務說明

### 子 Agent A — config（Haiku）

**任務：** 建立專案骨架與全域設定

閱讀 `doc/tasks/config.md`，完成以下工作：
- 建立 `.env.example`
- 建立 `.gitignore`（排除 `.env`、`data/`、`results/`、`__pycache__/`、`*.pkl`、`*.pyc`）
- 建立完整目錄骨架（含 `.gitkeep`）：
  - `data/collected/`
  - `data/models/stage1/`
  - `data/models/stage2/`
  - `data/models/stage3/`
  - `results/`
  - `tests/`
- 實作 `config.py`（完整內容見 `doc/detailed-design.md` §3）
- 建立 `requirements.txt`

**驗收：**
- `uv run python -c "import config"` 在 `.env` 存在時成功
- 移除 `.env` 後拋出 `KeyError: 'YOUTUBE_API_KEY'`

---

### 子 Agent B — tests 基礎設施（Haiku）

**任務：** 建立測試夾具與整合測試骨架

閱讀 `doc/tasks/tests.md`，完成以下工作：
- 建立 `tests/__init__.py`（空檔）
- 建立 `tests/fixtures/sample_raw.csv`（**≥ 30 筆假留言**）：
  - 欄位：`record_type`、`comment_id`、`video_id`、`author_id`、`author_name`、`content`、`like_count`、`reply_count`、`published_at`（ISO8601 UTC）、`parent_id`
  - 必須涵蓋邊界情境：單一留言用戶、凌晨發文用戶、含廣告關鍵字留言、含 URL 留言、同小時 3+ 則留言用戶、reply 型別留言、零按讚留言
- 建立 `tests/fixtures/sample_labeled.csv`（**≥ 50 位用戶**）：
  - 欄位：`author_id` + 13 個特徵（見 `config.py` 的 `FEATURE_COLS`）+ `is_bot`
  - 水軍（is_bot=1）與正常用戶（is_bot=0）各 ≥ 5 筆，其餘可隨機生成
- 建立 `tests/test_pipeline.py` 骨架（整合測試，可先留 pass，後續子 Agent 填充）

---

### 子 Agent C — collector（Sonnet）

**任務：** 實作 YouTube Data API v3 資料收集

閱讀 `doc/detailed-design.md` §4 與 `doc/tasks/collector.md`，完成以下工作：

1. 實作 `collector.py` 中所有公開函式（見設計文件 §4 的「公開 API」）：
   - `load_collected_ids`、`save_collected_ids`
   - `QuotaExceededError`、`check_quota`
   - `fetch_channel_uploads_playlist`
   - `fetch_recent_videos`
   - `fetch_video_stats`
   - `fetch_comments`（含回覆展開）
   - `run_collection`

2. 撰寫 `tests/test_collector.py`，所有 API 呼叫以 `unittest.mock.patch` mock

3. 執行 `uv run pytest tests/test_collector.py -v`，確保全部通過

4. 執行 `uv run python collector.py` 使用真實 API Key 收集資料
   - 確認 `data/collected/` 產生 `youtube_raw_YYYYMMDD.csv`
   - 確認 CSV 包含 `record_type="video"` 與 `record_type="comment"` 的列

**注意：**
- 配額限制 `QUOTA_DAILY_LIMIT = 9000`，安全邊際充足
- 單一影片留言失敗時記錄 warning，不終止整體收集
- 已收集的 video_id / comment_id 透過 `collected_ids.json` 跳過

---

### 子 Agent D — feature_engineering（Sonnet）

**任務：** 計算 13 個用戶特徵

閱讀 `doc/detailed-design.md` §5 與 `doc/tasks/feature_engineering.md`，完成以下工作：

1. 實作 `feature_engineering.py` 中所有公開函式：
   - `load_raw_comments`（讀取所有 youtube_raw_*.csv，去重，篩選時間窗口）
   - `compute_behavioral_features`（total_comments, unique_videos, concentration, avg_likes, zero_like_ratio, max_burst）
   - `compute_content_features`（avg_length, unique_content_ratio, self_similarity, has_ad_keywords, url_ratio）
     - `self_similarity` 使用 `TfidfVectorizer(analyzer='char', ngram_range=(1,2))`
   - `compute_temporal_features`（night_ratio, interval_std）
   - `run_feature_engineering`（主執行函式，合併三組特徵）

2. 撰寫 `tests/test_feature_engineering.py`，使用 `tests/fixtures/sample_raw.csv` 驗證

3. 執行 `uv run pytest tests/test_feature_engineering.py -v`

4. 執行 `uv run python feature_engineering.py`
   - 確認產生 `data/user_features.csv`，包含 `author_id` + 恰好 13 個特徵欄位

**關鍵細節：**
- `self_similarity`：留言數 < 2 時回傳 0.0；留言數 ≥ 2 時計算上三角 cosine similarity 均值
- `interval_std`：留言數 < 2 時回傳 0.0
- `concentration`：`unique_videos == 0` 時填 0
- 所有時間計算以 UTC 為準

---

### 子 Agent E — auto_labeler（Haiku）

**任務：** 啟發式規則自動標注

閱讀 `doc/detailed-design.md` §6 與 `doc/tasks/auto_labeler.md`，完成以下工作：

1. 實作 `auto_labeler.py`：
   - `classify_user(row: pd.Series) -> str`（純函式，回傳 'bot'、'normal' 或 'pending'）
     - **水軍條件（4 個均滿足）**：concentration > 5.0 **且** self_similarity > 0.7 **且** zero_like_ratio > 0.8 **且** interval_std < 60.0
     - **正常條件（3 個均滿足）**：unique_videos >= 3 **且** self_similarity < 0.3 **且** zero_like_ratio < 0.5
     - 同時滿足兩組條件（理論不可能）：優先回傳 'bot'
   - `run_labeling(features_path, labeled_output, pending_output)`
     - `pending_review.csv` 的 `is_bot` 欄位填空字串（非 NaN）

2. 撰寫 `tests/test_auto_labeler.py`：
   - 測試 `concentration=5.0`（等號，不算水軍）vs `concentration=5.01`（超過，算水軍）
   - 測試 pending 列的 `is_bot` 為空字串非 NaN

3. 執行 `uv run pytest tests/test_auto_labeler.py -v`

4. 執行 `uv run python auto_labeler.py`
   - 確認同時產生 `data/labeled_data.csv` 與 `data/pending_review.csv`

---

### 子 Agent F — merge_labels（Haiku）

**任務：** 人工複審標籤合併

閱讀 `doc/detailed-design.md` §7 與 `doc/tasks/merge_labels.md`，完成以下工作：

1. 實作 `merge_labels.py`：
   - `merge_reviewed_labels(labeled_path, pending_path, output_path) -> dict`
   - 只合併 `is_bot` 為 "0" 或 "1" 的列（轉為 int）
   - 空字串/NaN → 跳過（計入 skipped_empty）
   - 非法值（"2"、"yes" 等）→ 拋 `ValueError` 並終止
   - 重複 `author_id` → 保留既有版本，stderr 輸出警告

2. 為了在無人工介入的情況下驗證功能，**建立測試用的 `pending_review_test.csv`**：
   - 從 `data/pending_review.csv` 複製一份
   - 手動填入 is_bot 值（一半填 0，一半填 1，留幾列為空字串）
   - 執行 `merge_labels.py` 以此測試檔驗證合併功能

3. 撰寫 `tests/test_merge_labels.py`

4. 執行 `uv run pytest tests/test_merge_labels.py -v`

5. 執行 `uv run python merge_labels.py` 確認 `labeled_data.csv` 筆數增加

---

### 子 Agent G — train_stage1（Sonnet）

**任務：** Logistic Regression + XGBoost 訓練

閱讀 `doc/detailed-design.md` §8 與 `doc/tasks/train_stage1.md`，完成以下工作：

1. 實作 `train_stage1.py`：
   - `load_labeled_data(labeled_path, feature_cols) -> (X, y)`
     - 樣本數 < 50 時拋 `ValueError`
   - `train_logistic_regression(X_train, y_train) -> (Pipeline, best_params)`
     - `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"))])`
     - `GridSearchCV`（StratifiedKFold 5-fold）搜尋 `clf__C in [0.01, 0.1, 1.0, 10.0]`，scoring="f1"
   - `train_xgboost(X_train, y_train) -> (XGBClassifier, best_params)`
     - 自動計算 `scale_pos_weight`
     - GridSearchCV 搜尋 n_estimators, max_depth, learning_rate
   - `run_training(labeled_path, model_dir, results_dir)`
     - `StratifiedShuffleSplit(test_size=0.2, random_state=42)` 切分，記錄 test 索引
     - 儲存 `lr_model.pkl`、`xgb_model.pkl`、`scaler.pkl`（joblib）
     - 輸出 `stage1_report.json`（格式見設計文件 §8）

2. 撰寫 `tests/test_train_stage1.py`，使用 `tests/fixtures/sample_labeled.csv`

3. 執行 `uv run pytest tests/test_train_stage1.py -v`

4. 執行 `uv run python train_stage1.py`
   - 確認產生 `lr_model.pkl`、`xgb_model.pkl`、`scaler.pkl`、`stage1_report.json`

---

### 子 Agent H — train_stage2（Sonnet）

**任務：** 中文 BERT 微調

閱讀 `doc/detailed-design.md` §9 與 `doc/tasks/train_stage2.md`，完成以下工作：

1. 實作 `train_stage2.py`：
   - `build_comment_label_df(labeled_path, raw_dir)` — 合併用戶標籤與留言文字
   - `CommentDataset(torch.utils.data.Dataset)` — `__init__`、`__len__`、`__getitem__`
     - `__getitem__` 回傳 `{"input_ids": Tensor[max_len], "attention_mask": Tensor[max_len], "labels": FloatTensor scalar}`
   - `train_bert(train_df, val_df, model_name, output_dir, epochs)` — 完整訓練流程
     - `BCEWithLogitsLoss(pos_weight=...)`
     - AdamW + 線性 warmup scheduler
     - 每 epoch 計算 val F1，連續 2 epoch 無改善 early stop
     - 儲存最佳 checkpoint
   - `run_inference(model_dir, texts, batch_size=32)` — 推論，輸出 sigmoid 機率
   - `run_training(labeled_path, raw_dir, model_dir, results_dir)` — 主流程
     - Stratified 80/10/10 分割
     - 訓練後對全資料集推論，輸出 `data/bert_comment_scores.csv`
     - 輸出 `results/stage2_report.json`

2. 撰寫 `tests/test_train_stage2.py`：
   - **使用 `distilbert-base-multilingual-cased` 替代正式模型**（快速，適合 CI）
   - 測試 `build_comment_label_df` 排除未標注用戶
   - 測試 `CommentDataset.__getitem__` shape
   - 測試 CPU fallback
   - 測試 `run_inference` 輸出值在 [0, 1]

3. 執行 `uv run pytest tests/test_train_stage2.py -v`

4. 執行 `uv run python train_stage2.py`（使用真實 `hfl/chinese-macbert-base`）
   - 確認產生 `data/models/stage2/bert/`、`data/bert_comment_scores.csv`、`results/stage2_report.json`
   - GPU 不可用時自動 CPU fallback

---

### 子 Agent I — evaluate（Sonnet）

**任務：** 多模型統一評估報告

閱讀 `doc/detailed-design.md` §10 與 `doc/tasks/evaluate.md`，完成以下工作：

1. 實作 `evaluate.py`：
   - `load_test_split(labeled_path, stage1_report_path, feature_cols)` — 用 stage1 的 test_indices 還原測試集
   - `evaluate_sklearn_model(model, X_test, y_test, model_name)` — 計算 precision, recall, f1, auc_roc, confusion_matrix
   - `evaluate_bert_model(bert_model_dir, test_author_ids, raw_dir, y_test)` — 聚合留言分數至用戶層級
     - BERT 目錄不存在時回傳 None，不拋例外
   - `run_evaluation(labeled_path, model_dir, raw_dir, results_dir)` — 主流程
     - 輸出 `results/evaluation_report.json`
     - 輸出 `results/comparison_report.csv`（數值保留 4 位小數）

2. 撰寫 `tests/test_evaluate.py`

3. 執行 `uv run pytest tests/test_evaluate.py -v`

4. 執行 `uv run python evaluate.py`
   - 確認三個模型（LR、XGBoost、BERT）在**完全相同的 test set** 上評估

---

### 子 Agent J — fusion（Sonnet）

**任務：** 多模態融合模型

閱讀 `doc/detailed-design.md` §11 與 `doc/tasks/fusion.md`，完成以下工作：

1. 實作 `fusion.py`：
   - `build_fusion_features(labeled_path, model_dir, bert_scores_path, feature_cols)` — 構建 5 維融合特徵：
     - `lr_prob`（LR predict_proba）
     - `xgb_prob`（XGBoost predict_proba）
     - `bert_user_prob`（spam_prob 均值）
     - `bert_max_prob`（spam_prob 最大值）
     - `bert_comment_count`（留言數）
   - `train_fusion(X_fusion, y, test_indices)` — 5-fold CV 訓練 `LogisticRegression(class_weight="balanced", max_iter=500)`
     - Hold-out test set 使用 stage1 的相同 test_indices（防止 data leakage）
   - `run_fusion(labeled_path, model_dir, bert_scores_path, results_dir)` — 主流程
     - 儲存 `data/models/stage3/fusion_model.pkl`
     - 輸出 `results/stage3_report.json`（含 precision, recall, f1, auc_roc）

2. 撰寫 `tests/test_fusion.py`：
   - 測試 `build_fusion_features` 輸出恰好 5 個特徵欄位
   - 測試 `train_fusion` 不使用 test_indices 的樣本訓練

3. 執行 `uv run pytest tests/test_fusion.py -v`

4. 執行 `uv run python fusion.py`
   - 確認產生 `data/models/stage3/fusion_model.pkl` 與 `results/stage3_report.json`

---

## 七、主 Agent 的操作流程

### 開始時

1. 閱讀 `doc/tasks/progress.md` 確認當前進度
2. 閱讀 `doc/detailed-design.md` 與 `doc/proposal.md` 建立完整系統理解
3. 確認 `.env` 存在且包含 `YOUTUBE_API_KEY`

### 每個階段

1. **並行啟動** 同一階段的子 Agent（無依賴關係時）
2. **等待** 所有子 Agent 回報完成
3. **驗證** 子 Agent 的聲明：
   - 確認聲明建立的檔案確實存在
   - 確認測試確實通過（執行 `uv run pytest tests/test_<module>.py -v`）
4. **更新進度**：在 `doc/tasks/progress.md` 勾選已完成的模組
5. **進入下一階段**

### 完成標準

所有模組完成後，執行完整測試套件：

```powershell
uv run pytest tests/ -v --tb=short
```

所有測試必須通過。然後確認：
- `data/labeled_data.csv` 存在且有資料
- `results/evaluation_report.json` 存在且含 LR、XGBoost 的指標
- `results/stage2_report.json` 存在
- `results/stage3_report.json` 存在

向使用者報告：
1. 各模型的 Precision / Recall / F1 / AUC-ROC
2. 是否達到目標值（Precision ≥ 0.80、Recall ≥ 0.75、F1 ≥ 0.78、AUC-ROC ≥ 0.85）
3. 各測試通過數

---

## 八、關鍵約束（所有子 Agent 必須遵守）

1. **不硬編碼常數**：所有超參數、路徑、閾值均從 `config.py` 讀取，禁止在模組中重複定義
2. **模組間無 import 相依**：各模組僅透過檔案系統（CSV/JSON/PKL）傳遞資料
3. **測試不依賴真實 API/GPU**：API 呼叫全部 mock，BERT 測試使用小型替代模型
4. **邊界值防禦**：`concentration` 水軍條件為 **嚴格大於** 5.0（不含等號）
5. **時區一致性**：所有時間計算以 UTC 為準，`night_ratio` 不做時區轉換
6. **serialization 格式**：模型使用 `joblib.dump/load`（scikit-learn/XGBoost），BERT 使用 `save_pretrained`
7. **pending 欄位格式**：`pending_review.csv` 的 `is_bot` 欄位為**空字串**（非 NaN），確保 Excel 可直接填寫
8. **配額保護**：collector 的 `check_quota` 在 `used >= limit` 時拋例外，已收集資料寫入後再結束

---

## 九、進度更新格式

每完成一個模組，在 `doc/tasks/progress.md` 更新對應行：

```markdown
- [x] **config** — 全域設定、環境變數、目錄骨架、requirements.txt
```

---

*本 prompt 由主 Agent 起始，子 Agent 不需閱讀此文件，子 Agent 的 prompt 由主 Agent 動態生成。*
