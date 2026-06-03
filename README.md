# AstroSentinel

YouTube 政治頻道**網路水軍評論偵測系統** — 學術研究專案

透過多維度特徵分析，結合傳統機器學習與中文 BERT，對 YouTube 留言與用戶帳號進行水軍風險評估。

---

## 研究背景

隨著社群媒體普及，協同操控輿論的水軍（Astroturfing）行為嚴重影響公共討論品質。本專案針對 YouTube 政治／時事頻道留言，開發可重現的自動化偵測流程，涵蓋資料收集、特徵工程、自動標注、模型訓練與評估。

---

## 偵測對象

| 類型 | 描述 |
|------|------|
| Bot 機器人 | 自動化程式產生的留言（發文間隔極短、內容高度重複） |
| 付費刷評論 | 業配、廣告、行銷留言（含廣告關鍵字、夾帶連結） |
| 協同操控 | 多帳號推同一立場（時間群聚、內容相似） |
| 垃圾內容 | 無意義或大量複製的留言（高自我相似度） |

---

## 系統架構

```
YouTube Data API v3
        │
        ▼
  collector.py          ← 收集指定頻道過去 7 天影片留言
        │
        ▼
feature_engineering.py  ← 計算用戶行為 / 內容 / 時間特徵（13 維）
        │
        ▼
  auto_labeler.py       ← 啟發式規則自動標記
        │
        ▼
  merge_labels.py       ← 人工複審合併（選配）
        │
     ┌──┴──┐
     ▼     ▼
train_stage1.py    train_stage2.py
(LR / XGBoost)     (中文 BERT 微調)
     │     │
     └──┬──┘
        ▼
  evaluate.py           ← 輸出 precision / recall / F1 / AUC
        │
        ▼
  fusion.py             ← 多模態融合（選配）
```

---

## 目錄結構

```
AstroSentinel/
├── config.py                   # 全域設定（API 金鑰、路徑、閾值）
├── collector.py                # YouTube Data API 資料收集
├── feature_engineering.py      # 用戶特徵提取
├── auto_labeler.py             # 啟發式自動標注
├── merge_labels.py             # 人工複審合併
├── train_stage1.py             # 傳統 ML 訓練（LR / XGBoost）
├── train_stage2.py             # BERT 微調
├── evaluate.py                 # 模型評估
├── fusion.py                   # 多模態融合（選配）
├── predict_pending.py          # 對待複審樣本預測
├── generate_report.py          # 產生實驗報告
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── collected/              # 原始留言 CSV（youtube_raw_YYYYMMDD.csv）
│   ├── user_features.csv
│   ├── labeled_data.csv
│   ├── pending_review.csv
│   └── models/
│       ├── stage1/             # lr_model.pkl, xgb_model.pkl, scaler.pkl
│       ├── stage2/bert/        # Hugging Face 格式
│       └── stage3/             # fusion_model.pkl（選配）
├── results/
│   ├── stage1_report.json
│   ├── stage2_report.json
│   ├── stage3_report.json
│   ├── evaluation_report.json
│   ├── comparison_report.csv
│   └── report.md
├── tests/
│   ├── fixtures/               # 測試用 CSV fixtures
│   ├── test_collector.py
│   ├── test_feature_engineering.py
│   ├── test_auto_labeler.py
│   ├── test_merge_labels.py
│   ├── test_train_stage1.py
│   ├── test_train_stage2.py
│   ├── test_evaluate.py
│   ├── test_fusion.py
│   └── test_pipeline.py
└── doc/
    ├── proposal.md
    └── detailed-design.md
```

---

## 快速開始

### 環境需求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（推薦）或 pip

### 安裝

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 設定 API 金鑰

在專案根目錄建立 `.env` 檔案：

```
YOUTUBE_API_KEY=your_api_key_here
```

> **注意：** `.env` 已列入 `.gitignore`，不會上傳至 Git。

---

## 執行流程

依序執行以下步驟：

### 第一步：收集資料

```bash
python collector.py
```

輸出：`data/collected/youtube_raw_YYYYMMDD.csv`

### 第二步：特徵提取

```bash
python feature_engineering.py
```

輸出：`data/user_features.csv`（每位用戶 13 個特徵）

### 第三步：自動標注

```bash
python auto_labeler.py
```

輸出：
- `data/labeled_data.csv`（確定標籤）
- `data/pending_review.csv`（待人工複審）

### 第四步（選配）：人工複審合併

在 `data/pending_review.csv` 的 `is_bot` 欄位填入 `0` 或 `1` 後執行：

```bash
python merge_labels.py
```

### 第五步：訓練模型

```bash
# 第一階段：Logistic Regression + XGBoost
python train_stage1.py

# 第二階段：中文 BERT 微調（建議 GPU 環境）
python train_stage2.py
# GPU 環境：
# CUDA_VISIBLE_DEVICES=0 python train_stage2.py
```

### 第六步：評估

```bash
python evaluate.py
```

輸出：`results/evaluation_report.json`、`results/comparison_report.csv`

### 第七步（選配）：多模態融合

```bash
python fusion.py
```

---

## 特徵工程

以用戶帳號（`author_id`）為單位，使用固定 7 天回溯窗口計算 13 個特徵：

**行為特徵：**

| 特徵 | 說明 |
|------|------|
| `total_comments` | 7 天內總留言數 |
| `unique_videos` | 留言涉及的不同影片數 |
| `concentration` | `total_comments / unique_videos` |
| `avg_likes` | 平均每則留言按讚數 |
| `zero_like_ratio` | 零按讚留言比例 |
| `max_burst` | 單小時最大留言量 |

**內容特徵：**

| 特徵 | 說明 |
|------|------|
| `avg_length` | 平均留言字數 |
| `unique_content_ratio` | 不重複留言比例 |
| `self_similarity` | TF-IDF cosine similarity 均值 |
| `has_ad_keywords` | 是否含廣告關鍵字 |
| `url_ratio` | 含 URL 留言比例 |

**時間特徵：**

| 特徵 | 說明 |
|------|------|
| `night_ratio` | 凌晨 0–6 點留言比例 |
| `interval_std` | 發文間隔標準差（秒） |

---

## 實驗結果

測試集大小：365 筆，目標頻道：中天新聞、TVBS、三立新聞、民視新聞。

| 模型 | Precision | Recall | F1 | AUC-ROC |
|------|-----------|--------|-----|---------|
| Logistic Regression | 0.842 | **1.000** | **0.914** | **0.990** |
| XGBoost | 0.839 | 0.975 | 0.902 | 0.990 |
| BERT（MacBERT，聚合至用戶層級） | **0.872** | 0.850 | 0.861 | 0.971 |

> 由於資料不平衡，主要以 **F1-Score** 作為模型選擇依據。

---

## ML 方法說明

### 第一階段：傳統機器學習（Baseline）

- **Logistic Regression**：含 StandardScaler Pipeline，class_weight="balanced"，GridSearchCV 調整 C
- **XGBoost**：自動計算 `scale_pos_weight`，GridSearchCV 調整 `n_estimators`、`max_depth`、`learning_rate`
- 評估採 5-fold StratifiedKFold 交叉驗證

### 第二階段：中文 BERT 語意模型

- 模型：`hfl/chinese-macbert-base`（MacBERT）
- 輸入：單則留言文字（max_len=128）
- 輸出：留言層級水軍機率（`spam_prob`）
- 用戶標籤推衍至留言層級，以 `BCEWithLogitsLoss` + pos_weight 處理不平衡
- Early stopping（連續 2 個 epoch val F1 無改善）

### 第三階段：多模態融合（選配）

將 LR 機率、XGBoost 機率、BERT 留言分數聚合（均值 + 最大值）構成 5 維融合特徵，以 Logistic Regression 元學習器（Stacking）整合。

---

## 測試

```bash
pytest tests/
```

- 所有測試不依賴真實 API 或 GPU
- API 呼叫全部使用 `unittest.mock.patch` mock
- BERT 測試使用輕量模型替換

---

## 技術棧

| 項目 | 工具 |
|------|------|
| 程式語言 | Python 3.10+ |
| 套件管理 | uv |
| 資料收集 | google-api-python-client |
| 資料處理 | pandas, numpy |
| 特徵計算 | scikit-learn（TF-IDF, cosine similarity） |
| 傳統 ML | scikit-learn, xgboost |
| 深度學習 | transformers, torch |
| 測試 | pytest |
| 版本控制 | Git |

---

## 注意事項

- YouTube Data API v3 免費層級每日配額 10,000 units，系統預設使用上限 9,000 units（保留 1,000 緩衝）
- 資料不平衡（水軍比例遠低於正常用戶），建議使用 SMOTE 或 `class_weight` 因應
- 本專案使用啟發式自動標注，標籤品質受閾值設定影響，論文引用時需說明其侷限性

---

## 文件

- [研究提案](doc/proposal.md)
- [詳細設計文檔](doc/detailed-design.md)
