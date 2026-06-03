# train_stage2.py — BERT 微調

## 前置條件

- `config.py` 完成（C4）
- `data/labeled_data.csv` 已存在且 ≥ 50 位用戶
- `data/collected/youtube_raw_*.csv` 已存在（含留言文字）
- `data/models/stage2/` 目錄已存在
- 已安裝 `transformers`、`torch`

## 子任務

- [ ] **TS2-1** 實作 `build_comment_label_df(labeled_path, raw_dir)`：
  - 讀取 `labeled_data.csv` 取得 `{author_id: is_bot}` 映射
  - 讀取所有 `youtube_raw_*.csv`，篩選 `record_type in ("comment","reply")`
  - 以 `author_id` 關聯：`is_bot=1` → `is_spam=1`，`is_bot=0` → `is_spam=0`
  - 排除 `author_id` 不在 `labeled_data.csv` 的留言（unlabeled）
  - 回傳含 `comment_id`、`author_id`、`content`、`is_spam` 的 DataFrame

- [ ] **TS2-2** 實作 `CommentDataset(torch.utils.data.Dataset)`：
  - `__init__(texts, labels, tokenizer, max_len=128)`
  - `__len__` 回傳樣本數
  - `__getitem__(idx)` 回傳 `{"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}`，皆為 `[max_len]` 形狀的 LongTensor（labels 為 FloatTensor scalar）

- [ ] **TS2-3** 實作 `train_bert(train_df, val_df, model_name, output_dir, epochs)`：
  - 計算 `pos_weight = (is_spam==0).sum() / (is_spam==1).sum()`，傳入 `BCEWithLogitsLoss`
  - 載入 `AutoTokenizer` 與 `AutoModelForSequenceClassification(num_labels=1)`
  - AdamW optimizer（`lr=config.BERT_LR`）
  - 線性 warmup scheduler（`warmup_steps = BERT_WARMUP_RATIO × total_steps`）
  - 每個 epoch 結束後計算 val F1；連續 2 個 epoch 無改善時 early stop
  - 以最佳 val F1 的 checkpoint 呼叫 `model.save_pretrained(output_dir)`
  - 回傳 `{"best_epoch": N, "val_f1": ..., "val_precision": ..., "val_recall": ...}`

- [ ] **TS2-4** 實作 `run_inference(model_dir, texts, batch_size)`：
  - 載入模型後切換到 eval 模式
  - GPU 不可用時自動 fallback 到 CPU
  - 輸出過 sigmoid 後的機率值（確保 `[0, 1]` 範圍）
  - 回傳 `list[float]`，長度等於輸入 texts 長度

- [ ] **TS2-5** 實作 `run_training(labeled_path, raw_dir, model_dir, results_dir)`：
  - 呼叫 TS2-1 建立留言標籤 DataFrame
  - Stratified 80/10/10 分割（train/val/test）
  - 呼叫 `train_bert` 訓練並儲存至 `{model_dir}/bert/`
  - 對全資料集呼叫 `run_inference`，輸出 `data/bert_comment_scores.csv`（`comment_id`、`author_id`、`spam_prob`）
  - 輸出 `{results_dir}/stage2_report.json`

- [ ] **TS2-6** 寫測試 `tests/test_train_stage2.py`：
  - `build_comment_label_df` 正確排除未標注用戶的留言
  - `CommentDataset.__getitem__` 的 `input_ids` shape 為 `[128]`
  - GPU 不可用時 `run_inference` 不拋例外（CPU fallback）
  - `run_inference` 輸出值全部在 `[0, 1]` 範圍內
  - 測試使用 `distilbert-base-multilingual-cased` 替換正式模型（快速）

## 驗收條件

- `python train_stage2.py` 執行後產生 `data/models/stage2/bert/`、`data/bert_comment_scores.csv`、`results/stage2_report.json`
- `bert_comment_scores.csv` 的 `spam_prob` 欄位值全在 `[0, 1]`
- `tests/test_train_stage2.py` 全部通過
