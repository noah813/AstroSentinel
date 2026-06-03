# AstroSentinel — 總體進度

> 每完成一個模組的所有子任務後，勾選對應項目。  
> 子任務詳情見各模組的 `.md` 檔案。

## 模組完成狀態

- [x] **config** — 全域設定、環境變數、目錄骨架、requirements.txt（→ [config.md](config.md)）
- [x] **tests 基礎設施** — fixture 資料、`__init__.py`、整合測試骨架（→ [tests.md](tests.md)）
- [x] **collector** — YouTube API 收集、增量去重、配額保護（→ [collector.md](collector.md)）
- [x] **feature_engineering** — 13 個用戶特徵計算（→ [feature_engineering.md](feature_engineering.md)）
- [x] **auto_labeler** — 啟發式規則自動標注（→ [auto_labeler.md](auto_labeler.md)）
- [x] **merge_labels** — 人工複審標籤合併（→ [merge_labels.md](merge_labels.md)）
- [x] **train_stage1** — Logistic Regression + XGBoost 訓練（→ [train_stage1.md](train_stage1.md)）
- [x] **train_stage2** — 中文 BERT 微調（→ [train_stage2.md](train_stage2.md)）
- [x] **evaluate** — 多模型統一評估報告（→ [evaluate.md](evaluate.md)）
- [x] **fusion** *(選配)* — 多模態融合模型（→ [fusion.md](fusion.md)）

## 建議執行順序

```
config → tests 基礎設施 → collector → feature_engineering
       → auto_labeler → merge_labels
       → train_stage1 ─┐
       → train_stage2 ─┴→ evaluate → fusion（選配）
```

## 目標指標（evaluate.py 輸出）

| 指標 | 目標值 |
|------|--------|
| Precision | ≥ 0.80 |
| Recall | ≥ 0.75 |
| F1-Score | ≥ 0.78 |
| AUC-ROC | ≥ 0.85 |
