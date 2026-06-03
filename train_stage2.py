import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import glob
import config


def build_comment_label_df(labeled_path: str, raw_dir: str) -> pd.DataFrame:
    """合併用戶標籤與留言文字，排除未標注用戶"""
    labeled_df = pd.read_csv(labeled_path)[["author_id", "is_bot"]]
    label_map = dict(zip(labeled_df["author_id"], labeled_df["is_bot"].astype(int)))

    raw_files = glob.glob(os.path.join(raw_dir, "collected", "youtube_raw_*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"No youtube_raw_*.csv found in {os.path.join(raw_dir, 'collected')}"
        )

    dfs = [pd.read_csv(f) for f in raw_files]
    raw_df = pd.concat(dfs, ignore_index=True)
    comments = raw_df[raw_df["record_type"].isin(["comment", "reply"])].copy()
    comments = comments[["comment_id", "author_id", "content"]].dropna(subset=["content"])
    comments["is_spam"] = comments["author_id"].map(label_map)
    comments = comments.dropna(subset=["is_spam"])
    comments["is_spam"] = comments["is_spam"].astype(int)
    return comments[["comment_id", "author_id", "content", "is_spam"]].reset_index(drop=True)


class CommentDataset(Dataset):
    """PyTorch Dataset for comments"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """回傳 {'input_ids': LongTensor[max_len], 'attention_mask': LongTensor[max_len], 'labels': FloatTensor scalar}"""
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def train_bert(train_df, val_df, model_name, output_dir, epochs=3):
    """完整訓練流程，early stop，儲存最佳 checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_bert] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    # BCEWithLogitsLoss with pos_weight
    n_neg = int((train_df["is_spam"] == 0).sum())
    n_pos = int((train_df["is_spam"] == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = CommentDataset(
        train_df["content"].tolist(),
        train_df["is_spam"].tolist(),
        tokenizer,
        max_len=config.BERT_MAX_LEN,
    )
    val_dataset = CommentDataset(
        val_df["content"].tolist(),
        val_df["is_spam"].tolist(),
        tokenizer,
        max_len=config.BERT_MAX_LEN,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BERT_BATCH_SIZE, shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=config.BERT_LR)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(config.BERT_WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    no_improve = 0
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[train_bert] Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                if probs.ndim == 0:
                    probs = np.array([float(probs)])
                preds = (probs > 0.5).astype(int)
                val_preds.extend(preds.tolist())
                val_labels_list.extend(batch["labels"].numpy().tolist())

        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
        val_precision = precision_score(val_labels_list, val_preds, zero_division=0)
        val_recall = recall_score(val_labels_list, val_preds, zero_division=0)
        print(
            f"[train_bert] Epoch {epoch + 1} Val F1: {val_f1:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            best_metrics = {
                "best_epoch": epoch + 1,
                "val_f1": float(val_f1),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
            }
            print(f"[train_bert] New best checkpoint saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= 2:
                print(f"[train_bert] Early stop at epoch {epoch + 1}")
                break

    # If no improvement at all, save final model
    if not best_metrics:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        best_metrics = {
            "best_epoch": epochs,
            "val_f1": float(val_f1) if val_labels_list else 0.0,
            "val_precision": float(val_precision) if val_labels_list else 0.0,
            "val_recall": float(val_recall) if val_labels_list else 0.0,
        }

    return best_metrics


def run_inference(model_dir: str, texts: list, batch_size: int = 32) -> list:
    """載入模型，推論，回傳 sigmoid 機率值列表"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    probabilities = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=config.BERT_MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in encoding.items()})
            logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        probabilities.extend(probs.tolist())

    return probabilities


def run_training(
    labeled_path: str = "data/labeled_data.csv",
    raw_dir: str = "data",
    model_dir: str = "data/models/stage2",
    results_dir: str = "results",
) -> None:
    print("[run_training] Building comment-label DataFrame...")
    comment_df = build_comment_label_df(labeled_path, raw_dir)
    print(
        f"[run_training] Total labeled comments: {len(comment_df)}, "
        f"spam={comment_df['is_spam'].sum()}, "
        f"normal={(comment_df['is_spam'] == 0).sum()}"
    )

    if len(comment_df) < 50:
        raise ValueError(
            f"Insufficient training data: only {len(comment_df)} labeled comments found."
        )

    # Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        comment_df,
        test_size=0.2,
        random_state=42,
        stratify=comment_df["is_spam"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["is_spam"],
    )
    print(
        f"[run_training] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    bert_output_dir = os.path.join(model_dir, "bert")
    os.makedirs(bert_output_dir, exist_ok=True)

    print(f"[run_training] Training BERT model: {config.BERT_MODEL_NAME}")
    best_metrics = train_bert(
        train_df=train_df,
        val_df=val_df,
        model_name=config.BERT_MODEL_NAME,
        output_dir=bert_output_dir,
        epochs=config.BERT_EPOCHS,
    )

    # Run inference on full comment dataset
    print("[run_training] Running inference on all comments...")
    all_texts = comment_df["content"].tolist()
    spam_probs = run_inference(bert_output_dir, all_texts, batch_size=config.BERT_BATCH_SIZE)

    scores_df = pd.DataFrame(
        {
            "comment_id": comment_df["comment_id"].values,
            "author_id": comment_df["author_id"].values,
            "spam_prob": spam_probs,
        }
    )
    scores_path = os.path.join("data", "bert_comment_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(
        f"[run_training] Saved {len(scores_df)} comment scores to {scores_path} "
        f"(spam_prob range: [{scores_df['spam_prob'].min():.4f}, {scores_df['spam_prob'].max():.4f}])"
    )

    # Save stage2 report
    report = {
        "model_name": config.BERT_MODEL_NAME,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        **best_metrics,
    }
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "stage2_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[run_training] Saved report to {report_path}")
    print(f"[run_training] Best metrics: {best_metrics}")


if __name__ == "__main__":
    run_training()
    print("Stage2 training complete.")
