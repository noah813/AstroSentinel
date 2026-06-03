import pytest
import os
import sys
import pandas as pd
import numpy as np
import torch
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TEST_MODEL = "distilbert-base-multilingual-cased"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_RAW = os.path.join(FIXTURES_DIR, "sample_raw.csv")
SAMPLE_LABELED = os.path.join(FIXTURES_DIR, "sample_labeled.csv")


def test_build_comment_label_df_excludes_unlabeled(tmp_path):
    """排除未標注用戶的留言"""
    from train_stage2 import build_comment_label_df
    import shutil

    # 建立有 author_A（無標籤）和 sample_labeled 中用戶（有標籤）的資料
    raw_csv = tmp_path / "collected" / "youtube_raw_test.csv"
    raw_csv.parent.mkdir(parents=True)
    shutil.copy(SAMPLE_RAW, raw_csv)

    # 只標注 sample_labeled.csv 中的用戶
    result = build_comment_label_df(SAMPLE_LABELED, str(tmp_path))

    # 確認只有 sample_labeled.csv 中的 author_id 出現
    labeled_df = pd.read_csv(SAMPLE_LABELED)
    labeled_ids = set(labeled_df["author_id"])
    result_ids = set(result["author_id"])
    assert result_ids.issubset(labeled_ids), "結果不應包含未標注用戶"
    assert "is_spam" in result.columns


def test_build_comment_label_df_includes_labeled(tmp_path):
    """有標注的用戶留言應被包含"""
    from train_stage2 import build_comment_label_df
    import shutil

    # 建立一個含有 sample_labeled 中用戶的 raw CSV
    raw_dir = tmp_path / "collected"
    raw_dir.mkdir(parents=True)

    labeled_df = pd.read_csv(SAMPLE_LABELED)
    labeled_author = labeled_df["author_id"].iloc[0]

    rows = [
        {
            "record_type": "comment",
            "comment_id": "c_test_001",
            "video_id": "v_test_001",
            "author_id": labeled_author,
            "author_name": "TestUser",
            "content": "這是測試留言",
            "like_count": 0,
            "reply_count": 0,
            "published_at": "2026-05-28T10:00:00Z",
            "parent_id": "",
        },
        {
            "record_type": "comment",
            "comment_id": "c_test_002",
            "video_id": "v_test_001",
            "author_id": "unlabeled_author_xyz",
            "author_name": "Unknown",
            "content": "另一則測試留言",
            "like_count": 0,
            "reply_count": 0,
            "published_at": "2026-05-28T11:00:00Z",
            "parent_id": "",
        },
    ]
    test_raw_df = pd.DataFrame(rows)
    test_raw_df.to_csv(raw_dir / "youtube_raw_test.csv", index=False)

    result = build_comment_label_df(SAMPLE_LABELED, str(tmp_path))

    assert len(result) >= 1, "應至少包含一筆已標注用戶的留言"
    assert "unlabeled_author_xyz" not in result["author_id"].values
    assert labeled_author in result["author_id"].values
    assert "is_spam" in result.columns


def test_comment_dataset_shape():
    """CommentDataset.__getitem__ 的 shape 應為 [128]"""
    from train_stage2 import CommentDataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    texts = ["Hello world", "This is a test comment"]
    labels = [0, 1]
    dataset = CommentDataset(texts, labels, tokenizer, max_len=128)

    item = dataset[0]
    assert item["input_ids"].shape == torch.Size([128])
    assert item["attention_mask"].shape == torch.Size([128])
    assert item["labels"].dtype == torch.float32


def test_cpu_fallback():
    """GPU 不可用時 run_inference 不拋例外"""
    from train_stage2 import run_inference
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # 建立一個臨時小模型目錄
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEST_MODEL, num_labels=1)
        tokenizer.save_pretrained(tmp_dir)
        model.save_pretrained(tmp_dir)

        texts = ["測試留言", "另一則留言"]
        probs = run_inference(tmp_dir, texts, batch_size=2)

        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs), f"機率應在 [0,1]，實際：{probs}"


def test_run_inference_range():
    """run_inference 輸出值全部在 [0, 1]"""
    from train_stage2 import run_inference
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(TEST_MODEL, num_labels=1)
        tokenizer.save_pretrained(tmp_dir)
        model.save_pretrained(tmp_dir)

        texts = [f"留言{i}" for i in range(10)]
        probs = run_inference(tmp_dir, texts, batch_size=4)

        assert len(probs) == 10
        for p in probs:
            assert 0.0 <= p <= 1.0, f"機率 {p} 超出 [0,1] 範圍"
