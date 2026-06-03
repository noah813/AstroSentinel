import pytest
import pandas as pd
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_labeled_csv(path, rows):
    """建立假的 labeled_data.csv"""
    cols = ["author_id", "total_comments", "unique_videos", "concentration",
            "avg_likes", "zero_like_ratio", "max_burst", "avg_length",
            "unique_content_ratio", "self_similarity", "has_ad_keywords",
            "url_ratio", "night_ratio", "interval_std", "is_bot"]
    data = []
    for r in rows:
        row = {c: 0 for c in cols}
        row.update(r)
        data.append(row)
    # 如果 rows 為空，創建空 DataFrame 但保留欄位
    if not data:
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def make_pending_csv(path, rows):
    """建立假的 pending_review.csv"""
    cols = ["author_id", "total_comments", "unique_videos", "concentration",
            "avg_likes", "zero_like_ratio", "max_burst", "avg_length",
            "unique_content_ratio", "self_similarity", "has_ad_keywords",
            "url_ratio", "night_ratio", "interval_std", "is_bot"]
    data = []
    for r in rows:
        row = {c: 0 for c in cols}
        row["is_bot"] = ""
        row.update(r)
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, quoting=1)


def test_merge_valid_labels(tmp_path):
    """正常合併：'0' 和 '1' 的列成功合併"""
    from merge_labels import merge_reviewed_labels
    labeled = str(tmp_path / "labeled.csv")
    pending = str(tmp_path / "pending.csv")
    output = str(tmp_path / "output.csv")
    make_labeled_csv(labeled, [{"author_id": "existing_1", "is_bot": 0}])
    make_pending_csv(pending, [
        {"author_id": "new_bot", "is_bot": "1"},
        {"author_id": "new_normal", "is_bot": "0"},
        {"author_id": "still_pending", "is_bot": ""},
    ])
    stats = merge_reviewed_labels(labeled, pending, output)
    assert stats["merged"] == 2
    assert stats["skipped_empty"] == 1
    assert stats["skipped_duplicate"] == 0
    result = pd.read_csv(output)
    assert len(result) == 3  # 1 existing + 2 merged


def test_skip_empty_is_bot(tmp_path):
    """空字串 is_bot 應被跳過"""
    from merge_labels import merge_reviewed_labels
    labeled = str(tmp_path / "labeled.csv")
    pending = str(tmp_path / "pending.csv")
    output = str(tmp_path / "output.csv")
    make_labeled_csv(labeled, [])
    make_pending_csv(pending, [
        {"author_id": "u1", "is_bot": ""},
        {"author_id": "u2", "is_bot": ""},
    ])
    stats = merge_reviewed_labels(labeled, pending, output)
    assert stats["merged"] == 0
    assert stats["skipped_empty"] == 2


def test_invalid_is_bot_raises_error(tmp_path):
    """非法 is_bot 值應拋 ValueError"""
    from merge_labels import merge_reviewed_labels
    labeled = str(tmp_path / "labeled.csv")
    pending = str(tmp_path / "pending.csv")
    output = str(tmp_path / "output.csv")
    make_labeled_csv(labeled, [])
    make_pending_csv(pending, [
        {"author_id": "u1", "is_bot": "yes"},
    ])
    with pytest.raises(ValueError, match="非法"):
        merge_reviewed_labels(labeled, pending, output)


def test_duplicate_author_id_skipped(tmp_path):
    """重複 author_id 保留既有版本，stderr 有警告"""
    from merge_labels import merge_reviewed_labels
    labeled = str(tmp_path / "labeled.csv")
    pending = str(tmp_path / "pending.csv")
    output = str(tmp_path / "output.csv")
    make_labeled_csv(labeled, [{"author_id": "existing_user", "is_bot": 0}])
    make_pending_csv(pending, [
        {"author_id": "existing_user", "is_bot": "1"},  # 重複！
    ])
    import io
    from contextlib import redirect_stderr
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        stats = merge_reviewed_labels(labeled, pending, output)
    assert stats["skipped_duplicate"] == 1
    assert stats["merged"] == 0
    stderr_output = stderr_capture.getvalue()
    assert "existing_user" in stderr_output or "重複" in stderr_output or "duplicate" in stderr_output.lower()
    # 既有版本保留（is_bot=0）
    result = pd.read_csv(output)
    existing_row = result[result["author_id"] == "existing_user"]
    assert existing_row["is_bot"].values[0] == 0


def test_partial_pending_review(tmp_path):
    """部分填寫的 pending，只合併有值的列"""
    from merge_labels import merge_reviewed_labels
    labeled = str(tmp_path / "labeled.csv")
    pending = str(tmp_path / "pending.csv")
    output = str(tmp_path / "output.csv")
    make_labeled_csv(labeled, [])
    make_pending_csv(pending, [
        {"author_id": "u1", "is_bot": "1"},
        {"author_id": "u2", "is_bot": ""},
        {"author_id": "u3", "is_bot": "0"},
        {"author_id": "u4", "is_bot": ""},
    ])
    stats = merge_reviewed_labels(labeled, pending, output)
    assert stats["merged"] == 2
    assert stats["skipped_empty"] == 2
