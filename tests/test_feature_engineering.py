import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_RAW = os.path.join(FIXTURES_DIR, "sample_raw.csv")


def get_sample_df():
    """載入 sample_raw.csv 並轉換 published_at"""
    df = pd.read_csv(SAMPLE_RAW)
    df = df[df["record_type"].isin(["comment", "reply"])].copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def test_load_raw_comments_deduplication(tmp_path):
    """相同 comment_id 只保留一筆"""
    from feature_engineering import load_raw_comments
    import shutil
    collected_dir = tmp_path / "data" / "collected"
    collected_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_RAW, collected_dir / "youtube_raw_test.csv")
    # 複製兩份（模擬重複）
    shutil.copy(SAMPLE_RAW, collected_dir / "youtube_raw_test2.csv")
    result = load_raw_comments(str(tmp_path / "data"), lookback_days=3650)
    comment_ids = result["comment_id"].tolist()
    assert len(comment_ids) == len(set(comment_ids)), "comment_id 應去重"


def test_self_similarity_single_comment():
    """單一留言用戶的 self_similarity 應為 0.0"""
    from feature_engineering import compute_content_features
    df = get_sample_df()
    # 只取 author_A（應只有 1 則留言）
    single = df[df["author_id"] == "author_A"]
    if len(single) == 0:
        pytest.skip("sample_raw.csv 中找不到 author_A")
    result = compute_content_features(single)
    if "author_A" in result.index:
        assert result.loc["author_A", "self_similarity"] == 0.0


def test_interval_std_single_comment():
    """單一留言用戶的 interval_std 應為 0.0"""
    from feature_engineering import compute_temporal_features
    df = get_sample_df()
    single = df[df["author_id"] == "author_A"]
    if len(single) == 0:
        pytest.skip("sample_raw.csv 中找不到 author_A")
    result = compute_temporal_features(single)
    if "author_A" in result.index:
        assert result.loc["author_A", "interval_std"] == 0.0


def test_concentration_zero_videos():
    """unique_videos=0 時 concentration 應填 0（防禦案例）"""
    from feature_engineering import compute_behavioral_features
    df = pd.DataFrame({
        "author_id": ["edge_user"],
        "video_id": [None],  # 空 video_id
        "like_count": [0],
        "published_at": [pd.Timestamp("2026-05-28T10:00:00Z", tz="UTC")],
        "comment_id": ["c_edge"],
    })
    df["video_id"] = None
    result = compute_behavioral_features(df)
    if "edge_user" in result.index:
        assert result.loc["edge_user", "concentration"] == 0


def test_night_ratio_utc():
    """night_ratio 以 UTC 計算，凌晨時間應計入"""
    from feature_engineering import compute_temporal_features
    df = pd.DataFrame({
        "author_id": ["night_user", "night_user", "night_user", "day_user"],
        "published_at": pd.to_datetime([
            "2026-05-28T01:00:00Z",  # 凌晨 1 點 UTC → night
            "2026-05-28T03:00:00Z",  # 凌晨 3 點 UTC → night
            "2026-05-28T12:00:00Z",  # 中午 → not night
            "2026-05-28T14:00:00Z",  # 下午 → not night
        ], utc=True),
        "comment_id": ["n1", "n2", "n3", "d1"],
    })
    result = compute_temporal_features(df)
    night_user_ratio = result.loc["night_user", "night_ratio"]
    assert abs(night_user_ratio - 2/3) < 0.01, f"night_ratio 應為 2/3，實際：{night_user_ratio}"
    assert result.loc["day_user", "night_ratio"] == 0.0


def test_run_feature_engineering_output(tmp_path):
    """完整執行產生正確欄位"""
    from feature_engineering import run_feature_engineering
    import shutil
    import config
    collected_dir = tmp_path / "data" / "collected"
    collected_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_RAW, collected_dir / "youtube_raw_test.csv")
    output = str(tmp_path / "user_features.csv")
    run_feature_engineering(str(tmp_path / "data"), output, lookback_days=3650)
    df = pd.read_csv(output)
    assert "author_id" in df.columns
    for col in config.FEATURE_COLS:
        assert col in df.columns, f"缺少特徵欄位: {col}"
    assert len(df.columns) == len(config.FEATURE_COLS) + 1  # +1 for author_id


def test_max_burst():
    """同一小時內多則留言，max_burst 應正確計算"""
    from feature_engineering import compute_behavioral_features
    df = pd.DataFrame({
        "author_id": ["burst_user"] * 5,
        "video_id": ["v1"] * 5,
        "like_count": [0] * 5,
        "published_at": pd.to_datetime([
            "2026-05-28T10:00:00Z",
            "2026-05-28T10:15:00Z",
            "2026-05-28T10:30:00Z",
            "2026-05-28T10:45:00Z",
            "2026-05-28T11:00:00Z",  # 下一小時
        ], utc=True),
        "comment_id": [f"c{i}" for i in range(5)],
    })
    result = compute_behavioral_features(df)
    assert result.loc["burst_user", "max_burst"] == 4  # 10:00-10:45 有 4 則
