"""Integration tests for the AstroSentinel pipeline.

Tests verify that fixture data exists and has correct structure.
Subsequent agents will add complete pipeline tests.
"""
import pytest
import os
import pandas as pd


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_RAW = os.path.join(FIXTURES_DIR, "sample_raw.csv")
SAMPLE_LABELED = os.path.join(FIXTURES_DIR, "sample_labeled.csv")


def test_fixtures_exist():
    """Verify that test fixtures exist."""
    assert os.path.exists(SAMPLE_RAW), f"File not found: {SAMPLE_RAW}"
    assert os.path.exists(SAMPLE_LABELED), f"File not found: {SAMPLE_LABELED}"


def test_sample_raw_columns():
    """Verify sample_raw.csv has correct columns and structure."""
    df = pd.read_csv(SAMPLE_RAW)

    expected_cols = {
        "record_type", "comment_id", "video_id", "author_id",
        "author_name", "content", "like_count", "reply_count",
        "published_at", "parent_id"
    }
    assert expected_cols.issubset(set(df.columns)), \
        f"Missing columns. Have: {set(df.columns)}, need: {expected_cols}"

    assert len(df) >= 30, f"Expected >= 30 rows, got {len(df)}"


def test_sample_raw_boundary_conditions():
    """Verify sample_raw.csv covers required boundary conditions."""
    df = pd.read_csv(SAMPLE_RAW)

    # Single comment user (author_A)
    author_a = df[df["author_id"] == "author_A"]
    assert len(author_a) >= 1, "Missing author_A (single comment)"

    # Night time user (author_B) - 01:00, 02:30, 03:45 UTC
    author_b = df[df["author_id"] == "author_B"]
    assert len(author_b) >= 3, "Missing author_B (night comments)"

    # Ad keywords user (author_C)
    author_c = df[df["author_id"] == "author_C"]
    assert len(author_c) >= 2, "Missing author_C (ad keywords)"
    ad_content = author_c["content"].str.contains("訂閱|加LINE", regex=True, na=False)
    assert ad_content.any(), "author_C should have ad keywords"

    # URL user (author_D)
    author_d = df[df["author_id"] == "author_D"]
    assert len(author_d) >= 2, "Missing author_D (URL)"
    url_content = author_d["content"].str.contains("https://", regex=True, na=False)
    assert url_content.any(), "author_D should have URLs"

    # Burst user (author_E) - 3+ in same hour
    author_e = df[df["author_id"] == "author_E"]
    assert len(author_e) >= 4, "Missing author_E (burst)"

    # Reply type (author_F)
    author_f = df[df["author_id"] == "author_F"]
    assert len(author_f) >= 3, "Missing author_F (reply type)"
    assert (author_f["record_type"] == "reply").any(), "author_F should have reply records"

    # Zero likes (author_G)
    author_g = df[df["author_id"] == "author_G"]
    assert len(author_g) >= 3, "Missing author_G (zero likes)"
    assert (author_g["like_count"] == 0).any(), "author_G should have zero likes"


def test_sample_raw_data_types():
    """Verify sample_raw.csv has correct data types."""
    df = pd.read_csv(SAMPLE_RAW)

    assert df["like_count"].dtype in [int, 'int64'], "like_count should be integer"
    assert df["reply_count"].dtype in [int, 'int64'], "reply_count should be integer"
    assert df["record_type"].isin(["comment", "reply"]).all(), \
        "record_type should be 'comment' or 'reply'"


def test_sample_labeled_columns():
    """Verify sample_labeled.csv has correct columns and structure."""
    df = pd.read_csv(SAMPLE_LABELED)

    feature_cols = [
        "total_comments", "unique_videos", "concentration",
        "avg_likes", "zero_like_ratio", "max_burst",
        "avg_length", "unique_content_ratio", "self_similarity",
        "has_ad_keywords", "url_ratio", "night_ratio", "interval_std",
    ]

    assert "author_id" in df.columns, "Missing author_id column"
    assert "is_bot" in df.columns, "Missing is_bot column"

    for col in feature_cols:
        assert col in df.columns, f"Missing feature column: {col}"

    assert len(df) >= 50, f"Expected >= 50 rows, got {len(df)}"


def test_sample_labeled_has_both_classes():
    """Verify sample_labeled.csv has both bot and normal users."""
    df = pd.read_csv(SAMPLE_LABELED)

    bot_count = (df["is_bot"] == 1).sum()
    normal_count = (df["is_bot"] == 0).sum()

    assert bot_count >= 5, f"Expected >= 5 bots, got {bot_count}"
    assert normal_count >= 5, f"Expected >= 5 normal users, got {normal_count}"


def test_sample_labeled_bot_characteristics():
    """Verify bot samples have expected characteristics."""
    df = pd.read_csv(SAMPLE_LABELED)

    bots = df[df["is_bot"] == 1]

    # Check average bot characteristics
    # Bots should have high concentration, high self_similarity, high zero_like_ratio, low interval_std
    assert (bots["concentration"] > 5).any(), "Some bots should have high concentration"
    assert (bots["self_similarity"] > 0.7).any(), "Some bots should have high self_similarity"
    assert (bots["zero_like_ratio"] > 0.8).any(), "Some bots should have high zero_like_ratio"


def test_sample_labeled_normal_characteristics():
    """Verify normal user samples have expected characteristics."""
    df = pd.read_csv(SAMPLE_LABELED)

    normal = df[df["is_bot"] == 0]

    # Normal users should have diverse videos and low self_similarity
    assert (normal["unique_videos"] >= 3).any(), "Some normal users should have 3+ videos"
    assert (normal["self_similarity"] < 0.3).any(), "Some normal users should have low self_similarity"
    assert (normal["zero_like_ratio"] < 0.5).any(), "Some normal users should have low zero_like_ratio"


def test_sample_labeled_unique_authors():
    """Verify sample_labeled.csv has 50+ unique authors."""
    df = pd.read_csv(SAMPLE_LABELED)

    unique_authors = df["author_id"].nunique()
    assert unique_authors >= 50, f"Expected >= 50 unique authors, got {unique_authors}"


def test_sample_labeled_value_ranges():
    """Verify feature values are within reasonable ranges."""
    df = pd.read_csv(SAMPLE_LABELED)

    # All features should be non-negative
    assert (df["total_comments"] > 0).all(), "total_comments should be > 0"
    assert (df["unique_videos"] > 0).all(), "unique_videos should be > 0"
    assert (df["avg_likes"] >= 0).all(), "avg_likes should be >= 0"
    assert (df["max_burst"] > 0).all(), "max_burst should be > 0"
    assert (df["avg_length"] > 0).all(), "avg_length should be > 0"
    assert (df["interval_std"] >= 0).all(), "interval_std should be >= 0"

    # Ratios should be between 0 and 1
    assert (df["zero_like_ratio"] >= 0).all() and (df["zero_like_ratio"] <= 1).all(), \
        "zero_like_ratio should be [0, 1]"
    assert (df["unique_content_ratio"] >= 0).all() and (df["unique_content_ratio"] <= 1).all(), \
        "unique_content_ratio should be [0, 1]"
    assert (df["self_similarity"] >= 0).all() and (df["self_similarity"] <= 1).all(), \
        "self_similarity should be [0, 1]"
    assert (df["has_ad_keywords"].isin([0, 1])).all(), "has_ad_keywords should be 0 or 1"
    assert (df["url_ratio"] >= 0).all() and (df["url_ratio"] <= 1).all(), \
        "url_ratio should be [0, 1]"
    assert (df["night_ratio"] >= 0).all() and (df["night_ratio"] <= 1).all(), \
        "night_ratio should be [0, 1]"
