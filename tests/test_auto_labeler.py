import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_row(**kwargs):
    """Create a pd.Series with default feature values"""
    defaults = {
        "concentration": 0.0,
        "self_similarity": 0.0,
        "zero_like_ratio": 0.0,
        "interval_std": 100.0,
        "unique_videos": 1,
        "total_comments": 1,
        "avg_likes": 0.0,
        "max_burst": 1,
        "avg_length": 10.0,
        "unique_content_ratio": 1.0,
        "has_ad_keywords": 0,
        "url_ratio": 0.0,
        "night_ratio": 0.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_classify_bot():
    """All 4 bot conditions satisfied should return 'bot'"""
    from auto_labeler import classify_user

    row = make_row(
        concentration=5.01,
        self_similarity=0.71,
        zero_like_ratio=0.81,
        interval_std=59.9,
    )
    assert classify_user(row) == "bot"


def test_classify_bot_boundary_equal():
    """concentration=5.0 should NOT be classified as bot (strict >)"""
    from auto_labeler import classify_user

    row = make_row(
        concentration=5.0,  # Equal sign does not satisfy >
        self_similarity=0.71,
        zero_like_ratio=0.81,
        interval_std=59.9,
    )
    assert classify_user(row) != "bot"


def test_classify_normal():
    """All 3 normal conditions satisfied should return 'normal'"""
    from auto_labeler import classify_user

    row = make_row(
        unique_videos=3,
        self_similarity=0.29,
        zero_like_ratio=0.49,
    )
    assert classify_user(row) == "normal"


def test_classify_pending():
    """Neither condition satisfied should return 'pending'"""
    from auto_labeler import classify_user

    row = make_row(
        concentration=2.0,
        self_similarity=0.5,
        zero_like_ratio=0.5,
        unique_videos=1,
    )
    assert classify_user(row) == "pending"


def test_bot_priority_over_normal():
    """Bot conditions take priority when both are satisfied"""
    from auto_labeler import classify_user

    row = make_row(
        concentration=6.0,  # bot: > 5.0 ✓
        self_similarity=0.8,  # bot: > 0.7 ✓
        zero_like_ratio=0.9,  # bot: > 0.8 ✓
        interval_std=50.0,  # bot: < 60.0 ✓
        unique_videos=3,  # normal: >= 3 ✓
        # normal: self_similarity < 0.3? No, 0.8 > 0.3
        # normal: zero_like_ratio < 0.5? No, 0.9 > 0.5
    )
    # Bot conditions all satisfied → should return bot
    assert classify_user(row) == "bot"


def test_pending_is_bot_empty_string(tmp_path):
    """pending_review.csv is_bot column must be empty string, not NaN"""
    from auto_labeler import run_labeling
    import config

    features_path = "tests/fixtures/sample_labeled.csv"
    labeled_out = str(tmp_path / "labeled.csv")
    pending_out = str(tmp_path / "pending.csv")

    run_labeling(features_path, labeled_out, pending_out)

    # Check pending is_bot values
    if os.path.exists(pending_out):
        # Use keep_default_na=False to preserve empty strings
        pending_df = pd.read_csv(pending_out, dtype={"is_bot": str}, keep_default_na=False)
        if len(pending_df) > 0:
            is_bot_vals = pending_df["is_bot"].tolist()
            for val in is_bot_vals:
                assert val == "", f"is_bot should be empty string, got: {repr(val)}"


def test_labeled_no_empty_is_bot(tmp_path):
    """labeled_data.csv is_bot column must only contain 0 or 1"""
    from auto_labeler import run_labeling

    features_path = "tests/fixtures/sample_labeled.csv"
    labeled_out = str(tmp_path / "labeled.csv")
    pending_out = str(tmp_path / "pending.csv")

    run_labeling(features_path, labeled_out, pending_out)

    if os.path.exists(labeled_out):
        labeled_df = pd.read_csv(labeled_out)
        if len(labeled_df) > 0:
            assert set(labeled_df["is_bot"].unique()).issubset({0, 1})


def test_classify_bot_boundary_conditions():
    """Test all bot boundary conditions individually"""
    from auto_labeler import classify_user

    # concentration at boundary (should be bot when > 5.0)
    row1 = make_row(
        concentration=5.00001,
        self_similarity=0.71,
        zero_like_ratio=0.81,
        interval_std=59.9,
    )
    assert classify_user(row1) == "bot"

    # self_similarity at boundary
    row2 = make_row(
        concentration=5.01,
        self_similarity=0.70001,
        zero_like_ratio=0.81,
        interval_std=59.9,
    )
    assert classify_user(row2) == "bot"

    # zero_like_ratio at boundary
    row3 = make_row(
        concentration=5.01,
        self_similarity=0.71,
        zero_like_ratio=0.80001,
        interval_std=59.9,
    )
    assert classify_user(row3) == "bot"

    # interval_std at boundary
    row4 = make_row(
        concentration=5.01,
        self_similarity=0.71,
        zero_like_ratio=0.81,
        interval_std=59.99999,
    )
    assert classify_user(row4) == "bot"


def test_classify_normal_boundary_conditions():
    """Test all normal boundary conditions individually"""
    from auto_labeler import classify_user

    # unique_videos at boundary (>= 3)
    row1 = make_row(
        unique_videos=3,
        self_similarity=0.29,
        zero_like_ratio=0.49,
    )
    assert classify_user(row1) == "normal"

    # self_similarity at boundary (< 0.3)
    row2 = make_row(
        unique_videos=3,
        self_similarity=0.29999,
        zero_like_ratio=0.49,
    )
    assert classify_user(row2) == "normal"

    # zero_like_ratio at boundary (< 0.5)
    row3 = make_row(
        unique_videos=3,
        self_similarity=0.29,
        zero_like_ratio=0.49999,
    )
    assert classify_user(row3) == "normal"


def test_run_labeling_integration(tmp_path):
    """Integration test for run_labeling with fixture"""
    from auto_labeler import run_labeling

    features_path = "tests/fixtures/sample_labeled.csv"
    labeled_out = str(tmp_path / "labeled.csv")
    pending_out = str(tmp_path / "pending.csv")

    stats = run_labeling(features_path, labeled_out, pending_out)

    # Verify statistics dict structure
    assert "bot" in stats
    assert "normal" in stats
    assert "pending" in stats
    assert isinstance(stats["bot"], (int, np.integer))
    assert isinstance(stats["normal"], (int, np.integer))
    assert isinstance(stats["pending"], (int, np.integer))

    # Verify output files exist
    assert os.path.exists(labeled_out)

    # Verify labeled CSV structure
    labeled_df = pd.read_csv(labeled_out)
    assert "author_id" in labeled_df.columns
    assert "is_bot" in labeled_df.columns
    assert set(labeled_df["is_bot"].unique()).issubset({0, 1})
