import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_row(**kwargs):
    """Create a pd.Series with default feature values"""
    defaults = {
        "author_id": "test_user",
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


def get_mock_client(response_text):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def test_classify_user_llm_bot():
    """Test LLM correctly parsing 'bot' response"""
    from auto_labeler import classify_user_llm
    row = make_row()
    mock_client = get_mock_client("This user exhibits bot behavior. bot")
    assert classify_user_llm(row, mock_client) == "bot"


def test_classify_user_llm_normal():
    """Test LLM correctly parsing 'normal' response"""
    from auto_labeler import classify_user_llm
    row = make_row()
    mock_client = get_mock_client("The user is normal.")
    assert classify_user_llm(row, mock_client) == "normal"


def test_classify_user_llm_pending():
    """Test LLM correctly parsing 'pending' response"""
    from auto_labeler import classify_user_llm
    row = make_row()
    mock_client = get_mock_client("I am unsure. pending")
    assert classify_user_llm(row, mock_client) == "pending"


@patch("time.sleep", return_value=None)
def test_classify_user_llm_fallback(mock_sleep):
    """Test LLM failure falls back to 'pending'"""
    from auto_labeler import classify_user_llm
    row = make_row()
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error")
    assert classify_user_llm(row, mock_client) == "pending"


@patch("auto_labeler.genai.Client")
@patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"})
def test_run_labeling_integration(mock_client_class, tmp_path):
    """Integration test for run_labeling with mocked LLM"""
    from auto_labeler import run_labeling
    
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    
    # Let the first response be 'bot', second 'normal', rest 'pending'
    call_count = [0]
    def mock_generate_content(*args, **kwargs):
        mock_response = MagicMock()
        if call_count[0] == 0:
            mock_response.text = "bot"
        elif call_count[0] == 1:
            mock_response.text = "normal"
        else:
            mock_response.text = "pending"
        call_count[0] += 1
        return mock_response
        
    mock_instance.models.generate_content.side_effect = mock_generate_content

    features_path = "tests/fixtures/sample_labeled.csv"
    labeled_out = str(tmp_path / "labeled.csv")
    pending_out = str(tmp_path / "pending.csv")

    stats = run_labeling(features_path, labeled_out, pending_out)

    assert "bot" in stats
    assert "normal" in stats
    assert "pending" in stats
    
    assert os.path.exists(labeled_out)
    
    labeled_df = pd.read_csv(labeled_out)
    if len(labeled_df) > 0:
        assert "author_id" in labeled_df.columns
        assert "is_bot" in labeled_df.columns
        assert set(labeled_df["is_bot"].unique()).issubset({0, 1})

    pending_df = pd.read_csv(pending_out, dtype={"is_bot": str}, keep_default_na=False)
    if len(pending_df) > 0:
        is_bot_vals = pending_df["is_bot"].tolist()
        for val in is_bot_vals:
            assert val == "", f"is_bot should be empty string, got: {repr(val)}"
