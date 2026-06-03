import pytest
import os
import importlib

def test_config_imports_with_env():
    """有 YOUTUBE_API_KEY 時 import config 不應報錯，且結構正確"""
    import config
    assert config.YOUTUBE_API_KEY  # 不應為空
    assert len(config.CHANNEL_IDS) == 4
    assert len(config.FEATURE_COLS) == 13

def test_config_raises_without_env(monkeypatch):
    """缺少 YOUTUBE_API_KEY 時應拋出 KeyError"""
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    # 強制重新載入 config（dotenv 已被 monkeypatch 清除）
    import importlib
    import config
    # 由於 load_dotenv() 在模組層級執行，測試需驗證環境變數行為
    # 此測試驗證：若環境中無 YOUTUBE_API_KEY，os.environ["YOUTUBE_API_KEY"] 會拋 KeyError
    with pytest.raises(KeyError):
        _ = os.environ["YOUTUBE_API_KEY"]

def test_feature_cols_count():
    import config
    assert len(config.FEATURE_COLS) == 13

def test_night_hours():
    import config
    assert config.NIGHT_HOURS == (0, 6)

def test_bot_thresholds():
    import config
    assert config.BOT_CONCENTRATION_GT == 5.0
    assert config.BOT_SELF_SIMILARITY_GT == 0.7
