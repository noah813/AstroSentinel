import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone


# ── 測試 1：load_collected_ids — 檔案不存在時回傳空集合 ──────────────────────────
def test_load_collected_ids_missing_file():
    from collector import load_collected_ids
    result = load_collected_ids("/nonexistent/path.json")
    assert result == {"video_ids": set(), "comment_ids": set()}


# ── 測試 2：load_collected_ids — 正常讀取並轉 set ────────────────────────────────
def test_load_collected_ids_existing_file(tmp_path):
    from collector import load_collected_ids
    f = tmp_path / "ids.json"
    f.write_text('{"video_ids": ["v1","v2"], "comment_ids": ["c1"]}')
    result = load_collected_ids(str(f))
    assert result["video_ids"] == {"v1", "v2"}
    assert result["comment_ids"] == {"c1"}


# ── 測試 3：save_collected_ids — set 序列化為排序 list ───────────────────────────
def test_save_collected_ids(tmp_path):
    from collector import save_collected_ids, load_collected_ids
    f = tmp_path / "ids.json"
    save_collected_ids({"video_ids": {"v1", "v2"}, "comment_ids": {"c1"}}, str(f))
    loaded = json.loads(f.read_text())
    assert set(loaded["video_ids"]) == {"v1", "v2"}
    assert set(loaded["comment_ids"]) == {"c1"}


# ── 測試 4：check_quota — 達限時拋例外 ──────────────────────────────────────────
def test_check_quota_exceeds():
    from collector import check_quota, QuotaExceededError
    with pytest.raises(QuotaExceededError) as exc_info:
        check_quota(9000, 9000)
    assert exc_info.value.used == 9000


# ── 測試 5：check_quota — 未達限不拋例外 ─────────────────────────────────────────
def test_check_quota_within_limit():
    from collector import check_quota
    check_quota(8999, 9000)  # 不應拋例外


# ── 測試 6：fetch_channel_uploads_playlist — mock API 回傳 ───────────────────────
def test_fetch_channel_uploads_playlist():
    from collector import fetch_channel_uploads_playlist
    mock_service = MagicMock()
    mock_service.channels().list().execute.return_value = {
        "items": [{
            "contentDetails": {
                "relatedPlaylists": {"uploads": "UUtest123"}
            }
        }]
    }
    result = fetch_channel_uploads_playlist(mock_service, "UCtest")
    assert result == "UUtest123"


# ── 測試 7：fetch_recent_videos — 過濾 since 時間，skip 已收集 ──────────────────
def test_fetch_recent_videos_filters():
    from collector import fetch_recent_videos
    since = datetime(2026, 5, 25, tzinfo=timezone.utc)
    mock_service = MagicMock()
    mock_service.playlistItems().list().execute.return_value = {
        "items": [
            {
                "snippet": {
                    "resourceId": {"videoId": "v_new"},
                    "title": "New Video",
                    "channelId": "ch1",
                    "publishedAt": "2026-05-28T10:00:00Z",
                }
            },
            {
                "snippet": {
                    "resourceId": {"videoId": "v_old"},
                    "title": "Old Video",
                    "channelId": "ch1",
                    "publishedAt": "2026-05-20T10:00:00Z",
                }
            },
            {
                "snippet": {
                    "resourceId": {"videoId": "v_collected"},
                    "title": "Collected",
                    "channelId": "ch1",
                    "publishedAt": "2026-05-27T10:00:00Z",
                }
            },
        ],
        "nextPageToken": None,
    }
    result = fetch_recent_videos(
        mock_service, "PLtest", since, collected_video_ids={"v_collected"}
    )
    video_ids = [r["video_id"] for r in result]
    assert "v_new" in video_ids
    assert "v_old" not in video_ids
    assert "v_collected" not in video_ids


# ── 測試 8：fetch_comments — 正確填入 parent_id ─────────────────────────────────
def test_fetch_comments_parent_id():
    from collector import fetch_comments
    mock_service = MagicMock()
    mock_service.commentThreads().list().execute.return_value = {
        "items": [{
            "id": "thread1",
            "snippet": {
                "topLevelComment": {
                    "id": "c001",
                    "snippet": {
                        "authorChannelId": {"value": "auth1"},
                        "authorDisplayName": "User1",
                        "textOriginal": "Hello",
                        "likeCount": 5,
                        "publishedAt": "2026-05-28T10:00:00Z",
                    }
                },
                "totalReplyCount": 1,
            },
            "replies": {
                "comments": [{
                    "id": "r001",
                    "snippet": {
                        "authorChannelId": {"value": "auth2"},
                        "authorDisplayName": "User2",
                        "textOriginal": "Reply here",
                        "likeCount": 0,
                        "publishedAt": "2026-05-28T11:00:00Z",
                        "parentId": "thread1",
                    }
                }]
            }
        }],
        "nextPageToken": None,
    }
    result = fetch_comments(mock_service, "video1", collected_comment_ids=set())
    top_level = [r for r in result if r["record_type"] == "comment"]
    replies = [r for r in result if r["record_type"] == "reply"]
    assert len(top_level) == 1
    assert top_level[0]["parent_id"] == ""
    assert len(replies) == 1
    assert replies[0]["parent_id"] == "thread1"
