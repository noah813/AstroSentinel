import os
import json
import logging
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import config

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    def __init__(self, used: int, limit: int):
        super().__init__(f"配額耗盡：已用 {used}，上限 {limit}")
        self.used = used
        self.limit = limit


def load_collected_ids(path: str) -> dict[str, set[str]]:
    """檔案不存在時回傳空集合，不拋例外。"""
    if not os.path.exists(path):
        return {"video_ids": set(), "comment_ids": set()}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "video_ids": set(data.get("video_ids", [])),
            "comment_ids": set(data.get("comment_ids", [])),
        }
    except Exception as e:
        logger.warning(f"無法讀取 collected_ids 檔案 {path}：{e}，回傳空集合")
        return {"video_ids": set(), "comment_ids": set()}


def save_collected_ids(ids: dict[str, set[str]], path: str) -> None:
    """將 set 序列化為 list 後存入 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    serializable = {
        "video_ids": sorted(list(ids.get("video_ids", set()))),
        "comment_ids": sorted(list(ids.get("comment_ids", set()))),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def check_quota(used: int, limit: int) -> None:
    """used >= limit 時拋 QuotaExceededError。"""
    if used >= limit:
        raise QuotaExceededError(used, limit)


def fetch_channel_uploads_playlist(service, channel_id: str) -> str:
    """呼叫 channels.list 取得 uploads playlist ID。"""
    result = service.channels().list(
        part="contentDetails",
        id=channel_id,
    ).execute()
    return result["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_recent_videos(
    service,
    playlist_id: str,
    since: datetime,
    collected_video_ids: set[str],
) -> list[dict]:
    """逐頁呼叫 playlistItems.list，跳過 since 前或已收集的影片。"""
    videos = []
    next_page_token = None

    while True:
        request = service.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        response = request.execute()

        stop_paging = False
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            video_id = snippet.get("resourceId", {}).get("videoId", "")
            published_at_str = snippet.get("publishedAt", "")

            # Parse published_at
            try:
                published_at = datetime.fromisoformat(
                    published_at_str.replace("Z", "+00:00")
                )
            except Exception:
                logger.warning(f"無法解析時間 {published_at_str}，跳過影片 {video_id}")
                continue

            # Playlist is in reverse chronological order; stop if older than since
            if published_at < since:
                stop_paging = True
                break

            # Skip already-collected videos
            if video_id in collected_video_ids:
                continue

            videos.append({
                "video_id": video_id,
                "video_title": snippet.get("title", ""),
                "channel_id": snippet.get("channelId", ""),
                "published_at": published_at_str,
            })

        next_page_token = response.get("nextPageToken")
        if stop_paging or not next_page_token:
            break

    return videos


def fetch_video_stats(service, video_ids: list[str]) -> dict[str, dict]:
    """批次取得 view_count, comment_count，每批最多 50 個。"""
    stats = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        response = service.videos().list(
            part="statistics",
            id=",".join(batch),
        ).execute()
        for item in response.get("items", []):
            vid = item["id"]
            s = item.get("statistics", {})
            stats[vid] = {
                "view_count": int(s.get("viewCount", 0)),
                "comment_count": int(s.get("commentCount", 0)),
            }
    # Fill missing video IDs with zeros
    for vid in video_ids:
        if vid not in stats:
            stats[vid] = {"view_count": 0, "comment_count": 0}
    return stats


def fetch_comments(
    service,
    video_id: str,
    collected_comment_ids: set[str],
) -> list[dict]:
    """取頂層留言及回覆，跳過已收集 ID，parent_id 正確填入。"""
    comments = []
    next_page_token = None

    while True:
        request = service.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token,
        )
        response = request.execute()

        for thread in response.get("items", []):
            thread_id = thread["id"]
            top_snippet = thread["snippet"]["topLevelComment"]["snippet"]
            top_comment_id = thread["snippet"]["topLevelComment"]["id"]
            total_reply_count = thread["snippet"].get("totalReplyCount", 0)

            # Top-level comment
            if top_comment_id not in collected_comment_ids:
                comments.append({
                    "record_type": "comment",
                    "comment_id": top_comment_id,
                    "video_id": video_id,
                    "author_id": top_snippet.get("authorChannelId", {}).get("value", ""),
                    "author_name": top_snippet.get("authorDisplayName", ""),
                    "content": top_snippet.get("textOriginal", top_snippet.get("textDisplay", "")),
                    "like_count": int(top_snippet.get("likeCount", 0)),
                    "reply_count": int(total_reply_count),
                    "published_at": top_snippet.get("publishedAt", ""),
                    "parent_id": "",
                })

            # Replies
            if total_reply_count > 0:
                # Check if replies are fully included in the thread response
                thread_replies = thread.get("replies", {}).get("comments", [])
                if len(thread_replies) < total_reply_count:
                    # Need to fetch all replies via comments.list
                    replies_data = _fetch_all_replies(service, thread_id, collected_comment_ids)
                    comments.extend(replies_data)
                else:
                    for reply in thread_replies:
                        reply_id = reply["id"]
                        if reply_id in collected_comment_ids:
                            continue
                        reply_snippet = reply["snippet"]
                        comments.append({
                            "record_type": "reply",
                            "comment_id": reply_id,
                            "video_id": video_id,
                            "author_id": reply_snippet.get("authorChannelId", {}).get("value", ""),
                            "author_name": reply_snippet.get("authorDisplayName", ""),
                            "content": reply_snippet.get("textOriginal", reply_snippet.get("textDisplay", "")),
                            "like_count": int(reply_snippet.get("likeCount", 0)),
                            "reply_count": 0,
                            "published_at": reply_snippet.get("publishedAt", ""),
                            "parent_id": thread_id,
                        })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def _fetch_all_replies(
    service,
    thread_id: str,
    collected_comment_ids: set[str],
) -> list[dict]:
    """呼叫 comments.list 取得指定 thread 的所有回覆。"""
    replies = []
    next_page_token = None

    while True:
        request = service.comments().list(
            part="snippet",
            parentId=thread_id,
            maxResults=100,
            pageToken=next_page_token,
        )
        response = request.execute()

        for reply in response.get("items", []):
            reply_id = reply["id"]
            if reply_id in collected_comment_ids:
                continue
            reply_snippet = reply["snippet"]
            # parentId in snippet refers to the thread (top-level comment ID)
            replies.append({
                "record_type": "reply",
                "comment_id": reply_id,
                "video_id": reply_snippet.get("videoId", ""),
                "author_id": reply_snippet.get("authorChannelId", {}).get("value", ""),
                "author_name": reply_snippet.get("authorDisplayName", ""),
                "content": reply_snippet.get("textOriginal", reply_snippet.get("textDisplay", "")),
                "like_count": int(reply_snippet.get("likeCount", 0)),
                "reply_count": 0,
                "published_at": reply_snippet.get("publishedAt", ""),
                "parent_id": thread_id,
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


# Full schema columns for CSV output
_VIDEO_COLS = ["record_type", "video_id", "video_title", "channel_id", "published_at",
               "view_count", "comment_count"]
_COMMENT_COLS = ["record_type", "comment_id", "video_id", "author_id", "author_name",
                 "content", "like_count", "reply_count", "published_at", "parent_id"]
_ALL_COLS = list(dict.fromkeys(_VIDEO_COLS + _COMMENT_COLS))


def _make_video_row(video: dict, stats: dict) -> dict:
    """Build a complete row dict for a video record."""
    s = stats.get(video["video_id"], {"view_count": 0, "comment_count": 0})
    return {
        "record_type": "video",
        "video_id": video["video_id"],
        "video_title": video["video_title"],
        "channel_id": video["channel_id"],
        "published_at": video["published_at"],
        "view_count": s["view_count"],
        "comment_count": s["comment_count"],
        # Comment-specific columns filled with empty string
        "comment_id": "",
        "author_id": "",
        "author_name": "",
        "content": "",
        "like_count": "",
        "reply_count": "",
        "parent_id": "",
    }


def _make_comment_row(comment: dict) -> dict:
    """Build a complete row dict for a comment/reply record."""
    return {
        "record_type": comment["record_type"],
        "comment_id": comment["comment_id"],
        "video_id": comment["video_id"],
        "author_id": comment["author_id"],
        "author_name": comment["author_name"],
        "content": comment["content"],
        "like_count": comment["like_count"],
        "reply_count": comment["reply_count"],
        "published_at": comment["published_at"],
        "parent_id": comment["parent_id"],
        # Video-specific columns filled with empty string
        "video_title": "",
        "channel_id": "",
        "view_count": "",
        "comment_count": "",
    }


def run_collection(
    channel_ids: list[str],
    output_dir: str,
    collected_ids_path: str,
    quota_limit: int = 9_000,
) -> str:
    """主流程，回傳輸出 CSV 路徑。"""
    # Build YouTube service
    service = build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)

    # Load already-collected IDs
    collected_ids = load_collected_ids(collected_ids_path)
    collected_video_ids: set[str] = collected_ids["video_ids"]
    collected_comment_ids: set[str] = collected_ids["comment_ids"]

    # Determine since datetime
    since = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)

    # Determine output CSV path
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    base_path = os.path.join(output_dir, f"youtube_raw_{today_str}.csv")
    if os.path.exists(base_path):
        suffix = datetime.now(timezone.utc).strftime("%H%M%S")
        output_path = os.path.join(output_dir, f"youtube_raw_{today_str}_{suffix}.csv")
    else:
        output_path = base_path

    os.makedirs(output_dir, exist_ok=True)

    quota_used = 0
    all_rows: list[dict] = []
    remaining_channels = list(channel_ids)

    try:
        for channel_id in channel_ids:
            remaining_channels = [c for c in remaining_channels if c != channel_id]

            # 1. Fetch uploads playlist ID
            check_quota(quota_used, quota_limit)
            try:
                playlist_id = fetch_channel_uploads_playlist(service, channel_id)
            except HttpError as e:
                logger.warning(f"無法取得頻道 {channel_id} 的 playlist：{e}")
                continue
            quota_used += 1
            logger.info(f"頻道 {channel_id} playlist: {playlist_id}")

            # 2. Fetch recent videos
            check_quota(quota_used, quota_limit)
            try:
                videos = fetch_recent_videos(
                    service, playlist_id, since, collected_video_ids
                )
            except HttpError as e:
                logger.warning(f"無法取得頻道 {channel_id} 的影片列表：{e}")
                continue
            # Count pages fetched — approximate by number of API calls;
            # fetch_recent_videos may call multiple pages but we track as 1 per call externally
            # For simplicity, we count one call per fetch_recent_videos invocation
            quota_used += 1
            logger.info(f"頻道 {channel_id}：取得 {len(videos)} 部新影片")

            if not videos:
                continue

            video_ids_new = [v["video_id"] for v in videos]

            # 3. Fetch video statistics (one call per batch of 50)
            for i in range(0, len(video_ids_new), 50):
                check_quota(quota_used, quota_limit)
                quota_used += 1

            try:
                stats = fetch_video_stats(service, video_ids_new)
            except HttpError as e:
                logger.warning(f"無法取得影片統計資料：{e}")
                stats = {}

            # Build video rows
            for video in videos:
                row = _make_video_row(video, stats)
                all_rows.append(row)
                collected_video_ids.add(video["video_id"])

            # 4. Fetch comments for each video
            for video in videos:
                vid = video["video_id"]
                check_quota(quota_used, quota_limit)
                try:
                    comments = fetch_comments(service, vid, collected_comment_ids)
                    quota_used += max(1, len(comments) // 100 + 1)
                except HttpError as e:
                    logger.warning(f"取得影片 {vid} 留言失敗：{e}，跳過")
                    continue

                for comment in comments:
                    all_rows.append(_make_comment_row(comment))
                    collected_comment_ids.add(comment["comment_id"])

                logger.info(f"影片 {vid}：收集 {len(comments)} 則留言/回覆")

    except QuotaExceededError as e:
        logger.error(str(e))
        print(f"配額耗盡：{e}", file=sys.stderr)
        if remaining_channels:
            print(f"未收集頻道：{remaining_channels}", file=sys.stderr)

    # Write CSV (even if quota exceeded, write what we have)
    df = pd.DataFrame(all_rows)
    if df.empty:
        # Create empty DataFrame with all columns
        df = pd.DataFrame(columns=_ALL_COLS)
    else:
        # Ensure all columns exist
        for col in _ALL_COLS:
            if col not in df.columns:
                df[col] = ""

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"已寫入 CSV：{output_path}（{len(df)} 列）")

    # Save updated collected IDs
    save_collected_ids(
        {"video_ids": collected_video_ids, "comment_ids": collected_comment_ids},
        collected_ids_path,
    )

    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    csv_path = run_collection(
        channel_ids=config.CHANNEL_IDS,
        output_dir=config.COLLECTED_DIR,
        collected_ids_path=config.COLLECTED_IDS_PATH,
        quota_limit=config.QUOTA_DAILY_LIMIT,
    )
    print(f"Collection complete: {csv_path}")
