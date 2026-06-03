"""生成 sample_raw.csv - 30 筆假留言，涵蓋邊界情境"""
import csv
from datetime import datetime, timedelta
import os

# 輸出路徑
output_path = os.path.join(os.path.dirname(__file__), "sample_raw.csv")

# 準備資料
rows = []

# 1. author_A（單一留言用戶）- 1 則留言
rows.append({
    "record_type": "comment",
    "comment_id": "c001",
    "video_id": "v001",
    "author_id": "author_A",
    "author_name": "用戶A",
    "content": "這是一則精美的留言",
    "like_count": 5,
    "reply_count": 0,
    "published_at": "2026-05-28T10:00:00Z",
    "parent_id": ""
})

# 2. author_B（凌晨發文用戶）- 3 則留言
base_date = "2026-05-28T"
for idx, time_str in enumerate(["01:00:00Z", "02:30:00Z", "03:45:00Z"]):
    rows.append({
        "record_type": "comment",
        "comment_id": f"c_{2 + idx:02d}",
        "video_id": "v002",
        "author_id": "author_B",
        "author_name": "用戶B",
        "content": f"凌晨發文 {idx + 1}",
        "like_count": 0,
        "reply_count": 0,
        "published_at": base_date + time_str,
        "parent_id": ""
    })

# 3. author_C（含廣告關鍵字）- 2 則留言
rows.append({
    "record_type": "comment",
    "comment_id": "c005",
    "video_id": "v003",
    "author_id": "author_C",
    "author_name": "用戶C",
    "content": "快來訂閱我的頻道！",
    "like_count": 1,
    "reply_count": 2,
    "published_at": "2026-05-28T11:00:00Z",
    "parent_id": ""
})

rows.append({
    "record_type": "comment",
    "comment_id": "c006",
    "video_id": "v003",
    "author_id": "author_C",
    "author_name": "用戶C",
    "content": "加LINE領禮物：xxx",
    "like_count": 2,
    "reply_count": 1,
    "published_at": "2026-05-28T12:00:00Z",
    "parent_id": ""
})

# 4. author_D（含 URL）- 2 則留言
rows.append({
    "record_type": "comment",
    "comment_id": "c007",
    "video_id": "v004",
    "author_id": "author_D",
    "author_name": "用戶D",
    "content": "查看我的網站 https://example.com/abc",
    "like_count": 0,
    "reply_count": 0,
    "published_at": "2026-05-28T13:00:00Z",
    "parent_id": ""
})

rows.append({
    "record_type": "comment",
    "comment_id": "c008",
    "video_id": "v004",
    "author_id": "author_D",
    "author_name": "用戶D",
    "content": "更多資訊在 https://example.com/details",
    "like_count": 1,
    "reply_count": 0,
    "published_at": "2026-05-28T14:00:00Z",
    "parent_id": ""
})

# 5. author_E（同一小時 3+ 則 — 測試 max_burst）- 4 則留言
base_hour = "2026-05-28T15:"
for idx in range(4):
    rows.append({
        "record_type": "comment",
        "comment_id": f"c_0{9 + idx}",
        "video_id": "v005",
        "author_id": "author_E",
        "author_name": "用戶E",
        "content": f"同小時發文 {idx + 1}",
        "like_count": 2 if idx % 2 == 0 else 0,
        "reply_count": 0,
        "published_at": base_hour + f"{idx * 15:02d}:00Z",
        "parent_id": ""
    })

# 6. author_F（reply 型別）- 3 則
rows.append({
    "record_type": "reply",
    "comment_id": "c013",
    "video_id": "v006",
    "author_id": "author_F",
    "author_name": "用戶F",
    "content": "對上面評論的回覆",
    "like_count": 1,
    "reply_count": 0,
    "published_at": "2026-05-28T16:00:00Z",
    "parent_id": "c_parent_001"
})

rows.append({
    "record_type": "reply",
    "comment_id": "c014",
    "video_id": "v006",
    "author_id": "author_F",
    "author_name": "用戶F",
    "content": "再回覆一次",
    "like_count": 0,
    "reply_count": 0,
    "published_at": "2026-05-28T16:30:00Z",
    "parent_id": "c_parent_002"
})

rows.append({
    "record_type": "reply",
    "comment_id": "c015",
    "video_id": "v006",
    "author_id": "author_F",
    "author_name": "用戶F",
    "content": "最後一個回覆",
    "like_count": 2,
    "reply_count": 0,
    "published_at": "2026-05-28T17:00:00Z",
    "parent_id": "c_parent_003"
})

# 7. author_G（零按讚）- 3 則
for idx in range(3):
    rows.append({
        "record_type": "comment",
        "comment_id": f"c_{16 + idx:02d}",
        "video_id": "v007",
        "author_id": "author_G",
        "author_name": "用戶G",
        "content": f"無人點讚的留言 {idx + 1}",
        "like_count": 0,
        "reply_count": 0,
        "published_at": f"2026-05-28T{18 + idx}:00:00Z",
        "parent_id": ""
    })

# 8. author_H ~ author_Z（填充至 30 筆）- 需要 30 - 23 = 7 筆
# 目前已有 1+3+2+2+4+3+3 = 18 筆，還需 12 筆
additional_authors = [
    ("author_H", "用戶H", ["好影片", "推推"]),
    ("author_I", "用戶I", ["讚同", "同意"]),
    ("author_J", "用戶J", ["這很重要", "值得看"]),
    ("author_K", "用戶K", ["感謝分享", "學到了"]),
    ("author_L", "用戶L", ["很棒的內容", "繼續加油"]),
    ("author_M", "用戶M", ["不錯", "推薦"]),
]

content_list = [
    "這個視頻真不錯",
    "完全同意你的看法",
    "很有啟發性",
    "謝謝分享這個資訊",
    "希望看到更多類似內容",
    "辛苦了",
    "製作很用心",
    "這是我的想法",
    "說得太好",
    "值得深思"
]

hour = 19
content_idx = 0
for auth_id, auth_name, _ in additional_authors:
    num_comments = 2 if auth_id != "author_H" else 3
    for i in range(num_comments):
        rows.append({
            "record_type": "comment",
            "comment_id": f"c_{19 + len(rows):02d}",
            "video_id": f"v{8 + len(rows) % 5}",
            "author_id": auth_id,
            "author_name": auth_name,
            "content": content_list[content_idx % len(content_list)],
            "like_count": (len(rows) + i) % 5,
            "reply_count": (len(rows) + i) % 3,
            "published_at": f"2026-05-28T{(hour + len(rows) // 6) % 24:02d}:{(i * 20) % 60:02d}:00Z",
            "parent_id": ""
        })
        content_idx += 1

# 確保至少 30 筆
while len(rows) < 30:
    rows.append({
        "record_type": "comment",
        "comment_id": f"c_filler_{len(rows):02d}",
        "video_id": f"v{(len(rows) % 10) + 1}",
        "author_id": f"author_filler_{len(rows):02d}",
        "author_name": f"填充用戶{len(rows)}",
        "content": f"填充留言 {len(rows)}",
        "like_count": len(rows) % 7,
        "reply_count": len(rows) % 3,
        "published_at": f"2026-05-28T{(20 + len(rows) // 10) % 24:02d}:{(len(rows) * 5) % 60:02d}:00Z",
        "parent_id": ""
    })

# 寫入 CSV
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "record_type", "comment_id", "video_id", "author_id", "author_name",
        "content", "like_count", "reply_count", "published_at", "parent_id"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {output_path}")
print(f"Total: {len(rows)} comments")
print(f"Boundary conditions:")
print(f"  - author_A (1): OK")
print(f"  - author_B (night, 3): OK")
print(f"  - author_C (ads, 2): OK")
print(f"  - author_D (URL, 2): OK")
print(f"  - author_E (burst, 4): OK")
print(f"  - author_F (reply, 3): OK")
print(f"  - author_G (zero likes, 3): OK")
print(f"  - padding: {len(rows) - 18} OK")
