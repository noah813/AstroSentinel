"""生成 sample_labeled.csv - 50+ 位用戶的特徵及標籤"""
import csv
import os
from random import Random

# 使用固定種子確保可重現
rng = Random(42)

output_path = os.path.join(os.path.dirname(__file__), "sample_labeled.csv")

rows = []

# 定義特徵欄位
feature_cols = [
    "total_comments", "unique_videos", "concentration",
    "avg_likes", "zero_like_ratio", "max_burst",
    "avg_length", "unique_content_ratio", "self_similarity",
    "has_ad_keywords", "url_ratio", "night_ratio", "interval_std"
]

# 1. 水軍樣本（is_bot=1）- 至少 5 筆
# 特徵：高 concentration (>5), 高 self_similarity (>0.7), 高 zero_like_ratio (>0.8), 低 interval_std (<60)
bots = [
    {
        "author_id": "bot_001",
        "total_comments": 15,
        "unique_videos": 2,
        "concentration": 8.5,
        "avg_likes": 0.2,
        "zero_like_ratio": 0.85,
        "max_burst": 6,
        "avg_length": 25,
        "unique_content_ratio": 0.15,
        "self_similarity": 0.82,
        "has_ad_keywords": 1,
        "url_ratio": 0.3,
        "night_ratio": 0.4,
        "interval_std": 45.0,
        "is_bot": 1
    },
    {
        "author_id": "bot_002",
        "total_comments": 20,
        "unique_videos": 1,
        "concentration": 12.0,
        "avg_likes": 0.1,
        "zero_like_ratio": 0.9,
        "max_burst": 8,
        "avg_length": 20,
        "unique_content_ratio": 0.1,
        "self_similarity": 0.88,
        "has_ad_keywords": 1,
        "url_ratio": 0.25,
        "night_ratio": 0.6,
        "interval_std": 30.0,
        "is_bot": 1
    },
    {
        "author_id": "bot_003",
        "total_comments": 18,
        "unique_videos": 3,
        "concentration": 6.5,
        "avg_likes": 0.3,
        "zero_like_ratio": 0.83,
        "max_burst": 5,
        "avg_length": 30,
        "unique_content_ratio": 0.18,
        "self_similarity": 0.75,
        "has_ad_keywords": 1,
        "url_ratio": 0.4,
        "night_ratio": 0.3,
        "interval_std": 55.0,
        "is_bot": 1
    },
    {
        "author_id": "bot_004",
        "total_comments": 25,
        "unique_videos": 2,
        "concentration": 10.0,
        "avg_likes": 0.0,
        "zero_like_ratio": 1.0,
        "max_burst": 7,
        "avg_length": 18,
        "unique_content_ratio": 0.12,
        "self_similarity": 0.9,
        "has_ad_keywords": 1,
        "url_ratio": 0.2,
        "night_ratio": 0.5,
        "interval_std": 25.0,
        "is_bot": 1
    },
    {
        "author_id": "bot_005",
        "total_comments": 16,
        "unique_videos": 2,
        "concentration": 7.2,
        "avg_likes": 0.15,
        "zero_like_ratio": 0.88,
        "max_burst": 6,
        "avg_length": 22,
        "unique_content_ratio": 0.14,
        "self_similarity": 0.79,
        "has_ad_keywords": 1,
        "url_ratio": 0.35,
        "night_ratio": 0.4,
        "interval_std": 50.0,
        "is_bot": 1
    },
]

# 2. 正常樣本（is_bot=0）- 至少 5 筆
# 特徵：unique_videos ≥ 3, self_similarity < 0.3, zero_like_ratio < 0.5
normal = [
    {
        "author_id": "user_001",
        "total_comments": 8,
        "unique_videos": 4,
        "concentration": 1.5,
        "avg_likes": 2.5,
        "zero_like_ratio": 0.25,
        "max_burst": 2,
        "avg_length": 45,
        "unique_content_ratio": 0.85,
        "self_similarity": 0.15,
        "has_ad_keywords": 0,
        "url_ratio": 0.1,
        "night_ratio": 0.2,
        "interval_std": 150.0,
        "is_bot": 0
    },
    {
        "author_id": "user_002",
        "total_comments": 10,
        "unique_videos": 5,
        "concentration": 1.2,
        "avg_likes": 1.8,
        "zero_like_ratio": 0.3,
        "max_burst": 2,
        "avg_length": 55,
        "unique_content_ratio": 0.9,
        "self_similarity": 0.18,
        "has_ad_keywords": 0,
        "url_ratio": 0.0,
        "night_ratio": 0.1,
        "interval_std": 200.0,
        "is_bot": 0
    },
    {
        "author_id": "user_003",
        "total_comments": 12,
        "unique_videos": 6,
        "concentration": 1.0,
        "avg_likes": 3.0,
        "zero_like_ratio": 0.2,
        "max_burst": 2,
        "avg_length": 60,
        "unique_content_ratio": 0.92,
        "self_similarity": 0.12,
        "has_ad_keywords": 0,
        "url_ratio": 0.05,
        "night_ratio": 0.15,
        "interval_std": 250.0,
        "is_bot": 0
    },
    {
        "author_id": "user_004",
        "total_comments": 7,
        "unique_videos": 3,
        "concentration": 1.8,
        "avg_likes": 2.2,
        "zero_like_ratio": 0.4,
        "max_burst": 2,
        "avg_length": 50,
        "unique_content_ratio": 0.88,
        "self_similarity": 0.22,
        "has_ad_keywords": 0,
        "url_ratio": 0.1,
        "night_ratio": 0.25,
        "interval_std": 180.0,
        "is_bot": 0
    },
    {
        "author_id": "user_005",
        "total_comments": 9,
        "unique_videos": 4,
        "concentration": 1.3,
        "avg_likes": 1.5,
        "zero_like_ratio": 0.35,
        "max_burst": 2,
        "avg_length": 48,
        "unique_content_ratio": 0.87,
        "self_similarity": 0.2,
        "has_ad_keywords": 0,
        "url_ratio": 0.08,
        "night_ratio": 0.18,
        "interval_std": 190.0,
        "is_bot": 0
    },
]

rows.extend(bots)
rows.extend(normal)

# 3. 隨機填充至 50 位用戶
# 已有 10 筆，需要 40 筆
for i in range(6, 51):
    author_id = f"user_{i:03d}"
    is_bot = rng.choice([0, 1])

    if is_bot == 1:
        # 水軍特徵
        total_comments = rng.randint(12, 30)
        unique_videos = rng.randint(1, 3)
        concentration = rng.uniform(6.0, 12.0)
        avg_likes = rng.uniform(0.0, 0.5)
        zero_like_ratio = rng.uniform(0.75, 0.95)
        max_burst = rng.randint(4, 8)
        avg_length = rng.randint(15, 35)
        unique_content_ratio = rng.uniform(0.1, 0.25)
        self_similarity = rng.uniform(0.7, 0.92)
        has_ad_keywords = rng.choice([0, 1])
        url_ratio = rng.uniform(0.15, 0.5)
        night_ratio = rng.uniform(0.25, 0.7)
        interval_std = rng.uniform(20.0, 60.0)
    else:
        # 正常用戶特徵
        total_comments = rng.randint(5, 15)
        unique_videos = rng.randint(3, 8)
        concentration = rng.uniform(1.0, 2.5)
        avg_likes = rng.uniform(1.0, 4.0)
        zero_like_ratio = rng.uniform(0.1, 0.5)
        max_burst = rng.randint(1, 3)
        avg_length = rng.randint(40, 70)
        unique_content_ratio = rng.uniform(0.75, 0.95)
        self_similarity = rng.uniform(0.08, 0.3)
        has_ad_keywords = rng.choice([0, 1])
        url_ratio = rng.uniform(0.0, 0.15)
        night_ratio = rng.uniform(0.1, 0.3)
        interval_std = rng.uniform(100.0, 300.0)

    rows.append({
        "author_id": author_id,
        "total_comments": total_comments,
        "unique_videos": unique_videos,
        "concentration": round(concentration, 2),
        "avg_likes": round(avg_likes, 2),
        "zero_like_ratio": round(zero_like_ratio, 2),
        "max_burst": max_burst,
        "avg_length": round(avg_length, 2),
        "unique_content_ratio": round(unique_content_ratio, 2),
        "self_similarity": round(self_similarity, 2),
        "has_ad_keywords": has_ad_keywords,
        "url_ratio": round(url_ratio, 2),
        "night_ratio": round(night_ratio, 2),
        "interval_std": round(interval_std, 2),
        "is_bot": is_bot
    })

# 寫入 CSV
fieldnames = ["author_id"] + feature_cols + ["is_bot"]
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# 統計
bot_count = sum(1 for r in rows if r["is_bot"] == 1)
normal_count = sum(1 for r in rows if r["is_bot"] == 0)

print(f"Generated {output_path}")
print(f"Total users: {len(rows)}")
print(f"Bots: {bot_count}")
print(f"Normal: {normal_count}")
print(f"Unique authors: {len(set(r['author_id'] for r in rows))}")
