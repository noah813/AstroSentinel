import os
import glob
import pandas as pd


def generate_report(
    predictions_path: str = "results/pending_review_byAI.csv",
    raw_data_dir: str = "data/collected",
    output_path: str = "results/report.md",
) -> None:
    print("[report] Loading predictions...")
    pred_df = pd.read_csv(predictions_path)
    bots_df = pred_df[pred_df["is_bot"] == 1].copy()
    bot_ids = set(bots_df["author_id"].tolist())
    n_total = len(pred_df)
    n_bots = len(bots_df)
    print(f"[report] {n_bots} bots out of {n_total} users")

    print("[report] Loading raw comment data...")
    pattern = os.path.join(raw_data_dir, "youtube_raw_*.csv")
    files = glob.glob(pattern)
    raw_dfs = [pd.read_csv(f, low_memory=False) for f in files]
    raw_df = pd.concat(raw_dfs, ignore_index=True)

    # 建立 video_id -> video_title 映射表
    video_titles = (
        raw_df[raw_df["record_type"] == "video"]
        .drop_duplicates("video_id")
        .set_index("video_id")["video_title"]
        .to_dict()
    )

    comments_df = raw_df[raw_df["record_type"].isin(["comment", "reply"])].copy()
    comments_df["video_title"] = comments_df["video_id"].map(video_titles)

    # 取得水軍作者名稱（同一 author_id 可能有多筆，取第一個非空值）
    author_names = (
        comments_df[comments_df["author_id"].isin(bot_ids)]
        .groupby("author_id")["author_name"]
        .first()
        .to_dict()
    )

    # 取得水軍留言，按 author_id 分組
    bot_comments = (
        comments_df[comments_df["author_id"].isin(bot_ids)]
        .sort_values(["author_id", "published_at"])
    )

    lines = []

    # ── 標題與摘要 ──────────────────────────────────────
    lines.append("# 水軍偵測報告")
    lines.append("")
    lines.append(f"**分析日期**：2026-06-04")
    lines.append(f"**審查總人數**：{n_total:,}")
    lines.append(f"**水軍人數**：{n_bots:,}（佔 {n_bots/n_total*100:.1f}%）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 水軍特徵分布統計 ────────────────────────────────
    lines.append("## 水軍特徵統計摘要")
    lines.append("")
    lines.append("| 指標 | 平均值 | 最大值 | 最小值 |")
    lines.append("|------|--------|--------|--------|")

    metric_labels = {
        "total_comments":      "總留言數",
        "unique_videos":       "留言影片數",
        "concentration":       "集中度（留言/影片）",
        "avg_likes":           "平均讚數",
        "zero_like_ratio":     "零讚比例",
        "max_burst":           "最大爆發留言數（同小時）",
        "avg_length":          "平均留言長度",
        "unique_content_ratio":"獨特內容比例",
        "self_similarity":     "自我相似度",
        "has_ad_keywords":     "含廣告關鍵字（佔比）",
        "url_ratio":           "含網址比例",
        "night_ratio":         "深夜留言比例（UTC 0–5）",
        "interval_std":        "留言間隔標準差（秒）",
    }
    for col, label in metric_labels.items():
        mean_val = bots_df[col].mean()
        max_val  = bots_df[col].max()
        min_val  = bots_df[col].min()
        lines.append(f"| {label} | {mean_val:.2f} | {max_val:.2f} | {min_val:.2f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 各水軍詳細資料 ──────────────────────────────────
    lines.append("## 水軍詳細資料")
    lines.append("")

    bots_sorted = bots_df.sort_values("total_comments", ascending=False)

    for rank, (_, row) in enumerate(bots_sorted.iterrows(), start=1):
        aid = row["author_id"]
        name = author_names.get(aid, "（無法取得）")

        lines.append(f"### #{rank} {name}")
        lines.append("")
        lines.append(f"- **Author ID**：`{aid}`")
        lines.append(f"- **總留言數**：{int(row['total_comments'])}")
        lines.append(f"- **留言影片數**：{int(row['unique_videos'])}")
        lines.append(f"- **集中度**：{row['concentration']:.2f}")
        lines.append(f"- **平均讚數**：{row['avg_likes']:.2f}")
        lines.append(f"- **零讚比例**：{row['zero_like_ratio']:.2%}")
        lines.append(f"- **最大爆發數（同小時）**：{int(row['max_burst'])}")
        lines.append(f"- **平均留言長度**：{row['avg_length']:.1f} 字元")
        lines.append(f"- **獨特內容比例**：{row['unique_content_ratio']:.2%}")
        lines.append(f"- **自我相似度**：{row['self_similarity']:.4f}")
        lines.append(f"- **含廣告關鍵字**：{'是' if int(row['has_ad_keywords']) else '否'}")
        lines.append(f"- **含網址比例**：{row['url_ratio']:.2%}")
        lines.append(f"- **深夜留言比例**：{row['night_ratio']:.2%}")
        lines.append(f"- **留言間隔標準差**：{row['interval_std']:.1f} 秒")
        lines.append("")

        user_comments = bot_comments[bot_comments["author_id"] == aid]
        if len(user_comments) == 0:
            lines.append("_（此用戶在原始資料中無留言記錄）_")
        else:
            lines.append("**發文內容：**")
            lines.append("")
            lines.append("| # | 影片標題 | 留言內容 | 時間 | 讚數 |")
            lines.append("|---|----------|----------|------|------|")
            for i, (_, c) in enumerate(user_comments.iterrows(), start=1):
                content = str(c.get("content", "")).replace("|", "｜").replace("\n", " ").strip()
                raw_title = c.get("video_title", "")
                video_title = ("" if pd.isna(raw_title) else str(raw_title)).replace("|", "｜")[:40]
                published = str(c.get("published_at", ""))[:16]
                likes = c.get("like_count", 0)
                lines.append(f"| {i} | {video_title} | {content} | {published} | {likes} |")

        lines.append("")
        lines.append("---")
        lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[report] Saved -> {output_path}")


if __name__ == "__main__":
    generate_report()
