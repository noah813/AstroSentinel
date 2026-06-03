import os
import glob
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config


def load_raw_comments(data_dir: str, lookback_days: int) -> pd.DataFrame:
    """讀取所有 youtube_raw_*.csv，篩選留言/回覆，去重，過濾時間窗口"""
    pattern = os.path.join(data_dir, "collected", "youtube_raw_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"找不到任何 CSV 檔案，路徑模式：{pattern}")

    dfs = []
    for f in files:
        try:
            df_part = pd.read_csv(f, low_memory=False)
            dfs.append(df_part)
        except Exception as e:
            warnings.warn(f"讀取 {f} 失敗：{e}")

    if not dfs:
        raise ValueError("所有 CSV 檔案均讀取失敗。")

    df = pd.concat(dfs, ignore_index=True)

    # 篩選留言與回覆（排除 video 列）
    df = df[df["record_type"].isin(["comment", "reply"])].copy()

    # 轉換 published_at 為 UTC-aware datetime
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

    # 依 comment_id 去重（保留第一筆）
    df = df.drop_duplicates(subset=["comment_id"])

    # 過濾時間窗口
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=lookback_days)
    filtered = df[df["published_at"] >= cutoff].copy()

    if len(filtered) == 0:
        warnings.warn(
            f"過濾 lookback_days={lookback_days} 後無留言資料，改用全部留言（共 {len(df)} 筆）。",
            UserWarning,
        )
        return df.reset_index(drop=True)

    return filtered.reset_index(drop=True)


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算 total_comments, unique_videos, concentration, avg_likes, zero_like_ratio, max_burst"""
    # 確保 like_count 為數值
    df = df.copy()
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

    grouped = df.groupby("author_id")

    total_comments = grouped.size().rename("total_comments")
    unique_videos = grouped["video_id"].nunique().rename("unique_videos")
    avg_likes = grouped["like_count"].mean().rename("avg_likes")
    zero_like_ratio = grouped["like_count"].apply(
        lambda x: (x == 0).sum() / len(x)
    ).rename("zero_like_ratio")

    # max_burst：同作者同小時內最大留言數
    df["hour_bucket"] = df["published_at"].dt.floor("h")
    burst_counts = (
        df.groupby(["author_id", "hour_bucket"])
        .size()
        .groupby(level=0)
        .max()
        .rename("max_burst")
    )

    result = pd.concat(
        [total_comments, unique_videos, avg_likes, zero_like_ratio, burst_counts],
        axis=1,
    )

    # concentration = total_comments / unique_videos（unique_videos=0 時填 0）
    result["concentration"] = np.where(
        result["unique_videos"] == 0,
        0,
        result["total_comments"] / result["unique_videos"],
    )

    return result[
        ["total_comments", "unique_videos", "concentration",
         "avg_likes", "zero_like_ratio", "max_burst"]
    ]


def _self_sim(contents):
    """計算一組留言的 TF-IDF 字符 n-gram 平均 cosine 相似度"""
    if len(contents) < 2:
        return 0.0
    try:
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        mat = tfidf.fit_transform(contents)
        sim_mat = cosine_similarity(mat)
        n = sim_mat.shape[0]
        upper_tri = sim_mat[np.triu_indices(n, k=1)]
        return float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
    except Exception:
        return 0.0


def compute_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算 avg_length, unique_content_ratio, self_similarity, has_ad_keywords, url_ratio"""
    df = df.copy()
    df["content"] = df["content"].fillna("").astype(str)

    grouped = df.groupby("author_id")

    avg_length = grouped["content"].apply(
        lambda x: np.mean([len(c) for c in x])
    ).rename("avg_length")

    unique_content_ratio = grouped["content"].apply(
        lambda x: len(x.unique()) / len(x)
    ).rename("unique_content_ratio")

    self_similarity = grouped["content"].apply(
        lambda x: _self_sim(x.tolist())
    ).rename("self_similarity")

    def _has_ad(contents):
        for content in contents:
            for kw in config.AD_KEYWORDS:
                if kw in content:
                    return 1
        return 0

    has_ad_keywords = grouped["content"].apply(
        lambda x: _has_ad(x.tolist())
    ).rename("has_ad_keywords")

    def _url_ratio(contents):
        url_markers = ["http://", "https://", "www."]
        count = sum(
            any(u in c for u in url_markers) for c in contents
        )
        return count / len(contents) if len(contents) > 0 else 0.0

    url_ratio = grouped["content"].apply(
        lambda x: _url_ratio(x.tolist())
    ).rename("url_ratio")

    result = pd.concat(
        [avg_length, unique_content_ratio, self_similarity, has_ad_keywords, url_ratio],
        axis=1,
    )
    return result


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算 night_ratio, interval_std"""
    df = df.copy()

    grouped = df.groupby("author_id")

    # night_ratio：UTC 時間 0~5 點的比例
    night_start, night_end = config.NIGHT_HOURS  # (0, 6)
    night_range = list(range(night_start, night_end))

    night_ratio = grouped["published_at"].apply(
        lambda x: x.dt.hour.isin(night_range).mean()
    ).rename("night_ratio")

    # interval_std：時間間隔（秒）的標準差，留言數 < 2 時填 0.0
    def _interval_std(times):
        sorted_times = times.sort_values()
        if len(sorted_times) < 2:
            return 0.0
        diffs = sorted_times.diff().dropna().dt.total_seconds()
        return float(diffs.std()) if len(diffs) > 0 else 0.0

    interval_std = grouped["published_at"].apply(
        lambda x: _interval_std(x)
    ).rename("interval_std")

    result = pd.concat([night_ratio, interval_std], axis=1)
    return result


def run_feature_engineering(
    data_dir: str = config.DATA_DIR,
    output_path: str = os.path.join(config.DATA_DIR, "user_features.csv"),
    lookback_days: int = config.LOOKBACK_DAYS,
) -> None:
    """主執行函式"""
    import sys
    def _print(msg):
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()

    _print(f"[feature_engineering] data_dir={data_dir}, lookback_days={lookback_days}")
    df = load_raw_comments(data_dir, lookback_days)
    _print(f"[feature_engineering] loaded {len(df)} comments, {df['author_id'].nunique()} users")

    behavioral = compute_behavioral_features(df)
    content = compute_content_features(df)
    temporal = compute_temporal_features(df)

    # 合併三組特徵，以 author_id 為鍵
    features = behavioral.join(content, how="outer").join(temporal, how="outer")

    # 填充 NaN 為 0
    features = features.fillna(0)

    # 確保欄位順序符合 config.FEATURE_COLS
    features = features[config.FEATURE_COLS]

    # 寫入 CSV（reset_index 使 author_id 成為一般欄位）
    features = features.reset_index()
    features.rename(columns={"index": "author_id"}, inplace=True)

    # 確保 author_id 欄位名稱正確
    if "author_id" not in features.columns and features.columns[0] != "author_id":
        features.columns = ["author_id"] + list(features.columns[1:])

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    features.to_csv(output_path, index=False)
    _print(f"[feature_engineering] written to {output_path} ({len(features)} users, {len(features.columns)} cols)")


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering complete.")
