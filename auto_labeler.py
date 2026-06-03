import pandas as pd
import sys
import config


def classify_user(row: pd.Series) -> str:
    """
    Pure function that classifies a user based on feature thresholds.

    Bot conditions (all 4 must be satisfied):
      concentration > 5.0 AND self_similarity > 0.7 AND
      zero_like_ratio > 0.8 AND interval_std < 60.0

    Normal conditions (all 3 must be satisfied):
      unique_videos >= 3 AND self_similarity < 0.3 AND zero_like_ratio < 0.5

    If both bot and normal conditions are met, bot takes priority.
    If neither condition is met, return 'pending'.

    Args:
        row: A pd.Series containing feature columns

    Returns:
        str: 'bot', 'normal', or 'pending'
    """
    # Check bot conditions
    is_bot = (
        row["concentration"] > config.BOT_CONCENTRATION_GT and
        row["self_similarity"] > config.BOT_SELF_SIMILARITY_GT and
        row["zero_like_ratio"] > config.BOT_ZERO_LIKE_RATIO_GT and
        row["interval_std"] < config.BOT_INTERVAL_STD_LT
    )

    # Check normal conditions
    is_normal = (
        row["unique_videos"] >= config.NORMAL_UNIQUE_VIDEOS_GE and
        row["self_similarity"] < config.NORMAL_SELF_SIMILARITY_LT and
        row["zero_like_ratio"] < config.NORMAL_ZERO_LIKE_RATIO_LT
    )

    # Bot takes priority
    if is_bot:
        return "bot"
    if is_normal:
        return "normal"
    return "pending"


def run_labeling(
    features_path: str = "data/user_features.csv",
    labeled_output: str = "data/labeled_data.csv",
    pending_output: str = "data/pending_review.csv",
) -> dict:
    """
    Read user features CSV, classify each user, and write two output CSVs.

    Args:
        features_path: Path to input user_features.csv
        labeled_output: Path to write labeled_data.csv (bot/normal users)
        pending_output: Path to write pending_review.csv (pending users)

    Returns:
        dict: Statistics {"bot": N, "normal": N, "pending": N}

    Raises:
        ValueError: If required feature columns are missing
    """
    # Read features CSV
    df = pd.read_csv(features_path)

    # Validate required columns
    required_cols = set(config.FEATURE_COLS + ["author_id"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"user_features.csv missing columns: {missing}")

    # Classify each user
    df["label"] = df.apply(classify_user, axis=1)

    # Count statistics
    bot_count = (df["label"] == "bot").sum()
    normal_count = (df["label"] == "normal").sum()
    pending_count = (df["label"] == "pending").sum()

    # Prepare labeled data (bot + normal)
    labeled_df = df[df["label"].isin(["bot", "normal"])].copy()
    labeled_df["is_bot"] = (labeled_df["label"] == "bot").astype(int)
    labeled_cols = ["author_id"] + config.FEATURE_COLS + ["is_bot"]
    labeled_df[labeled_cols].to_csv(labeled_output, index=False)

    # Prepare pending data
    pending_df = df[df["label"] == "pending"].copy()
    pending_df["is_bot"] = ""  # Empty string, not NaN
    pending_cols = ["author_id"] + config.FEATURE_COLS + ["is_bot"]
    # Use quoting=1 (QUOTE_ALL) to ensure empty strings are properly written
    pending_df[pending_cols].to_csv(pending_output, index=False, quoting=1)

    return {
        "bot": bot_count,
        "normal": normal_count,
        "pending": pending_count,
    }


if __name__ == "__main__":
    stats = run_labeling()
    print(f"[auto_labeler] bot={stats['bot']}, normal={stats['normal']}, pending={stats['pending']}")
