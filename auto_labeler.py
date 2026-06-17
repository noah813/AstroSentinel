import os
import pandas as pd
import sys
import time
import config

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-genai package is required. Run: uv add google-genai")
    sys.exit(1)


def classify_user_llm(row: pd.Series, client: genai.Client) -> str:
    """
    Use Gemini to classify a user based on feature statistics.
    Returns 'bot', 'normal', or 'pending'.
    """
    prompt = f"""
    You are an expert at detecting YouTube astroturfing (water army) and bot behavior.
    Classify the following user based on their behavioral, content, and temporal features.
    
    User Features:
    - total_comments (7 days): {row['total_comments']}
    - unique_videos: {row['unique_videos']}
    - concentration (comments per video): {row['concentration']}
    - avg_likes: {row['avg_likes']}
    - zero_like_ratio: {row['zero_like_ratio']}
    - max_burst (max comments in a single hour): {row['max_burst']}
    - avg_length (average comment length): {row['avg_length']}
    - unique_content_ratio: {row['unique_content_ratio']}
    - self_similarity (text similarity between comments): {row['self_similarity']}
    - has_ad_keywords: {row['has_ad_keywords']}
    - url_ratio: {row['url_ratio']}
    - night_ratio (comments made between 0-6 AM UTC): {row['night_ratio']}
    - interval_std (standard deviation of time between comments): {row['interval_std']}
    
    Return exactly one word: "bot", "normal", or "pending".
    "bot": clearly exhibits bot/spam/astroturfing behavior.
    "normal": clearly behaves like a normal human user.
    "pending": unsure and needs manual review.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )
        ans = response.text.strip().lower()
        if "bot" in ans: return "bot"
        if "normal" in ans: return "normal"
        if "pending" in ans: return "pending"
        return "pending"
    except Exception as e:
        print(f"Error classifying user {row['author_id']}: {e}")
        time.sleep(1) # Backoff a bit
        return "pending"


def run_labeling(
    features_path: str = "data/user_features.csv",
    labeled_output: str = "data/labeled_data.csv",
    pending_output: str = "data/pending_review.csv",
) -> dict:
    """
    Read user features CSV, classify each user using Gemini (skipping already labeled ones),
    and update the output CSVs incrementally.
    """
    # Read features CSV
    df = pd.read_csv(features_path)

    # Validate required columns
    required_cols = set(config.FEATURE_COLS + ["author_id"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"user_features.csv missing columns: {missing}")

    # Load existing labeled data to avoid re-labeling
    existing_bot_ids = set()
    existing_normal_ids = set()
    if os.path.exists(labeled_output):
        try:
            old_labeled = pd.read_csv(labeled_output)
            if "author_id" in old_labeled.columns and "is_bot" in old_labeled.columns:
                existing_bot_ids = set(old_labeled[old_labeled["is_bot"] == 1]["author_id"])
                existing_normal_ids = set(old_labeled[old_labeled["is_bot"] == 0]["author_id"])
        except Exception as e:
            print(f"Warning: Failed to read existing {labeled_output}: {e}")
            
    # Load existing pending data
    existing_pending_ids = set()
    if os.path.exists(pending_output):
        try:
            old_pending = pd.read_csv(pending_output)
            if "author_id" in old_pending.columns:
                existing_pending_ids = set(old_pending["author_id"])
        except Exception as e:
            print(f"Warning: Failed to read existing {pending_output}: {e}")

    api_key = os.environ.get("GEMINI_API_KEY")
    client = None
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Falling back to marking new users as pending...")
    else:
        client = genai.Client(api_key=api_key)

    labels = []
    processed_count = 0
    for idx, row in df.iterrows():
        author_id = row["author_id"]
        
        # Check if already labeled
        if author_id in existing_bot_ids:
            labels.append("bot")
            continue
        if author_id in existing_normal_ids:
            labels.append("normal")
            continue
        if author_id in existing_pending_ids:
            labels.append("pending")
            continue
            
        # New user needing label
        if not client:
            labels.append("pending")
            continue
            
        processed_count += 1
        if processed_count > 1 and processed_count % 20 == 0:
            print(f"Queried Gemini for {processed_count} new users...")
            time.sleep(0.5) # simple rate limit avoidance
            
        label = classify_user_llm(row, client)
        labels.append(label)
        
    df["label"] = labels
    
    if client:
        print(f"Successfully labeled {processed_count} NEW users with Gemini.")

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
        "newly_labeled": processed_count
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Make sure to load the .env if not already loaded
    
    stats = run_labeling()
    print(f"[auto_labeler] bot={stats['bot']}, normal={stats['normal']}, pending={stats['pending']}")
