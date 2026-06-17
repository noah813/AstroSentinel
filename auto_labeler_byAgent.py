import os
import pandas as pd
import sys
import time
import json
import config

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
except ImportError:
    print("google-genai and pydantic packages are required. Run: uv add google-genai pydantic")
    sys.exit(1)

# Define Pydantic models for structured output
class UserLabel(BaseModel):
    author_id: str
    label: str  # 'bot', 'normal', or 'pending'

class BatchLabelResponse(BaseModel):
    results: list[UserLabel]


def classify_users_batch_llm(rows: list[pd.Series], client: genai.Client) -> dict[str, str]:
    """
    Use Gemini to classify a batch of users based on their features.
    Returns a dictionary mapping author_id -> label.
    """
    # Prepare the batch data as a JSON string
    batch_data = []
    for row in rows:
        batch_data.append({
            "author_id": row["author_id"],
            "total_comments": row['total_comments'],
            "unique_videos": row['unique_videos'],
            "concentration": round(row['concentration'], 2),
            "avg_likes": round(row['avg_likes'], 2),
            "zero_like_ratio": round(row['zero_like_ratio'], 2),
            "max_burst": row['max_burst'],
            "avg_length": round(row['avg_length'], 2),
            "unique_content_ratio": round(row['unique_content_ratio'], 2),
            "self_similarity": round(row['self_similarity'], 2),
            "has_ad_keywords": row['has_ad_keywords'],
            "url_ratio": round(row['url_ratio'], 2),
            "night_ratio": round(row['night_ratio'], 2),
            "interval_std": round(row['interval_std'], 2)
        })

    prompt = f"""
    You are an expert at detecting YouTube astroturfing (water army) and bot behavior.
    Classify the following batch of users based on their behavioral, content, and temporal features.
    
    For each user, return exactly one label: "bot", "normal", or "pending".
    "bot": clearly exhibits bot/spam/astroturfing behavior.
    "normal": clearly behaves like a normal human user.
    "pending": unsure and needs manual review.
    
    Users data:
    {json.dumps(batch_data, indent=2, ensure_ascii=False)}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=BatchLabelResponse,
            ),
        )
        
        # Parse the structured JSON response
        result_dict = {}
        try:
            response_data = json.loads(response.text)
            for item in response_data.get("results", []):
                uid = item.get("author_id")
                lbl = item.get("label", "pending").lower()
                if lbl not in ["bot", "normal", "pending"]:
                    lbl = "pending"
                result_dict[uid] = lbl
        except Exception as parse_e:
            print(f"Failed to parse JSON response: {parse_e}")
            for row in rows:
                result_dict[row["author_id"]] = "pending"
                
        return result_dict

    except Exception as e:
        print(f"Error classifying batch: {e}")
        time.sleep(2) # Backoff
        # Return pending for all in this batch on error
        return {row["author_id"]: "pending" for row in rows}


def run_batch_labeling(
    features_path: str = "data/user_features.csv",
    labeled_output: str = "data/labeled_data.csv",
    pending_output: str = "data/pending_review.csv",
    batch_size: int = 80
) -> dict:
    """
    Read user features CSV, classify pending users using Gemini in batches,
    and update the output CSVs incrementally.
    """
    df = pd.read_csv(features_path)
    required_cols = set(config.FEATURE_COLS + ["author_id"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"user_features.csv missing columns: {missing}")

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
            
    existing_pending_ids = set()
    if os.path.exists(pending_output):
        try:
            old_pending = pd.read_csv(pending_output, dtype={"is_bot": str}, keep_default_na=False)
            if "author_id" in old_pending.columns:
                existing_pending_ids = set(old_pending["author_id"])
        except Exception as e:
            print(f"Warning: Failed to read existing {pending_output}: {e}")

    api_key = os.environ.get("GEMINI_API_KEY")
    client = None
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Falling back to marking new users as pending...")
    else:
        print(f"Initializing Gemini Client for BATCH labeling... (batch_size={batch_size})")
        client = genai.Client(api_key=api_key)

    labels = []
    
    # Identify which users need labeling
    rows_to_process = []
    for idx, row in df.iterrows():
        author_id = row["author_id"]
        if author_id in existing_bot_ids:
            labels.append((idx, "bot"))
        elif author_id in existing_normal_ids:
            labels.append((idx, "normal"))
        elif author_id in existing_pending_ids:
            labels.append((idx, "pending"))
        else:
            rows_to_process.append((idx, row))
            
    print(f"Total users: {len(df)}. Already processed: {len(labels)}. Needing batch labeling: {len(rows_to_process)}")

    if client and rows_to_process:
        processed_count = 0
        
        # Process in batches
        for i in range(0, len(rows_to_process), batch_size):
            batch = rows_to_process[i:i+batch_size]
            batch_rows = [item[1] for item in batch]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(rows_to_process)-1)//batch_size + 1} ({len(batch)} users)...")
            
            batch_results = classify_users_batch_llm(batch_rows, client)
            
            for original_idx, row in batch:
                uid = row["author_id"]
                assigned_label = batch_results.get(uid, "pending")
                labels.append((original_idx, assigned_label))
                
            processed_count += len(batch)
            time.sleep(1) # Rate limit avoidance between batches
            
        print(f"Successfully batch-labeled {processed_count} NEW users.")
    else:
        # Mark all as pending if no client
        for original_idx, row in rows_to_process:
            labels.append((original_idx, "pending"))

    # Reconstruct label column in original order
    labels.sort(key=lambda x: x[0])
    df["label"] = [lbl for idx, lbl in labels]

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
    pending_df[pending_cols].to_csv(pending_output, index=False, quoting=1)

    return {
        "bot": bot_count,
        "normal": normal_count,
        "pending": pending_count,
        "newly_labeled": len(rows_to_process) if client else 0
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    stats = run_batch_labeling()
    print(f"[auto_labeler_byAgent] bot={stats['bot']}, normal={stats['normal']}, pending={stats['pending']}, newly_labeled={stats['newly_labeled']}")
