import os
import sys
import pandas as pd
import numpy as np
import torch
from opencc import OpenCC
from transformers import pipeline
import config
from feature_engineering import load_raw_comments

def main():
    pending_path = "data/pending_review.csv"
    if not os.path.exists(pending_path):
        print(f"Error: {pending_path} not found.")
        sys.exit(1)
        
    print("Loading pending users...")
    pending_df = pd.read_csv(pending_path, dtype={"is_bot": str}, keep_default_na=False)
    
    # Only process rows where is_bot is empty, limit to 100 for demonstration/speed
    to_label_all = pending_df[pending_df["is_bot"] == ""]
    limit = 10000
    to_label = to_label_all.head(limit).copy()
    if to_label.empty:
        print("No pending users to label.")
        return
        
    print(f"Found {len(to_label_all)} total pending users. Processing the first {len(to_label)} users (limit={limit}).")
    
    print("Loading raw comments to extract text content...")
    try:
        comments_df = load_raw_comments(config.DATA_DIR, config.LOOKBACK_DAYS)
    except Exception as e:
        print(f"Error loading raw comments: {e}")
        sys.exit(1)
        
    # Set up OpenCC for Traditional to Simplified Chinese conversion
    print("Initializing OpenCC (Traditional to Simplified conversion)...")
    cc = OpenCC('t2s')
    
    # Initialize zero-shot pipeline
    print("Initializing zero-shot classification model (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)...")
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("CUDA available! Running on GPU.")
    else:
        print("CUDA not available. Running on CPU (this might take a while for large datasets).")
        
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=device
        )
    except Exception as e:
        print(f"Failed to load zero-shot classification model: {e}")
        sys.exit(1)
        
    candidate_labels = ["正常評論", "網路水軍或垃圾廣告"]
    hypothesis_template = "這段文本關於 {}。"
    
    # Group comments by author_id
    comments_by_user = comments_df.groupby("author_id")["content"].apply(list).to_dict()
    
    # Map author_id to predicted label
    user_labels = {}
    
    print("Start zero-shot labeling process...")
    # Process each user
    for idx, row in to_label.iterrows():
        author_id = row["author_id"]
        user_comments = comments_by_user.get(author_id, [])
        
        # If no comments found, default to normal
        if not user_comments:
            user_labels[author_id] = 0
            continue
            
        # Clean list of comments (remove NaN or non-string values)
        user_comments = [str(c) for c in user_comments if pd.notna(c) and str(c).strip()]
        if not user_comments:
            user_labels[author_id] = 0
            continue
            
        # Sample up to 3 comments to save computation time
        sampled_comments = user_comments[:3]
        
        # Convert Traditional to Simplified Chinese
        simplified_comments = [cc.convert(c) for c in sampled_comments]
        
        # Predict zero-shot scores
        scores = []
        try:
            predictions = classifier(simplified_comments, candidate_labels=candidate_labels, multi_label=False, hypothesis_template=hypothesis_template)
            # Handle both single string output and list output from pipeline
            if not isinstance(predictions, list):
                predictions = [predictions]
                
            for pred in predictions:
                # Find the score of the "網路水軍或垃圾廣告" label
                label_idx = pred["labels"].index("網路水軍或垃圾廣告")
                scores.append(pred["scores"][label_idx])
        except Exception as e:
            # Fallback in case of pipeline errors
            scores = [0.0]
            
        # Average score threshold: if >= 0.5, label as 1 (bot), else 0 (normal)
        avg_score = np.mean(scores)
        user_labels[author_id] = 1 if avg_score >= 0.5 else 0
        
        # Print progress info every 10 users
        if len(user_labels) % 10 == 0 or len(user_labels) == len(to_label):
            print(f"Labeled {len(user_labels)}/{len(to_label)} users...")
            
    # Update pending_df with predictions
    for idx, row in pending_df.iterrows():
        author_id = row["author_id"]
        if row["is_bot"] == "" and author_id in user_labels:
            pending_df.at[idx, "is_bot"] = str(user_labels[author_id])
            
    # Write back to file
    pending_df.to_csv(pending_path, index=False, quoting=1)
    
    num_bots = sum(1 for val in user_labels.values() if val == 1)
    num_normals = sum(1 for val in user_labels.values() if val == 0)
    print(f"\nLabeling completed! Automatically labeled {len(user_labels)} users.")
    print(f"Detected Bots: {num_bots}")
    print(f"Detected Normals: {num_normals}")
    print(f"Results saved to {pending_path}.")

if __name__ == "__main__":
    main()
