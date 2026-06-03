import os
import joblib
import numpy as np
import pandas as pd
import config

FUSION_FEATURE_COLS = ["lr_prob", "xgb_prob", "bert_user_prob", "bert_max_prob", "bert_comment_count"]


def predict_pending(
    pending_path: str = "data/pending_review.csv",
    model_dir: str = "data/models",
    bert_scores_path: str = "data/bert_comment_scores.csv",
    output_path: str = "results/pending_review_byAI.csv",
) -> None:
    print("[predict] Loading pending_review data...")
    pending_df = pd.read_csv(pending_path)
    author_ids = pending_df["author_id"].values
    X = pending_df[config.FEATURE_COLS]

    print("[predict] Loading stage1 models...")
    lr_model = joblib.load(os.path.join(model_dir, "stage1", "lr_model.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "stage1", "xgb_model.pkl"))

    lr_probs = lr_model.predict_proba(X)[:, 1]
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    print("[predict] Aggregating BERT comment scores...")
    scores_df = pd.read_csv(bert_scores_path)
    bert_agg = scores_df.groupby("author_id").agg(
        bert_user_prob=("spam_prob", "mean"),
        bert_max_prob=("spam_prob", "max"),
        bert_comment_count=("spam_prob", "count"),
    )

    bert_user_probs = [
        bert_agg.at[aid, "bert_user_prob"] if aid in bert_agg.index else 0.5
        for aid in author_ids
    ]
    bert_max_probs = [
        bert_agg.at[aid, "bert_max_prob"] if aid in bert_agg.index else 0.5
        for aid in author_ids
    ]
    bert_comment_counts = [
        int(bert_agg.at[aid, "bert_comment_count"]) if aid in bert_agg.index else 0
        for aid in author_ids
    ]

    X_fusion = np.column_stack([lr_probs, xgb_probs, bert_user_probs, bert_max_probs, bert_comment_counts])

    print("[predict] Loading fusion model...")
    fusion_model = joblib.load(os.path.join(model_dir, "stage3", "fusion_model.pkl"))

    print("[predict] Predicting is_bot...")
    predictions = fusion_model.predict(X_fusion)

    result_df = pending_df.copy()
    result_df["is_bot"] = predictions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    n_bot = int(predictions.sum())
    n_total = len(predictions)
    print(f"[predict] Done. {n_bot}/{n_total} users predicted as bot ({n_bot/n_total*100:.1f}%)")
    print(f"[predict] Saved -> {output_path}")


if __name__ == "__main__":
    predict_pending()
