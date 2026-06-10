import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import config

def main():
    print("Starting visualization generation...")
    os.makedirs("results", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # ── 1. Model Comparison Chart ──────────────────────────────────────
    print("Generating: model_comparison.png")
    try:
        # Load stage1/2 comparison report
        df_comp = pd.read_csv("results/comparison_report.csv")
        
        # Load stage3 (fusion) report
        with open("results/stage3_report.json", "r", encoding="utf-8") as f:
            stage3 = json.load(f)
            
        # Add Fusion to comparison dataframe
        fusion_row = pd.DataFrame([{
            "model": "fusion_stacking",
            "precision": stage3["precision"],
            "recall": stage3["recall"],
            "f1": stage3["f1"],
            "auc_roc": stage3["auc_roc"]
        }])
        df_all = pd.concat([df_comp, fusion_row], ignore_index=True)
        
        # Melt dataframe for seaborn plotting
        df_melt = df_all.melt(id_vars="model", var_name="Metric", value_name="Value")
        
        # Format model names for display
        model_name_map = {
            "logistic_regression": "Logistic Regression",
            "xgboost": "XGBoost",
            "bert_aggregated": "BERT (Fine-tuned)",
            "fusion_stacking": "Fusion (Stacking)"
        }
        df_melt["model"] = df_melt["model"].map(model_name_map)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_melt, x="model", y="Value", hue="Metric", palette="muted")
        plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Model Stage / Architecture", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(loc="lower right")
        
        # Add value labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            xytext=(0, 5),
                            textcoords='offset points',
                            fontsize=9)
                            
        plt.tight_layout()
        plt.savefig("results/model_comparison.png", dpi=300)
        plt.close()
        print("-> Successfully saved model_comparison.png")
    except Exception as e:
        print(f"Error generating model comparison plot: {e}")
        
    # ── 2. Feature Importance Chart ────────────────────────────────────
    print("Generating: feature_importance.png")
    try:
        model_path = "data/models/stage1/xgb_model.pkl"
        if os.path.exists(model_path):
            xgb_model = joblib.load(model_path)
            importances = xgb_model.feature_importances_
            # Use feature_names_in_ if available, fallback to config.FEATURE_COLS
            feature_names = getattr(xgb_model, "feature_names_in_", config.FEATURE_COLS)
            
            feat_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis")
            plt.title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
            plt.xlabel("Relative Importance Score", fontsize=12)
            plt.ylabel("Behavioral / Text Feature", fontsize=12)
            plt.tight_layout()
            plt.savefig("results/feature_importance.png", dpi=300)
            plt.close()
            print("-> Successfully saved feature_importance.png")
        else:
            print("XGBoost model file not found at data/models/stage1/xgb_model.pkl, skipping.")
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        
    # Load labeled data for distribution plots
    labeled_path = "data/labeled_data.csv"
    if os.path.exists(labeled_path):
        df_lbl = pd.read_csv(labeled_path)
        df_lbl["User Type"] = df_lbl["is_bot"].map({0: "Normal User", 1: "Water Army (Bot)"})
        
        # ── 3. Feature Distribution: Self-Similarity ────────────────────
        print("Generating: feature_distribution_similarity.png")
        try:
            plt.figure(figsize=(9, 5))
            sns.kdeplot(data=df_lbl, x="self_similarity", hue="User Type", fill=True, common_norm=False, alpha=0.4, palette="Set1")
            plt.title("Distribution of User Comment Similarity (Self-Similarity)", fontsize=14, fontweight="bold")
            plt.xlabel("TF-IDF Cosine Similarity (Higher = Repeated wash-messages)", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.xlim(0, 1.0)
            plt.tight_layout()
            plt.savefig("results/feature_distribution_similarity.png", dpi=300)
            plt.close()
            print("-> Successfully saved feature_distribution_similarity.png")
        except Exception as e:
            print(f"Error generating self-similarity plot: {e}")
            
        # ── 4. Feature Distribution: Interval Std ───────────────────────
        print("Generating: feature_distribution_interval.png")
        try:
            plt.figure(figsize=(9, 5))
            # Limit X range to 0 - 20,000 for visual clarity as normal users interval std can be huge
            sns.kdeplot(data=df_lbl, x="interval_std", hue="User Type", fill=True, common_norm=False, alpha=0.4, palette="Set1")
            plt.title("Distribution of Comment Time Intervals (Interval Std)", fontsize=14, fontweight="bold")
            plt.xlabel("Standard Deviation of Posting Intervals (Seconds)", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.xlim(0, 15000)
            plt.tight_layout()
            plt.savefig("results/feature_distribution_interval.png", dpi=300)
            plt.close()
            print("-> Successfully saved feature_distribution_interval.png")
        except Exception as e:
            print(f"Error generating interval std plot: {e}")
            
        # ── 5. 24-Hour Activity Pattern ─────────────────────────────────
        print("Generating: temporal_active_hours.png")
        try:
            # Load raw comments to join with user labels
            raw_files = glob.glob("data/collected/youtube_raw_*.csv")
            if raw_files:
                dfs = []
                for f in raw_files:
                    try:
                        dfs.append(pd.read_csv(f))
                    except:
                        pass
                raw_df = pd.concat(dfs, ignore_index=True)
                comments = raw_df[raw_df["record_type"].isin(["comment", "reply"])].copy()
                comments = comments[["author_id", "published_at"]].dropna()
                
                # Merge with labels to get User Type
                user_label_map = dict(zip(df_lbl["author_id"], df_lbl["User Type"]))
                comments["User Type"] = comments["author_id"].map(user_label_map)
                comments = comments.dropna(subset=["User Type"])
                
                # Parse hour
                comments["published_at"] = pd.to_datetime(comments["published_at"], utc=True)
                # Convert to local time (UTC+8) to represent Taiwan local active hours
                comments["Hour"] = comments["published_at"].dt.tz_convert("Asia/Taipei").dt.hour
                
                # Calculate percentages per user type
                hour_counts = comments.groupby(["User Type", "Hour"]).size().reset_index(name="Count")
                total_counts = comments.groupby("User Type").size().to_dict()
                hour_counts["Percentage"] = hour_counts.apply(lambda r: r["Count"] / total_counts[r["User Type"]] * 100, axis=1)
                
                plt.figure(figsize=(10, 5))
                sns.lineplot(data=hour_counts, x="Hour", y="Percentage", hue="User Type", marker="o", linewidth=2.5, palette="Set1")
                plt.title("24-Hour Commenting Activity Profile (Taiwan Time UTC+8)", fontsize=14, fontweight="bold")
                plt.xlabel("Hour of Day (0 - 23)", fontsize=12)
                plt.ylabel("Percentage of Total Comments (%)", fontsize=12)
                plt.xticks(range(0, 24))
                plt.xlim(0, 23)
                plt.tight_layout()
                plt.savefig("results/temporal_active_hours.png", dpi=300)
                plt.close()
                print("-> Successfully saved temporal_active_hours.png")
            else:
                print("No raw youtube comments CSV files found for temporal analysis.")
        except Exception as e:
            print(f"Error generating temporal active hours plot: {e}")
            
    else:
        print("Labeled data file not found at data/labeled_data.csv, skipping distribution plots.")

    print("All visualizations generated successfully.")

if __name__ == "__main__":
    main()
