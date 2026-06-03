import pandas as pd
import sys


def merge_reviewed_labels(
    labeled_path: str = "data/labeled_data.csv",
    pending_path: str = "data/pending_review.csv",
    output_path: str = "data/labeled_data.csv",
) -> dict:
    """
    合併人工複審的標籤。
    - 只合併 is_bot 為 "0" 或 "1" 的列（轉為 int）
    - 跳過空字串/NaN（計入 skipped_empty）
    - 非法值（"2"、"yes" 等）拋 ValueError 並終止
    - 重複 author_id → 保留既有版本，stderr 輸出警告（計入 skipped_duplicate）
    回傳 {"merged": N, "skipped_empty": N, "skipped_duplicate": N}
    """
    # 讀取現有標注資料
    labeled_df = pd.read_csv(labeled_path)
    existing_ids = set(labeled_df["author_id"])

    # 讀取待複審資料，is_bot 作為字符串處理
    pending_df = pd.read_csv(pending_path, dtype={"is_bot": str}, keep_default_na=False)

    # 初始化統計
    merged = 0
    skipped_empty = 0
    skipped_duplicate = 0

    # 篩選要合併的列
    to_merge = []

    for idx, row in pending_df.iterrows():
        is_bot_val = row["is_bot"]

        # 檢查是否為空
        if is_bot_val == "" or pd.isna(is_bot_val):
            skipped_empty += 1
            continue

        # 驗證值為 "0" 或 "1"
        if is_bot_val not in ("0", "1"):
            raise ValueError(
                f"非法 is_bot 值 '{is_bot_val}'，位於第 {idx} 列 (author_id={row['author_id']})"
            )

        # 檢查重複 author_id
        author_id = row["author_id"]
        if author_id in existing_ids:
            print(
                f"警告：重複 author_id {author_id}，跳過",
                file=sys.stderr
            )
            skipped_duplicate += 1
            continue

        # 轉換為 int
        row_copy = row.copy()
        row_copy["is_bot"] = int(is_bot_val)
        to_merge.append(row_copy)
        merged += 1
        existing_ids.add(author_id)  # 更新現有 ID 集合

    # 合併資料
    if to_merge:
        merge_df = pd.DataFrame(to_merge)
        result_df = pd.concat([labeled_df, merge_df], ignore_index=True)
    else:
        result_df = labeled_df

    # 依 author_id 去重（保留第一次出現的版本）
    result_df = result_df.drop_duplicates(subset=["author_id"], keep="first")

    # 覆寫輸出檔案
    result_df.to_csv(output_path, index=False)

    return {
        "merged": merged,
        "skipped_empty": skipped_empty,
        "skipped_duplicate": skipped_duplicate,
    }


if __name__ == "__main__":
    import os
    pending_path = "data/pending_review_test.csv" if os.path.exists("data/pending_review_test.csv") else "data/pending_review.csv"
    stats = merge_reviewed_labels(pending_path=pending_path)
    print(f"[merge_labels] merged={stats['merged']}, skipped_empty={stats['skipped_empty']}, skipped_duplicate={stats['skipped_duplicate']}")
    df = pd.read_csv("data/labeled_data.csv")
    print(f"labeled_data.csv total rows: {len(df)}, bot: {df['is_bot'].sum()}, normal: {(df['is_bot']==0).sum()}")
