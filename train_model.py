import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

print("載入標注資料中...")
df = pd.read_csv('data/labeled_data.csv')

# 確認並移除缺失值 (如果有的話)
df = df.dropna(subset=['is_bot'])

features = ['total_comments', 'unique_videos', 'concentration', 
            'avg_likes', 'zero_like_ratio', 'max_burst', 
            'self_similarity', 'interval_std']

X = df[features].fillna(0)  # 處理可能遺失的數值特徵
y = df['is_bot'].astype(int)

print(f"總共 {len(df)} 筆標注資料。")
print(f"其中正常用戶 (0): {len(y[y==0])} 筆")
print(f"其中機器人/水軍 (1): {len(y[y==1])} 筆")
print("-" * 30)

# 切割訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練隨機森林模型
print("開始訓練 Random Forest 模型...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# 評估模型
y_pred = rf_model.predict(X_test_scaled)
print("\n=== 測試集評估結果 ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 特徵重要性分析
importances = rf_model.feature_importances_
feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp = feature_imp.sort_values(by='Importance', ascending=False)

print("\n=== 特徵重要性排名 ===")
print(feature_imp.to_string(index=False))

# 儲存模型與縮放器
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/bot_classifier_rf.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n模型與特徵縮放器已成功儲存至 `models/` 目錄！")
