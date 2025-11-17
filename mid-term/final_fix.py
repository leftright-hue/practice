
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os
import glob
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Step 1: Clear Matplotlib Font Cache ---
print("Attempting to clear matplotlib font cache...")
try:
    cachedir = matplotlib.get_cachedir()
    font_cache_files = glob.glob(os.path.join(cachedir, 'fontlist-*.json'))
    if not font_cache_files:
        print("No font cache files found to clear.")
    for f in font_cache_files:
        print(f"Deleting font cache file: {f}")
        os.remove(f)
    print("Font cache cleared successfully.")
except Exception as e:
    print(f"Could not clear font cache: {e}")

# --- Step 2: Font Setup ---
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    print(f"'malgun.ttf' found at {font_path}")
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
else:
    print("Warning: 'malgun.ttf' not found. Using default font.")
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print(f"Using font: {plt.rcParams['font.family']}")
print("Regenerating ALL plots...")

# --- Data Loading ---
try:
    service_df = pd.read_csv('data/service_usage.csv')
    complaints_df = pd.read_csv('data/citizen_complaints.csv')
except FileNotFoundError as e:
    print(f"Error: Data file not found. {e}. Make sure script is run from 'mid-term' directory.")
    exit()

service_df['date'] = pd.to_datetime(service_df['date'])
service_df['policy_period'] = service_df['date'] >= '2022-01-01'
service_df['treatment_group'] = service_df['district'].isin(['강북구', '강남구', '서초구'])

if not os.path.exists('images'):
    os.makedirs('images')

# --- Plot Generation (All 8 plots) ---

# Plot 1
plt.figure(figsize=(10, 6))
monthly_usage = service_df.groupby([service_df['date'].dt.to_period('M'), 'treatment_group'])['usage_count'].mean().reset_index()
monthly_usage['date'] = monthly_usage['date'].dt.to_timestamp()
for group in [True, False]:
    data = monthly_usage[monthly_usage['treatment_group'] == group]
    label = '처치군' if group else '대조군'
    plt.plot(data['date'], data['usage_count'], label=label, linewidth=2)
plt.axvline(x=pd.to_datetime('2022-01-01'), color='red', linestyle='--', alpha=0.7, label='정책 시작')
plt.title('월별 평균 이용률 변화'); plt.ylabel('평균 이용 건수'); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('images/monthly_usage_change.png'); plt.close()
print("Generated images/monthly_usage_change.png")

# Plot 2
plt.figure(figsize=(10, 6))
monthly_time = service_df.groupby([service_df['date'].dt.to_period('M'), 'treatment_group'])['processing_time'].mean().reset_index()
monthly_time['date'] = monthly_time['date'].dt.to_timestamp()
for group in [True, False]:
    data = monthly_time[monthly_time['treatment_group'] == group]
    label = '처치군' if group else '대조군'
    plt.plot(data['date'], data['processing_time'], label=label, linewidth=2)
plt.axvline(x=pd.to_datetime('2022-01-01'), color='red', linestyle='--', alpha=0.7, label='정책 시작')
plt.title('월별 평균 처리시간 변화'); plt.ylabel('평균 처리시간 (분)'); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('images/monthly_processing_time_change.png'); plt.close()
print("Generated images/monthly_processing_time_change.png")

# Plot 3
plt.figure(figsize=(10, 6))
monthly_satisfaction = service_df.groupby([service_df['date'].dt.to_period('M'), 'treatment_group'])['satisfaction'].mean().reset_index()
monthly_satisfaction['date'] = monthly_satisfaction['date'].dt.to_timestamp()
for group in [True, False]:
    data = monthly_satisfaction[monthly_satisfaction['treatment_group'] == group]
    label = '처치군' if group else '대조군'
    plt.plot(data['date'], data['satisfaction'], label=label, linewidth=2)
plt.axvline(x=pd.to_datetime('2022-01-01'), color='red', linestyle='--', alpha=0.7, label='정책 시작')
plt.title('월별 평균 만족도 변화'); plt.ylabel('평균 만족도 (5점 척도)'); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('images/monthly_satisfaction_change.png'); plt.close()
print("Generated images/monthly_satisfaction_change.png")

# Plot 4
plt.figure(figsize=(10, 6))
district_stats = service_df[service_df['policy_period']==True].groupby('district').agg({'satisfaction': 'mean'}).round(2)
colors = ['red' if district in ['강북구', '강남구', '서초구'] else 'blue' for district in district_stats.index]
plt.bar(district_stats.index, district_stats['satisfaction'], color=colors, alpha=0.7)
plt.title('자치구별 만족도 (정책 이후)'); plt.ylabel('평균 만족도'); plt.xticks(rotation=45)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='처치군'), Patch(facecolor='blue', alpha=0.7, label='대조군')]
plt.legend(handles=legend_elements); plt.tight_layout()
plt.savefig('images/district_satisfaction_after_policy.png'); plt.close()
print("Generated images/district_satisfaction_after_policy.png")

# Plot 5
plt.figure(figsize=(10, 6))
monthly_avg = service_df.groupby([service_df['date'].dt.to_period('M'), 'treatment_group'])['usage_count'].mean().unstack()
monthly_avg.index = monthly_avg.index.to_timestamp()
plt.plot(monthly_avg.index[monthly_avg.index < '2022-01-01'], monthly_avg[False][monthly_avg.index < '2022-01-01'], label='대조군', color='blue')
plt.plot(monthly_avg.index[monthly_avg.index < '2022-01-01'], monthly_avg[True][monthly_avg.index < '2022-01-01'], label='처치군', color='red')
plt.axvline(pd.to_datetime('2022-01-01'), color='black', linestyle='--', label='정책 시작')
plt.title('DID 분석: 평행 추세 가정 검증 (이용률)'); plt.ylabel('월 평균 이용률'); plt.legend()
plt.savefig('images/did_parallel_trends.png'); plt.close()
print("Generated images/did_parallel_trends.png")

# Plot 6
plt.figure(figsize=(10, 6))
treatment_ts = service_df[service_df['treatment_group']==True].groupby('date')['usage_count'].mean().sort_index()
train_data = treatment_ts[treatment_ts.index < '2022-01-01']
test_data = treatment_ts[treatment_ts.index >= '2022-01-01']
model = ARIMA(train_data, order=(1,1,1)); fitted_model = model.fit()
forecast = fitted_model.forecast(steps=len(test_data))
plt.plot(train_data.index, train_data, label='학습 데이터')
plt.plot(test_data.index, test_data, label='실제 데이터', color='orange')
plt.plot(test_data.index, forecast, label='ARIMA 예측', color='green', linestyle='--')
plt.title('ARIMA 모델 예측 결과'); plt.ylabel('이용 건수'); plt.legend()
plt.savefig('images/arima_forecast.png'); plt.close()
print("Generated images/arima_forecast.png")

# Plot 7
plt.figure(figsize=(10, 6))
complaints_df['date'] = pd.to_datetime(complaints_df['date'])
complaints_df['policy_period'] = complaints_df['date'] >= '2022-01-01'
complaints_df['treatment_group'] = complaints_df['district'].isin(['강북구', '강남구', '서초구'])
sentiment_analysis = complaints_df[complaints_df['treatment_group']==True].groupby(['policy_period'])['sentiment'].value_counts(normalize=True).unstack(fill_value=0)
sentiment_analysis.plot(kind='bar', stacked=True, figsize=(10,6), rot=0, ax=plt.gca())
plt.title('정책 전후 민원 감성 분석 (처치군)'); plt.ylabel('비율'); plt.xlabel('정책 기간')
plt.xticks([0, 1], ['정책 전', '정책 후']); plt.tight_layout()
plt.savefig('images/sentiment_analysis.png'); plt.close()
print("Generated images/sentiment_analysis.png")

# Plot 8
plt.figure(figsize=(10, 6))
ml_data = service_df.copy()
ml_data['year'] = ml_data['date'].dt.year; ml_data['month'] = ml_data['date'].dt.month; ml_data['day_of_week'] = ml_data['date'].dt.dayofweek
le_district = LabelEncoder(); le_service = LabelEncoder()
ml_data['district_encoded'] = le_district.fit_transform(ml_data['district'])
ml_data['service_encoded'] = le_service.fit_transform(ml_data['service_type'])
ml_data['policy_intervention'] = (ml_data['policy_period'] & ml_data['treatment_group']).astype(int)
features = ['district_encoded', 'service_encoded', 'year', 'month', 'day_of_week', 'treatment_group', 'policy_period', 'policy_intervention']
X = ml_data[features]; y_usage = ml_data['usage_count']
X_train, X_test, y_usage_train, y_usage_test = train_test_split(X, y_usage, test_size=0.2, random_state=42)
rf_usage = RandomForestRegressor(n_estimators=100, random_state=42); rf_usage.fit(X_train, y_usage_train)
feature_importance_usage = pd.Series(rf_usage.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_importance_usage, y=feature_importance_usage.index)
plt.title('머신러닝 모델 특성 중요도'); plt.xlabel('중요도'); plt.ylabel('특성'); plt.tight_layout()
plt.savefig('images/feature_importance.png'); plt.close()
print("Generated images/feature_importance.png")

print("All plots regenerated successfully.")
