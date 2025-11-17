
# generate_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Setup ---
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

# --- Data Loading and Preprocessing ---
service_df = pd.read_csv('C:/practice/mid-term/data/service_usage.csv')
complaints_df = pd.read_csv('C:/practice/mid-term/data/citizen_complaints.csv')

service_df['date'] = pd.to_datetime(service_df['date'])
complaints_df['date'] = pd.to_datetime(complaints_df['date'])

treatment_districts = ['강북구', '강남구', '서초구']
service_df['policy_period'] = service_df['date'] >= '2022-01-01'
service_df['treatment_group'] = service_df['district'].isin(treatment_districts)

complaints_df['policy_period'] = complaints_df['date'] >= '2022-01-01'
complaints_df['treatment_group'] = complaints_df['district'].isin(treatment_districts)

# --- Plot 1: Chapter 2 - DID Parallel Trends ---
plt.figure(figsize=(10, 6))
monthly_usage = service_df.groupby([
    service_df['date'].dt.to_period('M'), 'treatment_group'
])['usage_count'].mean().reset_index()
monthly_usage['date'] = monthly_usage['date'].dt.to_timestamp()

for group in [True, False]:
    data = monthly_usage[monthly_usage['treatment_group'] == group]
    label = '처치군' if group else '대조군'
    plt.plot(data['date'], data['usage_count'], label=label, linewidth=2)

plt.axvline(x=pd.to_datetime('2022-01-01'), color='red', linestyle='--', alpha=0.7, label='정책 시작')
plt.title('DID 분석: 평행 추세 가정 검증 (이용률)', fontsize=16)
plt.ylabel('월별 평균 이용 건수')
plt.xlabel('날짜')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('C:/practice/mid-term/images/did_parallel_trends.png', bbox_inches='tight')
plt.close()

# --- Plot 2: Chapter 3 - ARIMA Forecast ---
treatment_ts = service_df[service_df['treatment_group']==True].groupby('date')['usage_count'].mean().sort_index()
train_data = treatment_ts[treatment_ts.index < '2022-01-01']
test_data = treatment_ts[treatment_ts.index >= '2022-01-01']
model = ARIMA(train_data, order=(1,1,1))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=len(test_data))
forecast_index = test_data.index

plt.figure(figsize=(12, 7))
plt.plot(train_data.index, train_data, label='학습 데이터 (정책 전)', color='gray')
plt.plot(test_data.index, test_data, label='실제 데이터 (정책 후)', color='blue')
plt.plot(forecast_index, forecast, label='ARIMA 예측 (정책 없었다면)', color='red', linestyle='--')
plt.title('ARIMA 모델 예측 결과', fontsize=16)
plt.ylabel('일별 평균 이용 건수')
plt.xlabel('날짜')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('C:/practice/mid-term/images/arima_forecast.png', bbox_inches='tight')
plt.close()

# --- Plot 3: Chapter 4 - Sentiment Analysis ---
sentiment_analysis = complaints_df[complaints_df['treatment_group'] == True].groupby(['policy_period'])['sentiment'].value_counts(normalize=True).unstack(fill_value=0)
sentiment_analysis = sentiment_analysis.rename(index={False: '정책 전', True: '정책 후'})
sentiment_analysis.plot(kind='bar', figsize=(10, 6), colormap='viridis', rot=0)
plt.title('처치군 민원 감성 분석 (정책 전후 비교)', fontsize=16)
plt.ylabel('비율')
plt.xlabel('정책 기간')
plt.ylim(0, 0.6)
plt.legend(title='감성')
plt.savefig('C:/practice/mid-term/images/sentiment_analysis.png', bbox_inches='tight')
plt.close()


# --- Plot 4: Chapter 5 - Feature Importance ---
ml_data = service_df.copy()
ml_data['year'] = ml_data['date'].dt.year
ml_data['month'] = ml_data['date'].dt.month
ml_data['day_of_week'] = ml_data['date'].dt.dayofweek
ml_data['policy_intervention'] = (ml_data['policy_period'] & ml_data['treatment_group']).astype(int)

le_district = LabelEncoder()
le_service = LabelEncoder()
ml_data['district_encoded'] = le_district.fit_transform(ml_data['district'])
ml_data['service_encoded'] = le_service.fit_transform(ml_data['service_type'])

features = ['district_encoded', 'service_encoded', 'year', 'month', 'day_of_week', 'treatment_group', 'policy_period', 'policy_intervention']
X = ml_data[features]
y_usage = ml_data['usage_count']

rf_usage = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_usage.fit(X, y_usage)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_usage.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
plt.title('머신러닝 모델 특성 중요도 (이용률 예측)', fontsize=16)
plt.xlabel('특성 중요도')
plt.ylabel('특성')
plt.savefig('C:/practice/mid-term/images/feature_importance.png', bbox_inches='tight')
plt.close()

print("All plots generated and saved successfully.")
