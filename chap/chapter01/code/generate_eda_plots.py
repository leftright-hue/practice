import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Setup paths
current_dir = Path(__file__).parent
data_dir = Path("C:/practice/mid-term/data")
output_image_dir = Path("C:/practice/mid-term/images")
output_image_dir.mkdir(parents=True, exist_ok=True)

# Load data
try:
    df = pd.read_csv(data_dir / 'service_usage.csv')
except FileNotFoundError:
    print(f"Error: service_usage.csv not found at {data_dir / 'service_usage.csv'}")
    exit()

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['month_year'] = df['date'].dt.to_period('M')

# Define treatment and control groups
treatment_districts = ['강북구', '강남구', '서초구']
control_districts = ['송파구', '마포구']

df['group'] = df['district'].apply(lambda x: 'Treatment' if x in treatment_districts else ('Control' if x in control_districts else 'Other'))

# Filter out 'Other' if any
df = df[df['group'] != 'Other']

# Policy intervention date
policy_start_date = pd.to_datetime('2022-01-01')
df['policy_period'] = df['date'].apply(lambda x: 'Post-Policy' if x >= policy_start_date else 'Pre-Policy')

# Set Korean font for matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic' # For Windows
plt.rcParams['axes.unicode_minus'] = False

# --- Plot 1: 월별 평균 이용률 변화 (monthly_usage_change.png) ---
monthly_usage = df.groupby(['month_year', 'group'])['usage_count'].mean().reset_index()
monthly_usage['month_year'] = monthly_usage['month_year'].astype(str) # Convert Period to string for plotting

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_usage, x='month_year', y='usage_count', hue='group', marker='o')
plt.axvline(x='2022-01', color='red', linestyle='--', lw=2, label='정책 도입 (2022-01)')
plt.title('월별 평균 서비스 이용률 변화')
plt.xlabel('월')
plt.ylabel('평균 이용 건수')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='그룹')
plt.tight_layout()
plt.savefig(output_image_dir / 'monthly_usage_change.png')
plt.close()
print(f"Generated: {output_image_dir / 'monthly_usage_change.png'}")

# --- Plot 2: 월별 평균 처리시간 변화 (monthly_processing_time_change.png) ---
monthly_processing_time = df.groupby(['month_year', 'group'])['processing_time'].mean().reset_index()
monthly_processing_time['month_year'] = monthly_processing_time['month_year'].astype(str)

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_processing_time, x='month_year', y='processing_time', hue='group', marker='o')
plt.axvline(x='2022-01', color='red', linestyle='--', lw=2, label='정책 도입 (2022-01)')
plt.title('월별 평균 처리 시간 변화')
plt.xlabel('월')
plt.ylabel('평균 처리 시간 (분)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='그룹')
plt.tight_layout()
plt.savefig(output_image_dir / 'monthly_processing_time_change.png')
plt.close()
print(f"Generated: {output_image_dir / 'monthly_processing_time_change.png'}")

# --- Plot 3: 월별 평균 만족도 변화 (monthly_satisfaction_change.png) ---
monthly_satisfaction = df.groupby(['month_year', 'group'])['satisfaction'].mean().reset_index()
monthly_satisfaction['month_year'] = monthly_satisfaction['month_year'].astype(str)

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_satisfaction, x='month_year', y='satisfaction', hue='group', marker='o')
plt.axvline(x='2022-01', color='red', linestyle='--', lw=2, label='정책 도입 (2022-01)')
plt.title('월별 평균 만족도 변화')
plt.xlabel('월')
plt.ylabel('평균 만족도 (1-5점)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='그룹')
plt.tight_layout()
plt.savefig(output_image_dir / 'monthly_satisfaction_change.png')
plt.close()
print(f"Generated: {output_image_dir / 'monthly_satisfaction_change.png'}")

# --- Plot 4: 자치구별 만족도 (정책 이후) (district_satisfaction_after_policy.png) ---
# Filter data for post-policy period and calculate average satisfaction by district
post_policy_satisfaction = df[df['policy_period'] == 'Post-Policy'].groupby('district')['satisfaction'].mean().reset_index()

# Order districts for better visualization (Treatment first, then Control)
district_order = treatment_districts + control_districts
post_policy_satisfaction['district'] = pd.Categorical(post_policy_satisfaction['district'], categories=district_order, ordered=True)
post_policy_satisfaction = post_policy_satisfaction.sort_values('district')

plt.figure(figsize=(10, 6))
sns.barplot(data=post_policy_satisfaction, x='district', y='satisfaction', palette='viridis')
plt.title('정책 도입 이후 자치구별 평균 만족도')
plt.xlabel('자치구')
plt.ylabel('평균 만족도 (1-5점)')
plt.ylim(2.5, 4.5) # Adjust y-axis for better comparison
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(output_image_dir / 'district_satisfaction_after_policy.png')
plt.close()
print(f"Generated: {output_image_dir / 'district_satisfaction_after_policy.png'}")

print("\nAll EDA plots generated successfully.")
