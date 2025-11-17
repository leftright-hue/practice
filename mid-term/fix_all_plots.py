 1 # Imports
     2 import pandas as pd
     3 import numpy as np
     4 import matplotlib.pyplot as plt
     5 import matplotlib.font_manager as
       fm
     6 import seaborn as sns
     7 import warnings
     8 import os
     9 from datetime import datetime
    10 import statsmodels.api as sm
    11 from statsmodels.tsa.arima.model
       import ARIMA
    12 from sklearn.ensemble import
       RandomForestRegressor
    13 from sklearn.model_selection impot
       train_test_split
    14 from sklearn.preprocessing import
       LabelEncoder
    15 
    16 # --- Font Setup ---
    17 # Windows에 기본으로 설치된 'Malgn
       Gothic' 폰트의 경로를 직접 
       지정합니다.
    18 font_path =
       "C:/Windows/Fonts/malgun.ttf"
    19 if os.path.exists(font_path):
    20     font_name =
       fm.FontProperties(fname=font_path.
       get_name()
    21     plt.rcParams['font.family'] =
       font_name
    22 else:
    23     print("Warning: 'malgun.ttf' 
       not found at C:/Windows/Fonts. 
       Please check the path.")
    24     # 폰트가 없는 경우를 대비해, 
       이름으로 다시 시도합니다.
    25     plt.rcParams['font.family'] =
       'Malgun Gothic'
    26
    27 plt.rcParams['axes.unicode_minus']
       = False
    28 warnings.filterwarnings('ignore') 
    29 sns.set_style("whitegrid")        
    30
    31 print(f"Using font: {plt.rcParams[
       'font.family']}")
    32 print("Regenerating ALL plots...")
    33
    34 # --- Data Loading ---
    35 try:
    36     service_df = pd.read_csv(     
       'data/service_usage.csv')
    37     complaints_df = pd.read_csv(  
       'data/citizen_complaints.csv')    
    38 except FileNotFoundError as e:    
    39     print(f"Error: Data files not 
       found in 'data' directory. {e}.   
       Make sure script is run from      
       'mid-term' folder.")
    40     exit()
    41
    42 service_df['date'] =
       pd.to_datetime(service_df['date'])
    43 service_df['policy_period'] =     
       service_df['date'] >= '2022-01-01'
    44 service_df['treatment_group'] =   
       service_df['district'].isin([     
       '강북구', '강남구', '서초구'])    
    45
    46 # --- Image Directory ---
    47 if not os.path.exists('images'):  
    48     os.makedirs('images')
    49
    50 # --- Plot Generation (All 8 plot)
       ---
    51
    52 # Plot 1
    53 plt.figure(figsize=(10, 6))       
    54 monthly_usage =
       service_df.groupby([service_df[   
       'date'].dt.to_period('M'),        
       'treatment_group'])['usage_count' 
       ].mean().reset_index()
    55 monthly_usage['date'] =
       monthly_usage['date'
       ].dt.to_timestamp()
    56 for group in [True, False]:       
    57     data =
       monthly_usage[monthly_usage[      
       'treatment_group'] == group]      
    58     label = '처치군' if group else
       '대조군'
    59     plt.plot(data['date'], data[  
       'usage_count'], label=label,      
       linewidth=2)
    60 plt.axvline(x=pd.to_datetime(     
       '2022-01-01'), color='red',       
       linestyle='--', alpha=0.7, label= 
       '정책 시작')
    61 plt.title('월별 평균 이용률 변화';
       plt.ylabel('평균 이용 건수');     
       plt.legend(); plt.grid(True, alph=
       0.3)
    62 plt.savefig(
       'images/monthly_usage_change.png';
       plt.close()
    63 print("Generated
       images/monthly_usage_change.png") 
    64
    65 # Plot 2
    66 plt.figure(figsize=(10, 6))       
    67 monthly_time =
       service_df.groupby([service_df[   
       'date'].dt.to_period('M'),        
       'treatment_group'])[
       'processing_time'
       ].mean().reset_index()
    68 monthly_time['date'] =
       monthly_time['date'
       ].dt.to_timestamp()
    69 for group in [True, False]:       
    70     data =
       monthly_time[monthly_time[        
       'treatment_group'] == group]      
    71     label = '처치군' if group else
       '대조군'
    72     plt.plot(data['date'], data[  
       'processing_time'], label=label,  
       linewidth=2)
    73 plt.axvline(x=pd.to_datetime(     
       '2022-01-01'), color='red',       
       linestyle='--', alpha=0.7, label= 
       '정책 시작')
    74 plt.title('월별 평균 처리시간 변 '
       ); plt.ylabel('평균 처리시간 (분)'
       ); plt.legend(); plt.grid(True,   
       alpha=0.3)
    75 plt.savefig(
       'images/monthly_processing_time_ca
       nge.png'); plt.close()
    76 print("Generated
       images/monthly_processing_time_chn
       ge.png")
    77
    78 # Plot 3
    79 plt.figure(figsize=(10, 6))       
    80 monthly_satisfaction =
       service_df.groupby([service_df[   
       'date'].dt.to_period('M'),        
       'treatment_group'])['satisfaction'
       ].mean().reset_index()
    81 monthly_satisfaction['date'] =    
       monthly_satisfaction['date'       
       ].dt.to_timestamp()
    82 for group in [True, False]:       
    83     data =
       monthly_satisfaction[monthly_satif
       action['treatment_group'] == grou]
    84     label = '처치군' if group else
       '대조군'
    85     plt.plot(data['date'], data[  
       'satisfaction'], label=label,     
       linewidth=2)
    86 plt.axvline(x=pd.to_datetime(     
       '2022-01-01'), color='red',       
       linestyle='--', alpha=0.7, label= 
       '정책 시작')
    87 plt.title('월별 평균 만족도 변화';
       plt.ylabel('평균 만족도 (5점 척도'
       ); plt.legend(); plt.grid(True,   
       alpha=0.3)
    88 plt.savefig(
       'images/monthly_satisfaction_chane
       .png'); plt.close()
    89 print("Generated
       images/monthly_satisfaction_chang.
       png")
    90
    91 # Plot 4
    92 plt.figure(figsize=(10, 6))       
    93 district_stats =
       service_df[service_df[
       'policy_period']==True].groupby(  
       'district').agg({'satisfaction':  
       'mean'}).round(2)
    94 colors = ['red' if district in [  
       '강북구', '강남구', '서초구'] else
       'blue' for district in
       district_stats.index]
    95 plt.bar(district_stats.index,     
       district_stats['satisfaction'],   
       color=colors, alpha=0.7)
    96 plt.title('자치구별 만족도 (정책  
       이후)'); plt.ylabel('평균 만족도';
       plt.xticks(rotation=45)
    97 from matplotlib.patches import    
       Patch
    98 legend_elements = [Patch(facecolo=
       'red', alpha=0.7, label='처치군'),
       Patch(facecolor='blue', alpha=0.7,
       label='대조군')]
    99 plt.legend(handles=legend_element)
       ; plt.tight_layout()
   100 plt.savefig(
       'images/district_satisfaction_aftr
       _policy.png'); plt.close()        
   101 print("Generated
       images/district_satisfaction_afte_
       policy.png")
   102
   103 # Plot 5
   104 plt.figure(figsize=(10, 6))       
   105 monthly_avg =
       service_df.groupby([service_df[   
       'date'].dt.to_period('M'),        
       'treatment_group'])['usage_count' 
       ].mean().unstack()
   106 monthly_avg.index =
       monthly_avg.index.to_timestamp()  
   107 plt.plot(monthly_avg.index[monthl_
       avg.index < '2022-01-01'],        
       monthly_avg[False
       ][monthly_avg.index < '2022-01-01'
       ], label='대조군', color='blue')  
   108 plt.plot(monthly_avg.index[monthl_
       avg.index < '2022-01-01'],        
       monthly_avg[True][monthly_avg.indx
       < '2022-01-01'], label='처치군',  
       color='red')
   109 plt.axvline(pd.to_datetime(       
       '2022-01-01'), color='black',     
       linestyle='--', label='정책 시작')
   110 plt.title('DID 분석: 평행 추세 가 
       검증 (이용률)'); plt.ylabel('월   
       평균 이용률'); plt.legend()       
   111 plt.savefig(
       'images/did_parallel_trends.png');
       plt.close()
   112 print("Generated
       images/did_parallel_trends.png")  
   113
   114 # Plot 6
   115 plt.figure(figsize=(10, 6))       
   116 treatment_ts =
       service_df[service_df[
       'treatment_group']==True].groupby(
       'date')['usage_count'
       ].mean().sort_index()
   117 train_data =
       treatment_ts[treatment_ts.index < 
       '2022-01-01']
   118 test_data =
       treatment_ts[treatment_ts.index >=
       '2022-01-01']
   119 model = ARIMA(train_data, order=(,
       1,1)); fitted_model = model.fit() 
   120 forecast =
       fitted_model.forecast(steps=len   
       (test_data))
   121 plt.plot(train_data.index,        
       train_data, label='학습 데이터')  
   122 plt.plot(test_data.index,
       test_data, label='실제 데이터',   
       color='orange')
   123 plt.plot(test_data.index, forecas,
       label='ARIMA 예측', color='green',
       linestyle='--')
   124 plt.title('ARIMA 모델 예측 결과');
       plt.ylabel('이용 건수');
       plt.legend()
   125 plt.savefig(
       'images/arima_forecast.png');     
       plt.close()
   126 print("Generated
       images/arima_forecast.png")       
   127
   128 # Plot 7
   129 plt.figure(figsize=(10, 6))       
   130 complaints_df['date'] =
       pd.to_datetime(complaints_df['dat'
       ])
   131 complaints_df['policy_period'] =  
       complaints_df['date'] >=
       '2022-01-01'
   132 complaints_df['treatment_group'] =
       complaints_df['district'].isin([  
       '강북구', '강남구', '서초구'])    
   133 sentiment_analysis =
       complaints_df[complaints_df[      
       'treatment_group']==True].groupby[
       'policy_period'])['sentiment'     
       ].value_counts(normalize=True     
       ).unstack(fill_value=0)
   134 sentiment_analysis.plot(kind='bar,
       stacked=True, figsize=(10,6), rot0
       , ax=plt.gca())
   135 plt.title('정책 전후 민원 감성 분 
       (처치군)'); plt.ylabel('비율');   
       plt.xlabel('정책 기간')
   136 plt.xticks([0, 1], ['정책 전',    
       '정책 후']); plt.tight_layout()   
   137 plt.savefig(
       'images/sentiment_analysis.png'); 
       plt.close()
   138 print("Generated
       images/sentiment_analysis.png")   
   139
   140 # Plot 8
   141 plt.figure(figsize=(10, 6))       
   142 ml_data = service_df.copy()       
   143 ml_data['year'] = ml_data['date'  
       ].dt.year; ml_data['month'] =     
       ml_data['date'].dt.month; ml_data[
       'day_of_week'] = ml_data['date'   
       ].dt.dayofweek
   144 le_district = LabelEncoder();     
       le_service = LabelEncoder()       
   145 ml_data['district_encoded'] =     
       le_district.fit_transform(ml_data[
       'district'])
   146 ml_data['service_encoded'] =      
       le_service.fit_transform(ml_data[ 
       'service_type'])
   147 ml_data['policy_intervention'] =  
       (ml_data['policy_period'] &       
       ml_data['treatment_group']).astyp(
       int)
   148 features = ['district_encoded',   
       'service_encoded', 'year', 'month,
       'day_of_week', 'treatment_group', 
       'policy_period',
       'policy_intervention']
   149 X = ml_data[features]; y_usage =  
       ml_data['usage_count']
   150 X_train, X_test, y_usage_train,   
       y_usage_test = train_test_split(X,
       y_usage, test_size=0.2,
       random_state=42)
   151 rf_usage =
       RandomForestRegressor(n_estimator=
       100, random_state=42);
       rf_usage.fit(X_train,
       y_usage_train)
   152 feature_importance_usage =        
       pd.Series(rf_usage.feature_importn
       ces_,
       index=features).sort_values(asceni
       ng=False)
   153 sns.barplot(x=feature_importance_s
       age,
       y=feature_importance_usage.index) 
   154 plt.title('머신러닝 모델 특성     
       중요도'); plt.xlabel('중요도');   
       plt.ylabel('특성');
       plt.tight_layout()
   155 plt.savefig(
       'images/feature_importance.png'); 
       plt.close()
   156 print("Generated
       images/feature_importance.png")   
   157
   158 print("All plots regenerated      
       successfully.")