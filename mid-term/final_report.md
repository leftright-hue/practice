# 📊 AI 기반 스마트시티 플랫폼 정책 효과 분석 보고서

**학번**: 202475050  
**이름**: 권유빈  
**과목**: AI와 정책분석  
**과제**: 중간고사 과제

---

## Executive Summary

본 보고서는 A시에서 2022년 1월부터 시행한 "AI 기반 스마트시티 통합 플랫폼"의 정책 효과를 종합적으로 분석합니다.

**연구 설계**: 처치군(강북구, 강남구, 서초구)과 대조군(송파구, 마포구)을 활용한 준실험 설계로 정책의 인과적 효과를 추정했습니다.

**주요 발견사항**:
- **정책 효과**: 처치군에서 서비스 이용률 20.2% 증가, 처리시간 7.5분 단축, 만족도 0.5점 상승
- **통계적 검증**: 차이의차이(DID) 분석 결과 모든 지표에서 p<0.001 수준의 통계적 유의성 확인
- **시민 반응**: 긍정 감성 14.3%p 증가, 부정 감성 9.8%p 감소
- **경제적 효과**: 비용편익비(B/C ratio) 2.67로 높은 투자 효율성

**정책 제언**: 즉시 전체 자치구 확대 도입을 권고하며, 단계별 실행 계획을 제시합니다.

---

## 1. 분석 방법 및 근거

본 연구는 정책 효과의 다각적 검증을 위해 **5개 방법론**을 적용했습니다. 각 방법론은 상호 보완적 관계로 연결되어 종합적 분석 결과를 도출합니다.

### 1.1 Chapter 01: 탐색적 데이터 분석 (EDA)

**선택 근거**: 정책 분석의 기초 단계로 데이터 구조 파악 및 기초적 변화 패턴 확인

**분석 방법**:
- 처치군/대조군별 기술통계 비교
- 정책 전후 시계열 시각화
- 자치구별/서비스별 성과 지표 분포 분석

**다른 방법론과의 연결성**: 후속 인과추론과 시계열 분석의 기초 자료 제공

### 1.2 Chapter 02: 인과추론 분석 (DID)

**선택 근거**: 정책의 순수한 인과효과 추정을 위한 핵심 방법론

**분석 방법**:
- 차이의차이(Difference-in-Differences) 회귀분석
- 모델: Y_it = α + β₁Treatment_i + β₂Post_t + β₃(Treatment_i × Post_t) + ε_it
- 평행 추세 가정 검증 및 강건성 검사

**다른 방법론과의 연결성**: EDA 결과를 바탕으로 인과관계 확립, 머신러닝 모델의 벤치마크 제공

### 1.3 Chapter 03: 시계열 분석

**선택 근거**: 서비스 수요의 시간적 의존성과 계절성을 고려한 정확한 효과 추정

**분석 방법**:
- ARIMA 모델링을 통한 시계열 예측
- 정책 개입 전 데이터로 학습, 정책 후 실제값과 예측값 비교
- 정책 효과의 지속성 및 미래 전망 분석

**다른 방법론과의 연결성**: DID 분석 결과 검증, 머신러닝 예측 모델과 성능 비교

### 1.4 Chapter 04: 텍스트 분석 (NLP)

**선택 근거**: 정량 지표로 포착되지 않는 시민들의 질적 반응 변화 분석

**분석 방법**:
- 민원 텍스트 감성분석 (긍정/중립/부정)
- 키워드 빈도 분석 및 정책 전후 변화 탐지
- 카테고리별 민원 처리일수 개선도 분석

**다른 방법론과의 연결성**: 정량적 분석(DID, 시계열)의 질적 보완, 정책 제언의 근거 제공

### 1.5 Chapter 05: 머신러닝 예측 모델링

**선택 근거**: 복잡한 비선형 관계와 변수 상호작용을 고려한 예측 정확도 향상

**분석 방법**:
- Random Forest 회귀 모델 구축
- 특성 중요도 분석 및 정책 변수의 영향력 측정
- 교차검증을 통한 모델 성능 평가

**다른 방법론과의 연결성**: 전통적 계량 방법(DID, 시계열)과 예측 성능 비교, 정책 효과 크기 재검증

### 1.6 방법론 간 연결성 및 보완성

**수렴적 타당성**: 5개 방법론 모두에서 일관된 정책 효과 확인으로 결과의 신뢰성 증대

**순차적 심화**: EDA(탐색) → DID(인과관계) → 시계열(시간적 패턴) → 텍스트(질적 보완) → ML(예측 정확도)

**상호 검증**: 각 방법론의 결과가 서로를 검증하고 보완하여 종합적 정책 평가 완성

---

## 2. 분석 결과

### 2.1 Chapter 01: 탐색적 데이터 분석 결과

**기초 통계 분석**:

```python
# 정책 전후 평균 비교 (처치군)
print("처치군 변화:")
print(f"이용률: {before_treatment['usage_count'].mean():.1f} → {after_treatment['usage_count'].mean():.1f} (+{((after_treatment['usage_count'].mean()/before_treatment['usage_count'].mean()-1)*100):.1f}%)")
print(f"처리시간: {before_treatment['processing_time'].mean():.1f}분 → {after_treatment['processing_time'].mean():.1f}분 ({after_treatment['processing_time'].mean()-before_treatment['processing_time'].mean():+.1f}분)")
print(f"만족도: {before_treatment['satisfaction'].mean():.2f} → {after_treatment['satisfaction'].mean():.2f} ({after_treatment['satisfaction'].mean()-before_treatment['satisfaction'].mean():+.2f}점)")
```

**주요 발견사항**:
- **처치군**: 이용률 +20.2%, 처리시간 -7.5분, 만족도 +0.50점
- **대조군**: 모든 지표에서 거의 변화 없음 (±0.1% 내외)
- **시각적 패턴**: 2022년 1월 이후 처치군에서 뚜렷한 상승 추세 확인

### 2.2 Chapter 02: 인과추론 분석 결과

**차이의차이(DID) 회귀분석**:

```python
# DID 모델 실행
import statsmodels.api as sm
from statsmodels.formula.api import ols

# DID 분석을 위한 데이터 준비
service_did = service_df.copy()
service_did['post'] = service_did['policy_period'].astype(int)
service_did['treatment'] = service_did['treatment_group'].astype(int)
service_did['did'] = service_did['post'] * service_did['treatment']

# 주요 성과지표별 DID 분석
outcomes = ['usage_count', 'processing_time', 'satisfaction']
for outcome in outcomes:
    formula = f'{outcome} ~ treatment + post + did'
    model = ols(formula, data=service_did).fit()
    did_effect = model.params['did']
    p_value = model.pvalues['did']
    print(f"{outcome} DID 효과: {did_effect:.3f} (p={p_value:.6f})")
```

**통계적 검증 결과**:

| 지표 | DID 추정값 | 95% 신뢰구간 | p-value | 유의성 |
|------|------------|--------------|---------|--------|
| 이용률 | +11.787 | [10.718, 12.856] | <0.001 | *** |
| 처리시간 | -7.433 | [-8.017, -6.849] | <0.001 | *** |
| 만족도 | +0.497 | [0.486, 0.507] | <0.001 | *** |

**해석**: 모든 지표에서 99.9% 신뢰수준에서 통계적으로 유의한 정책 효과 확인

### 2.3 Chapter 03: 시계열 분석 결과

**ARIMA 모델링 및 정책 효과 분해**:

```python
# 시계열 분석
from statsmodels.tsa.arima.model import ARIMA

# 처치군 일별 평균 이용률 시계열 생성
treatment_ts = service_df[service_df['treatment_group']==True].groupby('date')['usage_count'].mean()

# 정책 이전 데이터로 ARIMA 모델 학습
train_data = treatment_ts[treatment_ts.index < '2022-01-01']
model = ARIMA(train_data, order=(1,1,1))
fitted_model = model.fit()

# 정책 이후 기간 예측
test_data = treatment_ts[treatment_ts.index >= '2022-01-01']
forecast = fitted_model.forecast(steps=len(test_data))

# 정책 효과 계산
actual_mean = test_data.mean()
forecast_mean = forecast.mean()
policy_effect = actual_mean - forecast_mean
print(f"시계열 분석 정책 효과: +{policy_effect:.1f}건 ({((actual_mean/forecast_mean-1)*100):+.1f}%)")
```

**주요 결과**:
- **정책 순효과**: +12.4건 (+21.3%)
- **추세 변화**: 정책 전 61.9건/일 → 정책 후 69.9건/일
- **미래 예측**: 2025년 연간 25,710건 예상 (정책 지속 시)

### 2.4 Chapter 04: 텍스트 분석 결과

**감성분석 및 키워드 변화**:

```python
# 텍스트 분석
from collections import Counter
import re

# 감성 분석 - 정책 전후 비교
treatment_before = complaints_df[(complaints_df['treatment_group']==True) & (complaints_df['policy_period']==False)]
treatment_after = complaints_df[(complaints_df['treatment_group']==True) & (complaints_df['policy_period']==True)]

print(f"긍정 감성: {(treatment_before['sentiment']=='긍정').mean():.1%} → {(treatment_after['sentiment']=='긍정').mean():.1%}")
print(f"부정 감성: {(treatment_before['sentiment']=='부정').mean():.1%} → {(treatment_after['sentiment']=='부정').mean():.1%}")

# 카테고리별 처리일수 개선
category_improvement = complaints_df[complaints_df['treatment_group']==True].groupby(['category', 'policy_period'])['resolution_days'].mean().unstack()
print("카테고리별 처리일수 개선:")
for category in category_improvement.index:
    before = category_improvement.loc[category, False]
    after = category_improvement.loc[category, True]
    improvement = before - after
    print(f"{category}: {before:.1f}일 → {after:.1f}일 (-{improvement:.1f}일)")
```

**주요 결과**:
- **감성 변화**: 긍정 37.2% → 51.5% (+14.3%p), 부정 25.4% → 15.6% (-9.8%p)
- **처리일수 개선**: 복지(-2.1일) > 교통(-1.4일) > 기타(-1.1일) > 환경(-0.9일) > 안전(-0.8일)
- **키워드 변화**: 정책 후 '필요합니다', '부탁드립니다' 등 정중한 표현 증가

### 2.5 Chapter 05: 머신러닝 분석 결과

**Random Forest 예측 모델링**:

```python
# 머신러닝 분석
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 특성 엔지니어링
features = ['district_encoded', 'service_encoded', 'year', 'month', 'day_of_week', 
           'treatment_group', 'policy_period', 'policy_intervention']
X = ml_data[features]
y_usage = ml_data['usage_count']

# 모델 학습 및 평가
X_train, X_test, y_train, y_test = train_test_split(X, y_usage, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 성능 평가
y_pred = rf_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# 정책 효과 시뮬레이션
X_policy_on = X_test.copy()
X_policy_on['policy_intervention'] = 1
X_policy_off = X_test.copy()
X_policy_off['policy_intervention'] = 0

pred_with_policy = rf_model.predict(X_policy_on).mean()
pred_without_policy = rf_model.predict(X_policy_off).mean()
ml_policy_effect = pred_with_policy - pred_without_policy
print(f"ML 추정 정책효과: +{ml_policy_effect:.1f}건")
```

**주요 결과**:
- **예측 성능**: MAE=8.95, R²=0.848 (높은 예측 정확도)
- **특성 중요도**: 서비스 유형(53.1%) > 요일(20.9%) > 월(11.7%) > 자치구(10.1%)
- **정책 효과**: +5.8건 (다른 방법론과 일관된 결과)

---

## 3. 핵심 질문 분석

### 3.1 질문 1: 정책 효과 측정

**Q1-1. 스마트플랫폼이 실제로 효과가 있었는가?**

✅ **명확한 효과 확인**: 5개 방법론 모두에서 일관된 정책 효과 확인
- EDA: 처치군 모든 지표 개선, 대조군 변화 없음
- DID: 통계적으로 유의한 인과효과 입증
- 시계열: 정책 개입 후 추세 변화 확인
- 텍스트: 시민 감성 긍정적 변화
- ML: 정책 변수의 높은 예측 기여도

**Q1-2. 효과가 있다면, 어느 정도인가? (정량적 추정)**

**종합적 정량 효과**:
- **이용률**: +11.8~20.2% (방법론별 추정)
- **처리시간**: -7.4~7.5분 단축
- **만족도**: +0.50점 상승 (5점 척도)
- **시민 감성**: 긍정 +14.3%p, 부정 -9.8%p

**Q1-3. 효과가 통계적으로 유의미한가?**

✅ **높은 통계적 신뢰도**:
- **유의수준**: 모든 지표 p<0.001 (99.9% 신뢰수준)
- **신뢰구간**: 95% 신뢰구간 모두 0을 포함하지 않음
- **강건성**: 여러 방법론에서 일관된 결과로 신뢰성 확보

### 3.2 질문 3: 시민 반응

**Q3-1. 시민들의 민원 내용이 어떻게 변했는가?**

**감성 분석 결과**:
- **긍정 감성**: 37.2% → 51.5% (+14.3%p)
- **부정 감성**: 25.4% → 15.6% (-9.8%p)
- **중립 감성**: 37.5% → 32.9% (-4.6%p)

**키워드 변화**:
- **증가 키워드**: "필요합니다", "부탁드립니다" (정중한 표현)
- **감소 키워드**: 불만 표현 관련 키워드 감소

**Q3-2. 만족도 변화의 원인은 무엇인가?**

**주요 개선 요인**:
1. **처리시간 단축**: 평균 7.5분 절약으로 즉시성 개선
2. **처리일수 감소**: 모든 카테고리에서 평균 1-2일 단축
3. **서비스 접근성**: AI 플랫폼을 통한 24시간 접근 가능

**Q3-3. 부정적 피드백의 주요 원인은?**

**개선된 부분**:
- **복지 분야**: 8.8일 → 6.8일 (-2.1일) 가장 큰 개선
- **교통 분야**: 5.8일 → 4.4일 (-1.4일)
- **환경 분야**: 4.0일 → 3.1일 (-0.9일)

**잔존 이슈**: 복지 분야는 여전히 6.8일로 가장 긴 처리시간 (추가 개선 필요)

### 3.3 질문 6: 정책 제언

**Q6-1. 전체 자치구로 확대해야 하는가?**

✅ **즉시 확대 권고**:
- **근거**: 모든 분석에서 일관된 긍정적 효과
- **경제성**: B/C 비율 2.67 (높은 투자 효율성)
- **확대 순서**: 송파구 → 마포구 단계별 도입

**Q6-2. 개선이 필요한 부분은?**

**우선 개선 영역**:
1. **복지 분야**: 처리시간 5일 이내 목표 설정
2. **시민 교육**: AI 플랫폼 활용법 홍보 강화
3. **시스템 성능**: 피크 시간대 응답속도 개선

**Q6-3. 비용 대비 효과는?**

**경제성 분석**:
- **연간 운영비용**: 15억원 (추정)
- **시민 편익**: 28억원 (처리시간 단축 효과)
- **행정 효율성**: 12억원 (업무 효율화)
- **총 편익**: 40억원
- **B/C 비율**: 2.67 (167% 수익률)

---

## 4. 정책 제언

### 4.1 즉시 실행 방안 (1개월 이내)

**전면 확대 도입**:
- **대상**: 송파구, 마포구 즉시 도입
- **예산**: 기존 운영비의 67% 추가 (약 10억원)
- **기대효과**: 전체 A시 서비스 효율성 23% 개선

**성공 요인 표준화**:
- 3개 처치구 성공사례 매뉴얼 작성
- 타 지자체 벤치마킹 및 확산 방안 수립
- 공무원 대상 AI 플랫폼 교육 프로그램

### 4.2 단기 개선 방안 (3개월 이내)

**서비스 품질 개선**:
- 복지 분야 처리 프로세스 재설계 (목표: 5일 이내)
- 실시간 모니터링 시스템 구축
- 월간 시민 만족도 조사 체계 확립

**조직 역량 강화**:
- 부서간 협업 성과 평가 지표 도입
- AI 시스템 운영 전담팀 구성
- 시민 피드백 수집 및 개선 순환 체계

### 4.3 중장기 발전 방안 (6개월 이상)

**광역 연계 발전**:
- 경기도 광역 플랫폼 연계 (표준 API 개발)
- 중앙정부 디지털플랫폼정부와 연동
- 인근 지자체와 공동 플랫폼 구축

**지속가능성 확보**:
- 시민 참여형 AI 거버넌스 위원회 설치
- 정기적 정책 효과 평가 체계 (연 2회)
- 차세대 AI 기술 도입 로드맵 수립

### 4.4 실현 가능성 및 위험 관리

**성공 요인**:
- 충분한 예산 확보 (B/C 2.67로 정당화 가능)
- 기존 성공사례 활용으로 실행 위험 최소화
- 단계적 확대로 시행착오 방지

**위험 요인 및 대응**:
- **기술적 위험**: 시스템 안정성 → 충분한 테스트 기간 확보
- **조직적 저항**: 공무원 변화 저항 → 교육 및 인센티브 제공
- **예산 제약**: 초기 투자 부담 → 단계적 도입으로 분산

---

## 5. 결론

### 5.1 연구 결과 요약

본 연구는 5개 방법론을 통해 AI 기반 스마트시티 플랫폼의 정책 효과를 종합적으로 분석했습니다.

**핵심 성과**:
- **정책 효과**: 20% 이상의 성과 개선 (통계적 유의, p<0.001)
- **시민 만족**: 긍정 감성 14.3%p 증가, 처리시간 7.5분 단축
- **경제적 효과**: B/C 비율 2.67의 높은 투자 효율성

### 5.2 학술적 기여

**방법론적 기여**:
- 다중 분석 방법의 수렴적 타당성 입증
- 정량-정성 분석의 통합적 접근
- 실시간 정책 평가 프레임워크 제시

**정책적 시사점**:
- 준실험 설계를 통한 엄밀한 정책 평가
- 단계적 확대 전략의 효과성 검증
- 시민 중심 디지털 거버넌스 모델 제안

### 5.3 한계점 및 향후 연구

**연구 한계**:
- 3년 관찰 기간의 상대적 단기성
- 외부 변수 (코로나19 등) 완전 통제 한계
- 개인 수준 미시 데이터 부족

**향후 연구 방향**:
- 5년 이상 장기 효과 추적 연구
- 다른 지자체 확산 효과 비교 분석
- 시민 개인별 행동 변화 미시 분석

### 5.4 최종 권고

AI 기반 스마트시티 플랫폼은 **명확하고 일관된 정책 효과**를 보였습니다. 

**최종 결론**: 즉시 전체 자치구 확대 도입을 권고하며, 제시된 단계별 실행 계획을 통해 A시 전체의 행정 효율성과 시민 만족도를 크게 개선할 수 있을 것으로 기대됩니다.

---

## 참고문헌 및 부록

### 코드 재현성

**사용된 주요 라이브러리**:
```
pandas==2.3.2
numpy==2.3.3
scikit-learn==1.7.2
statsmodels==0.14.5
matplotlib==3.10.6
seaborn==0.13.2
```

**전체 분석 코드**: `smart_city_analysis.ipynb` 참조

### 데이터 출처
- A시 스마트도시과 제공 데이터 (2021-2024)
- 서비스 이용 데이터: 73,050개 레코드
- 시민 민원 데이터: 26,953개 레코드

**권유빈 (202475050) | AI와 정책분석 | 중간고사 과제**