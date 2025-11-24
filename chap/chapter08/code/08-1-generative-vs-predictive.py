# 15-1-generative-vs-predictive.py
# 생성형 AI(Generative) vs 예측 AI(Predictive) 비교 실습
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error

# 재현성을 위한 시드 설정
np.random.seed(42)

def generate_data(n_samples=200):
    """
    가상의 정책 데이터 생성
    X: 정책 변수 (예: 예산 투입량)
    y: 정책 효과 (예: 고용률 증가분)
    """
    X = np.random.rand(n_samples, 1) * 10  # 0 ~ 10
    # 비선형 관계 + 노이즈
    y = 2 * np.sin(X).ravel() + 0.5 * X.ravel() + np.random.normal(0, 0.5, n_samples)
    return X, y

def predictive_approach(X, y, X_test):
    """
    예측 AI 접근법 (Predictive AI)
    - 조건부 확률 p(y|x)를 추정 (주로 평균값 예측)
    - 예: 선형 회귀
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    return y_pred, model

def generative_approach(X, y, X_test):
    """
    생성형 AI 접근법 (Generative AI)
    - 결합 확률 분포 p(x,y)를 학습하여 새로운 샘플 생성 가능
    - 예: Gaussian Mixture Model (GMM)을 이용한 밀도 추정 및 샘플링
    """
    # 데이터를 결합하여 (X, y) 벡터 생성
    data = np.hstack([X, y.reshape(-1, 1)])
    
    # GMM 학습
    gmm = GaussianMixture(n_components=5, random_state=42)
    gmm.fit(data)
    
    # 생성(Sampling): X_test 조건부 생성이 아니므로,
    # 여기서는 모델이 학습한 분포에서 무작위 샘플링을 통해 데이터 생성 능력을 보여줌
    samples, _ = gmm.sample(len(X_test))
    X_generated = samples[:, 0].reshape(-1, 1)
    y_generated = samples[:, 1]

    return X_generated, y_generated, gmm

def main():
    print(">>> 생성형 AI vs 예측 AI 비교 실습 시작")

    # 1. 데이터 생성
    X, y = generate_data()
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # 2. 예측 AI (Linear Regression)
    print(">>> [Predictive AI] 선형 회귀 학습 중...")
    y_pred_lin, lin_model = predictive_approach(X, y, X_test)
    
    # 3. 생성형 AI (GMM)
    print(">>> [Generative AI] GMM 분포 학습 중...")
    X_gen, y_gen, gmm_model = generative_approach(X, y, X_test)
    
    # 4. 결과 시각화
    plt.figure(figsize=(12, 5))
    
    # 예측 AI 결과
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X_test, y_pred_lin, color='red', linewidth=2, label='Predictive (Linear)')
    plt.title('Predictive AI: p(y|x)')
    plt.xlabel('Policy Input (X)')
    plt.ylabel('Effect (y)')
    plt.legend()
    
    # 생성형 AI 결과
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.3, color='gray', label='Original Data')
    plt.scatter(X_gen, y_gen, alpha=0.5, color='green', label='Generated Data (GMM)')
    plt.title('Generative AI: p(x,y)')
    plt.xlabel('Policy Input (X)')
    plt.ylabel('Effect (y)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    print(">>> 시각화 완료 (그래프는 화면에 표시됨)")
    
    print("\n[결론]")
    print("1. 예측 AI는 주어진 X에 대해 가장 그럴듯한 y값(평균) 하나를 예측합니다.")
    print("2. 생성형 AI는 데이터의 전체 분포를 학습하여, 새로운 (X, y) 쌍을 무한히 생성할 수 있습니다.")

if __name__ == "__main__":
    main()
