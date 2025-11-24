import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class ConditionalDiffusionForecaster(nn.Module):
    """
    시계열 예측을 위한 조건부 확산 모델 (Conditional DDPM)
    과거 데이터(x_past)를 조건으로 미래 데이터(x_future)를 생성
    """
    def __init__(self, past_dim, future_dim, hidden_dim, T=100):
        super().__init__()
        self.T = T
        self.past_dim = past_dim
        self.future_dim = future_dim
        
        # 노이즈 스케줄 설정 (Linear schedule)
        self.betas = torch.linspace(1e-4, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # 노이즈 예측 네트워크 (간소화된 MLP 구조)
        # 입력: 노이즈 섞인 미래(x_t) + 시간 임베딩(t) + 과거 조건(x_past)
        self.net = nn.Sequential(
            nn.Linear(future_dim + 1 + past_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_dim)
        )

    def forward(self, x_future, x_past):
        """
        학습 단계: 랜덤한 시점 t를 샘플링하고, 추가된 노이즈를 예측
        """
        batch_size = x_future.size(0)
        t = torch.randint(0, self.T, (batch_size, 1)).float()
        
        # 노이즈 생성
        noise = torch.randn_like(x_future)
        
        # x_t 생성 (Reparameterization trick)
        # alpha_bar_t를 배치 크기에 맞게 확장
        alpha_bar_t = self.alpha_bars[t.long().view(-1)].view(-1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_future + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 네트워크 입력 구성
        model_input = torch.cat([x_t, t, x_past], dim=-1)
        
        # 노이즈 예측
        predicted_noise = self.net(model_input)
        
        # 손실 계산 (MSE)
        return nn.MSELoss()(predicted_noise, noise)

    def sample(self, x_past, n_samples=100):
        """
        추론 단계: 노이즈에서 시작하여 순차적으로 노이즈를 제거 (Denoising)
        """
        with torch.no_grad():
            # x_past를 n_samples만큼 복제
            x_past_expanded = x_past.repeat(n_samples, 1)

            # 초기 노이즈 (Standard Gaussian)
            x_t = torch.randn(n_samples, self.future_dim)
            
            for t_idx in reversed(range(self.T)):
                t = torch.full((n_samples, 1), t_idx).float()
                
                # 현재 시점의 alpha 값들
                beta_t = self.betas[t_idx]
                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alpha_bars[t_idx]
                
                # 노이즈 예측
                model_input = torch.cat([x_t, t, x_past_expanded], dim=-1)
                predicted_noise = self.net(model_input)
                
                # Denoising step (DDPM 수식)
                if t_idx > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
                
            return x_t

def load_data():
    """
    다변량 정책 시계열 데이터 로드
    과거: 3개 정책 변수 × 20시점 (R&D, 기업지원, 세제혜택)
    미래: 3개 결과 변수 × 10시점 (특허, 고용, GDP)
    """
    data_path = Path(__file__).parent / '../data/08-2-diffusion-timeseries.csv'

    print(f"[데이터 로드: {data_path}]")
    df = pd.read_csv(data_path)

    print(f"  로드 완료: {len(df)} 샘플")
    print(f"  컬럼 수: {len(df.columns)}")

    # 과거 정책 변수 (3변수 × 20시점 = 60차원)
    past_len = 20
    past_rnd_cols = [f'past_rnd_t{i}' for i in range(past_len)]
    past_firm_cols = [f'past_firm_t{i}' for i in range(past_len)]
    past_tax_cols = [f'past_tax_t{i}' for i in range(past_len)]

    # (n_samples, 60) - RND, FIRM, TAX를 일렬로 연결
    X_past = np.concatenate([
        df[past_rnd_cols].values,
        df[past_firm_cols].values,
        df[past_tax_cols].values
    ], axis=1)

    # 미래 결과 변수 (3변수 × 10시점 = 30차원)
    future_len = 10
    future_patents_cols = [f'future_patents_t{i}' for i in range(future_len)]
    future_employment_cols = [f'future_employment_t{i}' for i in range(future_len)]
    future_gdp_cols = [f'future_gdp_t{i}' for i in range(future_len)]

    # (n_samples, 30) - PATENTS, EMPLOYMENT, GDP를 일렬로 연결
    X_future = np.concatenate([
        df[future_patents_cols].values,
        df[future_employment_cols].values,
        df[future_gdp_cols].values
    ], axis=1)

    # Tensor로 변환
    X_past_tensor = torch.FloatTensor(X_past)
    X_future_tensor = torch.FloatTensor(X_future)

    print(f"  과거 정책 shape: {X_past_tensor.shape} (샘플, 60차원: 3변수×20시점)")
    print(f"  미래 결과 shape: {X_future_tensor.shape} (샘플, 30차원: 3변수×10시점)")

    return X_past_tensor, X_future_tensor

def train_diffusion():
    print("=" * 80)
    print("15-2. Diffusion 모델 기반 조건부 시계열 예측")
    print("=" * 80)

    # 데이터 로드
    print("\n[1단계] 데이터 로드")
    X_past, X_future = load_data()

    dataset = TensorDataset(X_past, X_future)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 모델 초기화
    print("\n[2단계] Diffusion 모델 초기화")
    past_dim = X_past.shape[1]  # 60 (3변수 × 20시점)
    future_dim = X_future.shape[1]  # 30 (3변수 × 10시점)
    model = ConditionalDiffusionForecaster(past_dim=past_dim, future_dim=future_dim, hidden_dim=256, T=100)
    print(f"  과거 정책 차원: {past_dim} (R&D+기업지원+세제혜택 × 20시점)")
    print(f"  미래 결과 차원: {future_dim} (특허+고용+GDP × 10시점)")
    print(f"  Hidden 차원: 256, 노이즈 스텝: 100")

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 학습
    print("\n[3단계] 모델 학습 (100 에폭)")
    model.train()
    loss_history = []
    for epoch in range(100):
        total_loss = 0
        for batch_past, batch_future in dataloader:
            optimizer.zero_grad()
            loss = model(batch_future, batch_past)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100, Loss: {avg_loss:.4f}")

    print(f"\n  최종 Loss: {loss_history[-1]:.4f}")

    return model, X_past

def evaluate_uncertainty(model, X_past):
    """
    불확실성 정량화 평가
    """
    print("\n[4단계] 불확실성 정량화")
    model.eval()

    # 테스트: 첫 번째 샘플의 과거 값을 조건으로 사용
    test_past = X_past[0:1]  # shape: (1, 20)

    # 1000개의 가능한 미래 경로 샘플링
    print(f"  샘플링 시작: 1000개 미래 시나리오 생성 중...")
    sampled_futures = model.sample(test_past, n_samples=1000).numpy()  # shape: (1000, 10)

    # 시나리오 통계
    mean_forecast = sampled_futures.mean(axis=0)
    std_forecast = sampled_futures.std(axis=0)
    percentile_2_5 = pd.DataFrame(sampled_futures).quantile(0.025).values
    percentile_97_5 = pd.DataFrame(sampled_futures).quantile(0.975).values

    # 결과 출력
    print(f"\n  미래 10 시점 예측 요약:")
    for t in range(10):
        print(f"    시점 {t}: 평균={mean_forecast[t]:.4f}, 표준편차={std_forecast[t]:.4f}, "
              f"95% CI=[{percentile_2_5[t]:.4f}, {percentile_97_5[t]:.4f}]")

    # 대표 시나리오 추출
    best_case_idx = sampled_futures[:, -1].argmax()  # 마지막 시점 최대값
    worst_case_idx = sampled_futures[:, -1].argmin()  # 마지막 시점 최소값
    median_val = pd.Series(sampled_futures[:, -1]).median()
    median_idx = abs(sampled_futures[:, -1] - median_val).argmin()

    best_case = sampled_futures[best_case_idx]
    worst_case = sampled_futures[worst_case_idx]
    most_likely = sampled_futures[median_idx]

    print(f"\n  대표 시나리오:")
    print(f"    Best-case 최종값: {best_case[-1]:.4f}")
    print(f"    Worst-case 최종값: {worst_case[-1]:.4f}")
    print(f"    Most-likely 최종값: {most_likely[-1]:.4f}")

    return {
        'mean': mean_forecast,
        'std': std_forecast,
        'best_case': best_case,
        'worst_case': worst_case,
        'most_likely': most_likely,
        'ci_lower': percentile_2_5,
        'ci_upper': percentile_97_5
    }

if __name__ == "__main__":
    # 모델 학습
    trained_model, past_data = train_diffusion()

    # 불확실성 평가
    uncertainty_results = evaluate_uncertainty(trained_model, past_data)

    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)
    print("\n※ 이 코드는 교육 목적의 예제입니다.")
    print("  실제 정책 분석에서는 시나리오별 정책 대응 전략, ")
    print("  꼬리 위험(tail risk) 스트레스 테스트 등의 추가 분석이 필요합니다.")
