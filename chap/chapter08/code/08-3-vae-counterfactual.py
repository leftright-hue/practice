# 15-3-vae-counterfactual.py
# VAE 기반 반사실적(Counterfactual) 정책 분석 실습
# 데이터: practice/chapter15/data/15-3-policy-individual-data.csv
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 재현성을 위한 시드 설정
torch.manual_seed(42)
np.random.seed(42)

class VAECounterfactual(nn.Module):
    def __init__(self, input_dim, latent_dim, causal_order):
        super(VAECounterfactual, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.causal_order = causal_order
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2) # Mean & LogVar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def generate_counterfactual(self, x_factual, policy_intervention):
        """
        반사실적 샘플 생성
        x_factual: 실제 관측 데이터
        policy_intervention: 정책 변수에 대한 개입 값 (Scalar)
        """
        # 1. 실제 데이터를 잠재 공간으로 매핑 (인코딩)
        h = self.encoder(x_factual)
        mu, logvar = h.chunk(2, dim=-1)
        
        # 노이즈 고정 (같은 사람임을 보장하기 위해)
        # 여기서는 평균값(mu)을 그대로 사용하거나, 
        # 재구성 시 사용된 eps를 저장했다가 다시 쓸 수 있음.
        # 간단히 mu + noise로 샘플링
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        # 2. 인과 순서에 따라 정책 변수 조작 (Latent Space Manipulation)
        # 주의: 단순 VAE는 Latent 차원과 입력 변수가 1:1 대응되지 않음.
        # 본 예제에서는 'DirectLiNGAM + VAE'의 개념을 단순화하여,
        # Latent Space의 특정 차원이 정책 변수와 강하게 연관되어 있다고 가정하거나,
        # Decoder 입력 전 z를 수정하는 대신, 
        # 여기서는 문서의 예시 코드처럼 z 자체를 수정하는 방식을 따름.
        # 하지만 실제로는 z의 특정 차원이 '정책'을 의미하도록 학습되어야 함 (Disentanglement).
        
        # 편의상 Latent Dim의 첫 번째 차원이 정책과 관련있다고 가정하고 수정하거나,
        # 또는 문서의 코드처럼 'causal_order'를 이용해 z를 수정하는 로직을 구현.
        # 문서 코드: z[:, policy_idx] = policy_intervention
        # 이 코드가 작동하려면 latent_dim >= input_dim 이거나, 
        # 구조적 VAE여야 함. 여기서는 일반 VAE이므로, 
        # 개념적으로 z를 수정하는 것이 아니라, 
        # x_factual에서 정책 변수만 바꾼 뒤 다시 인코딩? -> 아님.
        
        # 문서의 의도: Latent z가 인과적 요인을 나타냄.
        # 여기서는 policy_idx에 해당하는 z 값을 강제로 변경.
        try:
            policy_idx = self.causal_order.index('policy')
            # z의 차원이 충분하다고 가정
            if policy_idx < self.latent_dim:
                z[:, policy_idx] = policy_intervention
        except ValueError:
            pass

        # 3. 디코딩하여 반사실적 데이터 생성
        x_counterfactual = self.decoder(z)
        return x_counterfactual

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.MSELoss(reduction='sum')(recon_x, x)
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def main():
    print(">>> VAE 반사실적 분석 실습 시작")

    # 1. 데이터 로드
    # 데이터: practice/chapter15/data/15-3-policy-individual-data.csv
    # 변수: [교육수준, 소득, 정책수혜여부(policy), 고용상태]
    # 인과: 교육 -> 소득 -> 정책 -> 고용
    data_path = Path(__file__).parent / '../data/08-3-policy-individual-data.csv'
    print(f"[데이터 로드: {data_path.name}]")

    df = pd.read_csv(data_path)
    print(f"  샘플 수: {len(df)}")
    print(f"  변수: {list(df.columns)}")

    # NumPy 배열로 변환
    data = df[['education', 'income', 'policy', 'employment_score']].values
    policy = df['policy'].values
    data_tensor = torch.FloatTensor(data)
    
    # 정규화 (Standard Scaling)
    mean = data_tensor.mean(dim=0)
    std = data_tensor.std(dim=0)
    data_norm = (data_tensor - mean) / std
    
    # 2. 모델 학습
    input_dim = 4
    latent_dim = 4 # 인과 변수 개수와 맞춤
    causal_order = ['education', 'income', 'policy', 'employment']
    
    model = VAECounterfactual(input_dim, latent_dim, causal_order)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(">>> 모델 학습 중 (Epochs: 100)...")
    for epoch in range(100):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data_norm)
        loss = loss_function(recon_x, data_norm, mu, logvar)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
    # 3. 반사실적 분석 수행
    print(">>> 반사실적 시나리오 분석")
    # 특정 샘플 선택 (정책 수혜를 받지 못한 사람: policy=0)
    policy_0_indices = np.where(policy == 0)[0]
    if len(policy_0_indices) == 0:
        # policy=0인 샘플이 없으면 첫 번째 샘플을 사용하고 강제로 policy=0으로 설정
        target_idx = 0
        x_factual = data_norm[target_idx:target_idx+1].clone()
        x_factual[0, 2] = (0 - mean[2]) / std[2]  # policy를 0으로 설정
        print(f"    [Note] No policy=0 samples found, using sample {target_idx} with policy forced to 0")
    else:
        target_idx = policy_0_indices[0]
        x_factual = data_norm[target_idx:target_idx+1]
    
    print(f"    [Fact] 실제 상태 (Normalized): {x_factual.detach().numpy()}")
    
    # 시나리오: 만약 이 사람이 정책 수혜를 받았다면? (policy=1에 해당하는 값으로 개입)
    # 정규화된 공간에서의 1값 계산
    policy_val_1 = (1 - mean[2]) / std[2]
    
    # Latent Space에서의 개입 (개념적 구현)
    # 실제로는 Latent Disentanglement가 선행되어야 정확함
    x_counter = model.generate_counterfactual(x_factual, policy_val_1)
    
    print(f"    [Counterfactual] 반사실 상태: {x_counter.detach().numpy()}")
    
    # 결과 해석 (고용 점수 변화 확인)
    emp_idx = 3
    fact_emp = x_factual[0, emp_idx].item() * std[emp_idx] + mean[emp_idx]
    counter_emp = x_counter[0, emp_idx].item() * std[emp_idx] + mean[emp_idx]
    
    print(f"\n    실제 고용 점수: {fact_emp:.2f}")
    print(f"    반사실 고용 점수 (정책 수혜 시): {counter_emp:.2f}")
    print(f"    효과 (Difference): {counter_emp - fact_emp:.2f}")
    
    # 시각화
    plt.figure(figsize=(8, 5))
    labels = ['Education', 'Income', 'Policy', 'Employment']
    x = np.arange(len(labels))
    width = 0.35
    
    # Denormalize for plotting
    fact_vals = x_factual[0].detach().numpy() * std.numpy() + mean.numpy()
    counter_vals = x_counter[0].detach().numpy() * std.numpy() + mean.numpy()
    
    plt.bar(x - width/2, fact_vals, width, label='Factual (No Policy)')
    plt.bar(x + width/2, counter_vals, width, label='Counterfactual (Policy)')
    
    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('Counterfactual Analysis using VAE')
    plt.legend()
    
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    # plt.savefig() 생략
    print(f">>> 반사실적 분석 완료")

if __name__ == "__main__":
    main()
