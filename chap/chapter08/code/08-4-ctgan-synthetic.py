import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import platform
import sys
import io

# 한글 출력을 위한 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# 한글 폰트 설정 (운영체제별)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_welfare_data():
    """
    복지 수혜자 데이터 로드
    데이터: practice/chapter15/data/15-4-welfare-data.csv
    """
    data_path = Path(__file__).parent / '../data/08-4-welfare-data.csv'
    df = pd.read_csv(data_path)
    print(f"[데이터 로드: {data_path.name}]")
    print(f"  샘플 수: {len(df)}")
    print(f"  변수: {list(df.columns)}")
    return df

# === Simple GAN Implementation (Fallback) ===
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class SimpleTabularGAN:
    """
    CTGAN 라이브러리가 없을 경우를 위한 간단한 GAN 구현
    """
    def __init__(self, input_dim, latent_dim=10):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()

    def fit(self, data, epochs=300):
        data_tensor = torch.FloatTensor(data.values)
        batch_size = 64
        
        print("SimpleGAN 학습 시작...")
        for epoch in range(epochs):
            for i in range(0, len(data_tensor), batch_size):
                real_data = data_tensor[i:i+batch_size]
                current_batch_size = real_data.size(0)
                
                # 1. 판별자(Discriminator) 학습
                self.d_optimizer.zero_grad()
                
                real_labels = torch.ones(current_batch_size, 1)
                fake_labels = torch.zeros(current_batch_size, 1)
                
                outputs = self.discriminator(real_data)
                d_loss_real = self.criterion(outputs, real_labels)
                
                z = torch.randn(current_batch_size, self.latent_dim)
                fake_data = self.generator(z)
                outputs = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(outputs, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # 2. 생성자(Generator) 학습
                self.g_optimizer.zero_grad()
                z = torch.randn(current_batch_size, self.latent_dim)
                fake_data = self.generator(z)
                outputs = self.discriminator(fake_data)
                
                g_loss = self.criterion(outputs, real_labels) # 가짜를 진짜로 속여야 함
                g_loss.backward()
                self.g_optimizer.step()
                
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    def sample(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            generated_data = self.generator(z).numpy()
        return pd.DataFrame(generated_data, columns=['Age', 'Income', 'Region', 'Benefit'])

def evaluate_similarity(real_df, synthetic_df):
    print("\n=== 데이터 통계 비교 ===")
    print(f"{'항목':<10} | {'실제 데이터 (Mean)':<20} | {'합성 데이터 (Mean)':<20}")
    print("-" * 60)
    for col in real_df.columns:
        print(f"{col:<10} | {real_df[col].mean():<20.2f} | {synthetic_df[col].mean():<20.2f}")
        
    print("\n=== 상관계수 행렬 비교 ===")
    print("실제 데이터 상관계수:")
    print(real_df.corr().round(2))
    print("\n합성 데이터 상관계수:")
    print(synthetic_df.corr().round(2))

if __name__ == "__main__":
    print("=== 15-4. CTGAN 기반 합성 데이터 생성 예제 ===")
    
    # 1. 데이터 로드
    real_data = load_welfare_data()
    print(f"실제 데이터 로드 완료: {len(real_data)} 샘플")
    
    # 2. 모델 학습 (CTGAN 시도 -> 실패 시 SimpleGAN)
    try:
        from ctgan import CTGAN
        print("CTGAN 라이브러리 감지됨. CTGAN 모델을 사용합니다.")
        model = CTGAN(epochs=300, verbose=True)
        model.fit(real_data, discrete_columns=['Region'])
    except ImportError:
        print("CTGAN 라이브러리가 설치되지 않았습니다. (pip install ctgan)")
        print("교육용 SimpleTabularGAN 모델로 대체합니다.")
        # SimpleGAN을 위해 데이터 정규화 필요 (간소화를 위해 생략하거나 간단히 수행)
        # 여기서는 원본 스케일 그대로 학습 (성능은 떨어질 수 있음)
        model = SimpleTabularGAN(input_dim=4)
        model.fit(real_data)
        
    # 3. 합성 데이터 생성
    synthetic_data = model.sample(1000)
    print(f"합성 데이터 생성 완료: {len(synthetic_data)} 샘플")
    
    # 4. 결과 평가
    evaluate_similarity(real_data, synthetic_data)
    
    # 5. 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(real_data['Income'], real_data['Benefit'], alpha=0.5, label='실제')
    plt.title("실제 데이터: 소득 vs 수급액")
    plt.xlabel("소득")
    plt.ylabel("수급액")
    
    plt.subplot(1, 2, 2)
    plt.scatter(synthetic_data['Income'], synthetic_data['Benefit'], alpha=0.5, color='orange', label='합성')
    plt.title("합성 데이터: 소득 vs 수급액")
    plt.xlabel("소득")
    plt.ylabel("수급액")
    
    plt.tight_layout()
    # 이미지 저장하지 않음 (CLAUDE.md 원칙)
    # plt.savefig() 생략
    print("=== CTGAN 합성 데이터 생성 완료 ===")
