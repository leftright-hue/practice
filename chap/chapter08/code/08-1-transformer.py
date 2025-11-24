#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chapter 15: Transformer 기반 다변량 정책 시계열 예측
정책 시나리오: 정부 혁신성장 정책 효과 분석
데이터: practice/chapter15/data/15-1-policy-timeseries.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class PositionalEncoding(nn.Module):
    """위치 인코딩: 시퀀스 순서 정보 주입"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultivariatePolicyTransformer(nn.Module):
    """
    다변량 정책 시계열 예측 Transformer
    입력: 3개 정책 변수 × 50개월 (R&D, 기업지원, 세제혜택)
    출력: 3개 결과 변수 (특허, 고용, GDP)
    """
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2, output_dim=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src shape: [seq_len, batch_size, input_dim]
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 마지막 시점 출력으로 3개 변수 예측
        prediction = self.decoder(output[-1, :, :])
        return prediction, output

def load_data():
    """다변량 정책 시계열 데이터 로드"""
    data_path = Path(__file__).parent / '../data/08-1-policy-timeseries.csv'

    print(f"[데이터 로드: {data_path}]")
    df = pd.read_csv(data_path)

    print(f"  로드 완료: {len(df)} 샘플")

    # 정책 변수 3개 × 50개월 = 150개 컬럼
    seq_len = 50
    rnd_cols = [f'rnd_t_{i}' for i in range(seq_len)]
    firm_cols = [f'firm_t_{i}' for i in range(seq_len)]
    tax_cols = [f'tax_t_{i}' for i in range(seq_len)]

    # (n_samples, seq_len, 3)
    X = np.stack([
        df[rnd_cols].values,
        df[firm_cols].values,
        df[tax_cols].values
    ], axis=-1)

    # 결과 변수 3개
    y = df[['patents', 'employment', 'gdp_growth']].values

    # Tensor 변환
    X_tensor = torch.FloatTensor(X)  # (n_samples, seq_len, 3)
    y_tensor = torch.FloatTensor(y)  # (n_samples, 3)

    print(f"  입력 shape: {X_tensor.shape} (샘플, 50개월, 3변수)")
    print(f"  출력 shape: {y_tensor.shape} (샘플, 3결과)")

    return X_tensor, y_tensor

def train_model():
    print("=" * 80)
    print("15-1. Transformer 기반 다변량 정책 시계열 예측")
    print("정책 시나리오: 정부 혁신성장 정책 효과 분석")
    print("=" * 80)

    # 데이터 로드
    print("\n[1단계] 데이터 로드")
    X, y = load_data()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 초기화
    print("\n[2단계] Transformer 모델 초기화")
    model = MultivariatePolicyTransformer(input_dim=3, d_model=64, nhead=4, num_layers=2, output_dim=3)
    print(f"  입력: 3개 정책 변수 (R&D, 기업지원, 세제혜택)")
    print(f"  출력: 3개 결과 변수 (특허, 고용, GDP)")
    print(f"  모델: d_model=64, nhead=4, num_layers=2")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    print("\n[3단계] 모델 학습 (50 에폭)")
    model.train()
    loss_history = []

    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            # Transformer: [seq_len, batch_size, feature_dim]
            batch_X = batch_X.permute(1, 0, 2)

            optimizer.zero_grad()
            prediction, _ = model(batch_X)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, MSE Loss: {avg_loss:.4f}")

    print(f"\n  최종 MSE Loss: {loss_history[-1]:.4f}")

    return model, loss_history, X, y

def evaluate_model(model, X, y):
    """모델 성능 평가"""
    print("\n[4단계] 모델 평가")
    model.eval()

    with torch.no_grad():
        predictions, _ = model(X.permute(1, 0, 2))

    # 변수별 성능 평가
    y_np = y.numpy()
    pred_np = predictions.numpy()

    results = {}
    var_names = ['특허', '고용', 'GDP']

    print(f"\n  변수별 예측 성능:")
    for i, name in enumerate(var_names):
        mae = np.mean(np.abs(y_np[:, i] - pred_np[:, i]))
        rmse = np.sqrt(np.mean((y_np[:, i] - pred_np[:, i]) ** 2))

        # R² 계산
        ss_res = np.sum((y_np[:, i] - pred_np[:, i]) ** 2)
        ss_tot = np.sum((y_np[:, i] - np.mean(y_np[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        results[name] = {
            'mean_true': y_np[:, i].mean(),
            'mean_pred': pred_np[:, i].mean(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        print(f"\n    [{name}]")
        print(f"      실제 평균: {results[name]['mean_true']:.2f}")
        print(f"      예측 평균: {results[name]['mean_pred']:.2f}")
        print(f"      MAE: {mae:.2f}")
        print(f"      RMSE: {rmse:.2f}")
        print(f"      R²: {r2:.3f}")

    # 샘플 예측 확인
    print(f"\n  예측 샘플 (처음 3개):")
    for i in range(3):
        print(f"    샘플 {i}:")
        print(f"      특허 - 실제: {y_np[i,0]:.1f}건, 예측: {pred_np[i,0]:.1f}건")
        print(f"      고용 - 실제: {y_np[i,1]:.1f}명, 예측: {pred_np[i,1]:.1f}명")
        print(f"      GDP  - 실제: {y_np[i,2]:.2f}%, 예측: {pred_np[i,2]:.2f}%")

    return results

if __name__ == "__main__":
    # 모델 학습
    trained_model, losses, X_data, y_data = train_model()

    # 모델 평가
    results = evaluate_model(trained_model, X_data, y_data)

    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)
    print("\n※ 이 코드는 교육 목적의 예제입니다.")
    print("  실제 정책 분석에서는 교차 검증, 하이퍼파라미터 튜닝,")
    print("  Attention 가중치 시각화 등의 추가 분석이 필요합니다.")
