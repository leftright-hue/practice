"""
제12장: 복잡계 이론과 정책 시뮬레이션
12.1 창발성 시뮬레이션: 개별 에이전트의 단순 규칙이 복잡한 패턴 생성

이 코드는 단순한 국지적 상호작용 규칙이 어떻게 시스템 수준의
창발적 패턴을 만들어내는지 보여줍니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def simulate_emergence(n_agents=100, steps=50, threshold=0.3, seed=42):
    """
    창발성 시뮬레이션

    Parameters:
    -----------
    n_agents : int, 에이전트 수
    steps : int, 시뮬레이션 단계
    threshold : float, 상태 변화 임계값
    seed : int, 랜덤 시드

    Returns:
    --------
    history : list, 각 단계별 평균 상태값
    state_history : array, 전체 상태 이력
    """
    np.random.seed(seed)

    # 초기 상태: 무작위 (0 또는 1)
    states = np.random.choice([0, 1], n_agents)
    history = []
    state_history = []

    for step in range(steps):
        new_states = states.copy()

        # 각 에이전트의 상태 업데이트
        for i in range(n_agents):
            # 이웃 에이전트들의 상태 확인
            left_neighbor = states[(i-1) % n_agents]
            right_neighbor = states[(i+1) % n_agents]
            neighbors = [left_neighbor, right_neighbor]

            # 이웃들의 평균 상태가 임계값을 넘으면 활성화
            if np.mean(neighbors) > threshold:
                new_states[i] = 1
            else:
                new_states[i] = 0

        states = new_states
        history.append(np.mean(states))
        state_history.append(states.copy())

    return history, np.array(state_history)

def visualize_emergence(state_history, history):
    """창발 패턴 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 시공간 패턴 (Space-Time Diagram)
    ax1.imshow(state_history.T, cmap='binary', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Agent ID', fontsize=12)
    ax1.set_title('Emergent Spatiotemporal Pattern', fontsize=14, fontweight='bold')

    # 주석 추가 - x축 아래
    fig.text(0.25, 0.02,
             '※ 각 행(가로줄)은 개별 에이전트, 각 열(세로줄)은 시간 단계\n   흰색=비활성(0), 검은색=활성(1) 상태를 의미\n   시간이 지나면서 대부분 에이전트가 활성(검은색) 상태로 동기화됨',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 시스템 수준 행동
    ax2.plot(history, 'b-', linewidth=2, label='System State')
    ax2.fill_between(range(len(history)), 0, history, alpha=0.3)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Average State', fontsize=12)
    ax2.set_title('System-Level Behavior', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])

    # 주석 추가 - x축 아래
    fig.text(0.75, 0.02,
             f'※ 초기값={history[0]:.2f} → 최종값={history[-1]:.2f}\n   S자 곡선은 티핑 포인트 존재 의미: 임계 질량 도달 시 정책이 자발적으로 확산',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.subplots_adjust(bottom=0.15)
    plt.savefig('emergence_pattern.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_sensitivity(n_agents=100, steps=50):
    """임계값 변화에 대한 민감도 분석"""
    thresholds = np.linspace(0.1, 0.9, 9)
    final_states = []
    convergence_times = []

    for threshold in thresholds:
        history, _ = simulate_emergence(n_agents, steps, threshold)
        final_states.append(history[-1])

        # 수렴 시간 계산 (변화율이 0.01 이하가 되는 시점)
        convergence_time = steps
        for i in range(10, steps):
            if abs(history[i] - history[i-1]) < 0.01:
                convergence_time = i
                break
        convergence_times.append(convergence_time)

    # 결과 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(thresholds, final_states, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Final System State', fontsize=12)
    ax1.set_title('Threshold Sensitivity', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 임계점 표시
    ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.text(0.5, 0.5, '임계점', fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # 주석 추가 - x축 아래
    fig.text(0.25, 0.02,
             '※ 임계값 0.5에서 상전이 발생\n   임계값 < 0.5 → 완전 활성화 / 임계값 ≥ 0.5 → 완전 비활성화\n   정책 설계 시사점: 낮은 임계값이 확산에 유리',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax2.bar(thresholds, convergence_times, width=0.08, color='skyblue', edgecolor='navy')
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Convergence Time', fontsize=12)
    ax2.set_title('Time to Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 주석 추가 - x축 아래
    fig.text(0.75, 0.02,
             '※ 모든 임계값에서 약 10단계에 수렴\n   수렴 속도는 임계값과 무관\n   함의: 네트워크 구조가 속도 결정, 임계값은 방향만 결정',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))

    plt.subplots_adjust(bottom=0.15)
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return thresholds, final_states, convergence_times

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("복잡계 창발성 시뮬레이션")
    print("=" * 60)

    # 기본 시뮬레이션 실행
    print("\n1. 기본 창발 패턴 생성 중...")
    history, state_history = simulate_emergence(n_agents=100, steps=50, threshold=0.3)

    print(f"   - 초기 시스템 상태: {history[0]:.3f}")
    print(f"   - 최종 시스템 상태: {history[-1]:.3f}")
    print(f"   - 상태 변화 범위: {max(history) - min(history):.3f}")

    # 시각화
    print("\n2. 창발 패턴 시각화...")
    visualize_emergence(state_history, history)

    # 민감도 분석
    print("\n3. 임계값 민감도 분석 중...")
    thresholds, final_states, convergence_times = analyze_sensitivity()

    print("\n임계값별 분석 결과:")
    print("-" * 40)
    for t, fs, ct in zip(thresholds, final_states, convergence_times):
        print(f"임계값: {t:.1f} | 최종상태: {fs:.3f} | 수렴시간: {ct}")

    print("\n" + "=" * 60)
    print("시뮬레이션 완료!")
    print("생성된 파일: emergence_pattern.png, sensitivity_analysis.png")
    print("=" * 60)

if __name__ == "__main__":
    main()