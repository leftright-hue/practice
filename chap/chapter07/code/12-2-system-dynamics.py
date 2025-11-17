"""
제12장: 복잡계 이론과 정책 시뮬레이션
12.2 시스템 다이내믹스: 정책 피드백 시뮬레이션

한계기업 정책의 선순환/악순환 구조를 시스템 다이내믹스로 모델링합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class PolicyFeedbackModel:
    """정책 피드백 시스템 다이내믹스 모델"""

    def __init__(self, innovation_capacity='high'):
        """
        Parameters:
        -----------
        innovation_capacity : str, 'high' 또는 'low' (기업의 혁신 역량)
        """
        self.innovation_capacity = innovation_capacity

        # 모델 매개변수 설정
        if innovation_capacity == 'high':
            self.innovation_rate = 0.3  # 높은 혁신률
            self.efficiency_gain = 0.2  # 효율성 증가율
            self.dependency_rate = 0.05  # 낮은 의존도
        else:
            self.innovation_rate = 0.05  # 낮은 혁신률
            self.efficiency_gain = 0.02  # 낮은 효율성 증가율
            self.dependency_rate = 0.4  # 높은 의존도

    def dynamics(self, state, t, policy_support):
        """
        시스템 다이내믹스 미분방정식

        Parameters:
        -----------
        state : array, [기업건전성, 혁신수준, 경제효율성]
        t : float, 시간
        policy_support : float, 정책 지원 강도

        Returns:
        --------
        dstate_dt : array, 상태 변화율
        """
        health, innovation, efficiency = state

        # 피드백 루프 계산
        # 강화 루프: 혁신 → 건전성 → 효율성 → 혁신
        innovation_feedback = self.innovation_rate * health * (1 - innovation)

        # 균형 루프: 정책 지원 → 의존도 → 혁신 저해
        dependency_feedback = -self.dependency_rate * policy_support * innovation

        # 상태 방정식
        dhealth_dt = self.efficiency_gain * efficiency + policy_support - 0.1 * health
        dinnovation_dt = innovation_feedback + dependency_feedback
        defficiency_dt = 0.15 * innovation * health - 0.05 * efficiency

        return [dhealth_dt, dinnovation_dt, defficiency_dt]

    def simulate(self, initial_state, policy_support, time_points):
        """시뮬레이션 실행"""
        solution = odeint(self.dynamics, initial_state, time_points,
                         args=(policy_support,))
        return solution

def run_policy_comparison():
    """고혁신 vs 저혁신 기업의 정책 효과 비교"""

    # 시뮬레이션 설정
    time_points = np.linspace(0, 50, 200)
    initial_state = [0.3, 0.3, 0.3]  # 초기 상태
    policy_levels = [0.0, 0.2, 0.5, 0.8]  # 정책 지원 수준

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, innovation_capacity in enumerate(['high', 'low']):
        model = PolicyFeedbackModel(innovation_capacity)
        capacity_label = 'High Innovation' if innovation_capacity == 'high' else 'Low Innovation'

        for policy_support in policy_levels:
            solution = model.simulate(initial_state, policy_support, time_points)

            # 각 변수별 시계열 플롯
            axes[idx, 0].plot(time_points, solution[:, 0],
                            label=f'Policy={policy_support:.1f}')
            axes[idx, 1].plot(time_points, solution[:, 1],
                            label=f'Policy={policy_support:.1f}')
            axes[idx, 2].plot(time_points, solution[:, 2],
                            label=f'Policy={policy_support:.1f}')

        # 축 레이블 설정
        axes[idx, 0].set_title(f'{capacity_label}: Firm Health', fontsize=12)
        axes[idx, 1].set_title(f'{capacity_label}: Innovation Level', fontsize=12)
        axes[idx, 2].set_title(f'{capacity_label}: Economic Efficiency', fontsize=12)

        for j in range(3):
            axes[idx, j].set_xlabel('Time', fontsize=10)
            axes[idx, j].set_ylabel('Level', fontsize=10)
            axes[idx, j].legend(loc='best')
            axes[idx, j].grid(True, alpha=0.3)
            axes[idx, j].set_ylim([0, 1])

    plt.suptitle('Policy Feedback Dynamics: High vs Low Innovation Firms',
                fontsize=14, fontweight='bold')

    # 주석 추가
    fig.text(0.5, 0.02,
             '※ 고혁신 기업(상단): 정책 지원 증가 시 건전성/혁신/효율성 모두 상승 (선순환)\n'
             '   저혁신 기업(하단): 정책 지원 증가 시 의존도 증가로 혁신 저하 (악순환)\n'
             '   함의: 동일한 정책도 기업 역량에 따라 정반대 효과 발생',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.subplots_adjust(bottom=0.12)
    plt.savefig('policy_feedback_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feedback_loops():
    """피드백 루프 구조 분석"""

    # 정책 지원 범위
    policy_range = np.linspace(0, 1, 50)
    final_states_high = []
    final_states_low = []

    time_points = np.linspace(0, 100, 500)
    initial_state = [0.3, 0.3, 0.3]

    for policy in policy_range:
        # 고혁신 기업
        model_high = PolicyFeedbackModel('high')
        solution_high = model_high.simulate(initial_state, policy, time_points)
        final_states_high.append(solution_high[-1])

        # 저혁신 기업
        model_low = PolicyFeedbackModel('low')
        solution_low = model_low.simulate(initial_state, policy, time_points)
        final_states_low.append(solution_low[-1])

    final_states_high = np.array(final_states_high)
    final_states_low = np.array(final_states_low)

    # 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variables = ['Firm Health', 'Innovation', 'Efficiency']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, (var, color) in enumerate(zip(variables, colors)):
        axes[i].plot(policy_range, final_states_high[:, i], '-',
                    color=color, linewidth=2.5, label='High Innovation')
        axes[i].plot(policy_range, final_states_low[:, i], '--',
                    color=color, linewidth=2.5, label='Low Innovation')
        axes[i].set_xlabel('Policy Support Level', fontsize=12)
        axes[i].set_ylabel(f'Final {var}', fontsize=12)
        axes[i].set_title(f'Policy Impact on {var}', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        # y축 자동 조정
        y_min = min(final_states_high[:, i].min(), final_states_low[:, i].min())
        y_max = max(final_states_high[:, i].max(), final_states_low[:, i].max())
        axes[i].set_ylim([max(0, y_min - 0.1), y_max + 0.1])

        # 최적 정책 수준 표시
        max_idx_high = np.argmax(final_states_high[:, i])
        max_idx_low = np.argmax(final_states_low[:, i])
        axes[i].plot(policy_range[max_idx_high], final_states_high[max_idx_high, i],
                    'o', color='red', markersize=8)
        axes[i].plot(policy_range[max_idx_low], final_states_low[max_idx_low, i],
                    's', color='red', markersize=8)

    plt.suptitle('Feedback Loop Analysis: Optimal Policy Levels',
                fontsize=14, fontweight='bold')

    # 주석 추가
    fig.text(0.5, 0.01,
             '※ 빨간 점(○, □): 각 지표를 최대화하는 최적 정책 수준\n'
             '   고혁신 기업(실선): 정책 증가 시 모든 지표 상승\n'
             '   저혁신 기업(점선): 과도한 정책은 오히려 혁신 저하\n'
             '   함의: 맞춤형 정책 설계 필요 (one-size-fits-all 정책의 한계)',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    plt.subplots_adjust(bottom=0.20)
    plt.savefig('feedback_loop_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return policy_range, final_states_high, final_states_low

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("시스템 다이내믹스: 정책 피드백 시뮬레이션")
    print("한계기업 정책의 선순환/악순환 구조 분석")
    print("=" * 60)

    # 1. 정책 효과 비교
    print("\n1. 고혁신 vs 저혁신 기업의 정책 효과 비교...")
    run_policy_comparison()

    # 2. 피드백 루프 분석
    print("\n2. 피드백 루프 구조 분석 중...")
    policy_range, final_high, final_low = analyze_feedback_loops()

    # 3. 결과 요약
    print("\n분석 결과 요약:")
    print("-" * 40)

    # 최적 정책 수준 찾기
    opt_policy_high = policy_range[np.argmax(final_high[:, 2])]
    opt_policy_low = policy_range[np.argmax(final_low[:, 2])]

    print(f"고혁신 기업 최적 정책 지원 수준: {opt_policy_high:.2f}")
    print(f"저혁신 기업 최적 정책 지원 수준: {opt_policy_low:.2f}")

    # 정책 효과 차이
    max_eff_high = np.max(final_high[:, 2])
    max_eff_low = np.max(final_low[:, 2])
    print(f"\n최대 경제 효율성:")
    print(f"  - 고혁신 기업: {max_eff_high:.3f}")
    print(f"  - 저혁신 기업: {max_eff_low:.3f}")
    print(f"  - 효율성 차이: {max_eff_high - max_eff_low:.3f}")

    print("\n" + "=" * 60)
    print("시뮬레이션 완료!")
    print("생성된 파일:")
    print("  - policy_feedback_comparison.png")
    print("  - feedback_loop_analysis.png")
    print("=" * 60)

if __name__ == "__main__":
    main()