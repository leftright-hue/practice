"""
제12장: 복잡계 이론과 정책 시뮬레이션
12.4 강화학습과 적응형 정책: 스마트 그리드 에너지 관리

Q-learning과 Policy Gradient를 사용한 적응형 에너지 정책 학습
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SmartGridEnvironment:
    """스마트 그리드 환경 시뮬레이션"""

    def __init__(self, n_time_slots=24):
        """
        Parameters:
        -----------
        n_time_slots : int, 하루 시간 슬롯 수 (시간 단위)
        """
        self.n_time_slots = n_time_slots
        self.current_slot = 0

        # 시간대별 기본 수요 패턴 (정규화)
        self.base_demand = self._generate_demand_pattern()

        # 재생에너지 생산 패턴 (태양광 가정)
        self.renewable_supply = self._generate_renewable_pattern()

        # 시간대별 전력 가격
        self.prices = self._generate_price_pattern()

        # 에너지 저장장치 (ESS) 상태
        self.storage_capacity = 100  # kWh
        self.storage_level = 50  # 현재 저장량

    def _generate_demand_pattern(self):
        """일일 전력 수요 패턴 생성"""
        # 아침, 저녁 피크를 가진 패턴
        hours = np.arange(24)
        demand = (
            0.5 +
            0.3 * np.exp(-((hours - 8) ** 2) / 8) +  # 아침 피크
            0.4 * np.exp(-((hours - 19) ** 2) / 6)   # 저녁 피크
        )
        return demand / demand.max()

    def _generate_renewable_pattern(self):
        """재생에너지 생산 패턴 생성"""
        hours = np.arange(24)
        # 낮 시간대 태양광 생산
        solar = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
        solar[hours < 6] = 0
        solar[hours > 18] = 0
        return solar

    def _generate_price_pattern(self):
        """시간대별 전력 가격 패턴"""
        # 수요에 비례하는 가격
        base_price = 100  # 원/kWh
        prices = base_price * (0.5 + 1.5 * self.base_demand)
        return prices

    def reset(self):
        """환경 초기화"""
        self.current_slot = 0
        self.storage_level = 50
        return self._get_state()

    def _get_state(self):
        """현재 상태 반환"""
        state = [
            self.current_slot / 24,  # 시간 (정규화)
            self.base_demand[self.current_slot],  # 수요
            self.renewable_supply[self.current_slot],  # 재생에너지
            self.storage_level / self.storage_capacity,  # ESS 상태
            self.prices[self.current_slot] / 200  # 가격 (정규화)
        ]
        return np.array(state)

    def step(self, action):
        """
        행동 실행 및 다음 상태 전환

        Parameters:
        -----------
        action : int
            0: 그리드에서 구매
            1: ESS 충전
            2: ESS 방전
            3: 재생에너지만 사용
        """
        demand = self.base_demand[self.current_slot] * 100  # kWh
        renewable = self.renewable_supply[self.current_slot] * 50
        price = self.prices[self.current_slot]

        cost = 0
        penalty = 0

        if action == 0:  # 그리드 구매
            cost = (demand - renewable) * price
            self.storage_level = min(self.storage_capacity,
                                   self.storage_level + renewable * 0.3)

        elif action == 1:  # ESS 충전
            charge_amount = min(30, self.storage_capacity - self.storage_level)
            cost = (demand - renewable + charge_amount) * price
            self.storage_level += charge_amount * 0.9  # 충전 효율

        elif action == 2:  # ESS 방전
            discharge_amount = min(30, self.storage_level)
            self.storage_level -= discharge_amount
            net_demand = max(0, demand - renewable - discharge_amount)
            cost = net_demand * price

        elif action == 3:  # 재생에너지만
            if renewable >= demand:
                cost = 0
                excess = renewable - demand
                self.storage_level = min(self.storage_capacity,
                                       self.storage_level + excess * 0.9)
            else:
                # 수요 미충족 패널티
                shortage = demand - renewable
                cost = shortage * price * 1.5
                penalty = shortage * 10

        # 보상 계산 (비용 최소화 + 재생에너지 활용 보너스)
        renewable_usage = min(renewable, demand) / demand if demand > 0 else 0
        reward = -cost/100 - penalty/100 + renewable_usage * 0.5

        # 시간 진행
        self.current_slot = (self.current_slot + 1) % self.n_time_slots
        done = (self.current_slot == 0)

        next_state = self._get_state()
        info = {
            'cost': cost,
            'renewable_usage': renewable_usage,
            'storage': self.storage_level
        }

        return next_state, reward, done, info

class QLearningAgent:
    """Q-learning 에이전트"""

    def __init__(self, state_size=5, action_size=4, learning_rate=0.1, gamma=0.95):
        """
        Parameters:
        -----------
        state_size : int, 상태 공간 크기
        action_size : int, 행동 공간 크기
        learning_rate : float, 학습률
        gamma : float, 할인율
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Q-테이블 초기화 (이산화된 상태 사용)
        self.q_table = {}

    def _discretize_state(self, state):
        """연속 상태를 이산 상태로 변환"""
        # 각 차원을 10개 구간으로 이산화
        discrete = tuple(min(9, int(s * 10)) for s in state)
        return discrete

    def get_action(self, state):
        """ε-greedy 정책으로 행동 선택"""
        discrete_state = self._discretize_state(state)

        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)

        return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state, done):
        """Q-값 업데이트"""
        discrete_state = self._discretize_state(state)
        discrete_next = self._discretize_state(next_state)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if discrete_next not in self.q_table:
            self.q_table[discrete_next] = np.zeros(self.action_size)

        current_q = self.q_table[discrete_state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next])

        # Q-값 업데이트
        self.q_table[discrete_state][action] += self.learning_rate * (target - current_q)

        # ε 감소
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class PolicyGradientAgent:
    """간단한 Policy Gradient 에이전트 (REINFORCE)"""

    def __init__(self, state_size=5, action_size=4, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # 간단한 선형 정책 네트워크
        self.weights = np.random.randn(state_size, action_size) * 0.1
        self.bias = np.zeros(action_size)

        # 에피소드 메모리
        self.states = []
        self.actions = []
        self.rewards = []

    def get_action(self, state):
        """정책에 따른 행동 선택"""
        # 선형 변환 + softmax
        logits = np.dot(state, self.weights) + self.bias
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 확률적 행동 선택
        action = np.random.choice(self.action_size, p=probs)
        return action

    def store_transition(self, state, action, reward):
        """전이 저장"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update(self):
        """에피소드 종료 후 정책 업데이트"""
        if len(self.rewards) == 0:
            return

        # 리턴 계산 (discounted rewards)
        returns = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + 0.95 * discounted_reward
            returns.insert(0, discounted_reward)

        # 정규화
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # 정책 그래디언트 업데이트
        for state, action, G in zip(self.states, self.actions, returns):
            logits = np.dot(state, self.weights) + self.bias
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # 그래디언트 계산
            grad = -probs
            grad[action] += 1
            grad *= G

            # 가중치 업데이트
            self.weights += self.learning_rate * np.outer(state, grad)
            self.bias += self.learning_rate * grad

        # 메모리 초기화
        self.states = []
        self.actions = []
        self.rewards = []

def train_agents(n_episodes=500):
    """에이전트 학습"""
    env = SmartGridEnvironment()
    q_agent = QLearningAgent()
    pg_agent = PolicyGradientAgent()

    q_rewards = []
    pg_rewards = []
    q_costs = []
    pg_costs = []

    for episode in range(n_episodes):
        # Q-learning 에이전트 학습
        state = env.reset()
        episode_reward = 0
        episode_cost = 0

        for _ in range(24):  # 하루 시뮬레이션
            action = q_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            q_agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            episode_cost += info['cost']
            state = next_state

        q_rewards.append(episode_reward)
        q_costs.append(episode_cost)

        # Policy Gradient 에이전트 학습
        state = env.reset()
        episode_reward = 0
        episode_cost = 0

        for _ in range(24):
            action = pg_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            pg_agent.store_transition(state, action, reward)
            episode_reward += reward
            episode_cost += info['cost']
            state = next_state

        pg_agent.update()
        pg_rewards.append(episode_reward)
        pg_costs.append(episode_cost)

        if episode % 50 == 0:
            print(f"Episode {episode}: Q-reward={np.mean(q_rewards[-10:]):.2f}, "
                  f"PG-reward={np.mean(pg_rewards[-10:]):.2f}")

    return q_rewards, pg_rewards, q_costs, pg_costs, q_agent, pg_agent

def evaluate_policies(q_agent, pg_agent):
    """학습된 정책 평가"""
    env = SmartGridEnvironment()

    # 기준선: 항상 그리드 구매
    baseline_cost = 0
    state = env.reset()
    for _ in range(24):
        _, _, _, info = env.step(0)  # 항상 그리드 구매
        baseline_cost += info['cost']

    # Q-learning 정책
    q_cost = 0
    q_renewable = []
    state = env.reset()
    q_actions = []
    for _ in range(24):
        action = q_agent.get_action(state)
        q_actions.append(action)
        next_state, _, _, info = env.step(action)
        q_cost += info['cost']
        q_renewable.append(info['renewable_usage'])
        state = next_state

    # Policy Gradient 정책
    pg_cost = 0
    pg_renewable = []
    state = env.reset()
    pg_actions = []
    for _ in range(24):
        action = pg_agent.get_action(state)
        pg_actions.append(action)
        next_state, _, _, info = env.step(action)
        pg_cost += info['cost']
        pg_renewable.append(info['renewable_usage'])
        state = next_state

    return {
        'baseline': baseline_cost,
        'q_learning': {'cost': q_cost, 'renewable': q_renewable, 'actions': q_actions},
        'policy_gradient': {'cost': pg_cost, 'renewable': pg_renewable, 'actions': pg_actions}
    }

def visualize_results(q_rewards, pg_rewards, q_costs, pg_costs, evaluation):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 학습 곡선 (보상)
    window = 20
    q_smooth = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    pg_smooth = np.convolve(pg_rewards, np.ones(window)/window, mode='valid')

    axes[0, 0].plot(q_smooth, label='Q-Learning', color='blue', linewidth=2)
    axes[0, 0].plot(pg_smooth, label='Policy Gradient', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Average Reward', fontsize=12)
    axes[0, 0].set_title('Learning Curves (Rewards)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 비용 감소
    q_cost_smooth = np.convolve(q_costs, np.ones(window)/window, mode='valid')
    pg_cost_smooth = np.convolve(pg_costs, np.ones(window)/window, mode='valid')

    axes[0, 1].plot(q_cost_smooth, label='Q-Learning', color='blue', linewidth=2)
    axes[0, 1].plot(pg_cost_smooth, label='Policy Gradient', color='red', linewidth=2)
    axes[0, 1].axhline(y=evaluation['baseline'], color='gray', linestyle='--',
                      label='Baseline (Grid Only)')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Total Daily Cost (KRW)', fontsize=12)
    axes[0, 1].set_title('Cost Reduction Over Training', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 재생에너지 활용률
    hours = range(24)
    axes[1, 0].plot(hours, evaluation['q_learning']['renewable'],
                   'bo-', label='Q-Learning', markersize=4)
    axes[1, 0].plot(hours, evaluation['policy_gradient']['renewable'],
                   'rs-', label='Policy Gradient', markersize=4)
    axes[1, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[1, 0].set_ylabel('Renewable Usage Rate', fontsize=12)
    axes[1, 0].set_title('Renewable Energy Utilization', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])

    # 4. 행동 패턴
    action_labels = ['Grid', 'Charge', 'Discharge', 'Renewable']
    q_action_counts = [evaluation['q_learning']['actions'].count(i) for i in range(4)]
    pg_action_counts = [evaluation['policy_gradient']['actions'].count(i) for i in range(4)]

    x = np.arange(len(action_labels))
    width = 0.35

    axes[1, 1].bar(x - width/2, q_action_counts, width, label='Q-Learning', color='blue')
    axes[1, 1].bar(x + width/2, pg_action_counts, width, label='Policy Gradient', color='red')
    axes[1, 1].set_xlabel('Action Type', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Action Distribution (24 hours)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(action_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Reinforcement Learning for Smart Grid Management',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rl_smart_grid.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("강화학습 기반 스마트 그리드 에너지 관리")
    print("=" * 60)

    # 1. 에이전트 학습
    print("\n1. 에이전트 학습 중 (500 에피소드)...")
    q_rewards, pg_rewards, q_costs, pg_costs, q_agent, pg_agent = train_agents(n_episodes=500)

    # 2. 정책 평가
    print("\n2. 학습된 정책 평가 중...")
    evaluation = evaluate_policies(q_agent, pg_agent)

    print("\n평가 결과:")
    print("-" * 40)
    print(f"일일 전력 비용:")
    print(f"  - 기준선 (그리드만): {evaluation['baseline']:.0f} KRW")
    print(f"  - Q-Learning: {evaluation['q_learning']['cost']:.0f} KRW "
          f"({(1 - evaluation['q_learning']['cost']/evaluation['baseline'])*100:.1f}% 절감)")
    print(f"  - Policy Gradient: {evaluation['policy_gradient']['cost']:.0f} KRW "
          f"({(1 - evaluation['policy_gradient']['cost']/evaluation['baseline'])*100:.1f}% 절감)")

    print(f"\n재생에너지 평균 활용률:")
    print(f"  - Q-Learning: {np.mean(evaluation['q_learning']['renewable']):.1%}")
    print(f"  - Policy Gradient: {np.mean(evaluation['policy_gradient']['renewable']):.1%}")

    # 3. 결과 시각화
    print("\n3. 결과 시각화...")
    visualize_results(q_rewards, pg_rewards, q_costs, pg_costs, evaluation)

    print("\n" + "=" * 60)
    print("시뮬레이션 완료!")
    print("생성된 파일: rl_smart_grid.png")
    print("=" * 60)

if __name__ == "__main__":
    main()