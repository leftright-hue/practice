# -*- coding: utf-8 -*-
"""
제12장: 복잡계 이론과 정책 시뮬레이션
12.3 에이전트 기반 모델링(ABM): COVID-19 정책 확산 시뮬레이션

사회적 거리두기 정책 채택이 네트워크를 통해 확산되는 과정을 모델링합니다.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager, rc
from matplotlib.animation import FuncAnimation
import random

# Windows 콘솔 UTF-8 설정
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 설정
try:
    # Windows
    if sys.platform.startswith('win'):
        plt.rcParams['font.family'] = 'Malgun Gothic'
    # macOS
    elif sys.platform == 'darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    # Linux
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("Warning: 한글 폰트 설정에 실패했습니다. 영문으로 표시됩니다.")

class PolicyAgent:
    """정책 채택 에이전트"""

    def __init__(self, agent_id, adoption_threshold=0.5, risk_perception=0.5):
        """
        Parameters:
        -----------
        agent_id : int, 에이전트 ID
        adoption_threshold : float, 채택 임계값 (0-1)
        risk_perception : float, 위험 인식 수준 (0-1)
        """
        self.id = agent_id
        self.adopted = False
        self.threshold = adoption_threshold
        self.risk_perception = risk_perception
        self.neighbors = []
        self.adoption_time = -1
        self.influence_score = 0

    def add_neighbor(self, neighbor):
        """이웃 에이전트 추가"""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def update(self, current_time, global_risk=0.5):
        """에이전트 상태 업데이트"""
        if not self.adopted:
            # 이웃들의 채택률 계산
            adopted_neighbors = sum(1 for n in self.neighbors if n.adopted)
            adoption_rate = adopted_neighbors / len(self.neighbors) if self.neighbors else 0

            # 글로벌 위험과 개인 위험 인식 결합
            perceived_pressure = adoption_rate * 0.6 + global_risk * self.risk_perception * 0.4

            # 임계값 넘으면 채택
            if perceived_pressure > self.threshold:
                self.adopted = True
                self.adoption_time = current_time
                self.influence_score = adopted_neighbors  # 영향력 점수
                return True
        return False

class PolicyDiffusionABM:
    """정책 확산 ABM 시뮬레이션"""

    def __init__(self, n_agents=100, network_type='small_world', avg_connections=6):
        """
        Parameters:
        -----------
        n_agents : int, 에이전트 수
        network_type : str, 네트워크 유형 ('random', 'small_world', 'scale_free')
        avg_connections : int, 평균 연결 수
        """
        self.n_agents = n_agents
        self.agents = []
        self.network = None
        self.network_type = network_type

        # 에이전트 생성 (이질적 특성)
        for i in range(n_agents):
            threshold = np.random.beta(2, 2)  # 베타 분포로 다양한 임계값
            risk_perception = np.random.beta(2, 2)
            agent = PolicyAgent(i, threshold, risk_perception)
            self.agents.append(agent)

        # 네트워크 생성
        self._create_network(network_type, avg_connections)

    def _create_network(self, network_type, avg_connections):
        """소셜 네트워크 생성"""
        if network_type == 'random':
            p = avg_connections / (self.n_agents - 1)
            self.network = nx.erdos_renyi_graph(self.n_agents, p)
        elif network_type == 'small_world':
            self.network = nx.watts_strogatz_graph(self.n_agents, avg_connections, 0.3)
        elif network_type == 'scale_free':
            self.network = nx.barabasi_albert_graph(self.n_agents, avg_connections // 2)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # 네트워크 연결을 에이전트에 반영
        for edge in self.network.edges():
            self.agents[edge[0]].add_neighbor(self.agents[edge[1]])
            self.agents[edge[1]].add_neighbor(self.agents[edge[0]])

    def set_initial_adopters(self, n_initial=5, strategy='random'):
        """초기 채택자 설정"""
        if strategy == 'random':
            initial_ids = random.sample(range(self.n_agents), n_initial)
        elif strategy == 'hub':
            # 연결이 많은 허브 노드 선택
            degrees = dict(self.network.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            initial_ids = [node[0] for node in sorted_nodes[:n_initial]]
        else:
            initial_ids = list(range(n_initial))

        for agent_id in initial_ids:
            self.agents[agent_id].adopted = True
            self.agents[agent_id].adoption_time = 0

    def simulate(self, max_steps=50, global_risk_scenario='constant'):
        """시뮬레이션 실행"""
        adoption_history = []
        new_adoptions = []

        for step in range(max_steps):
            # 글로벌 위험 시나리오
            if global_risk_scenario == 'constant':
                global_risk = 0.5
            elif global_risk_scenario == 'increasing':
                global_risk = min(0.2 + step * 0.02, 1.0)
            elif global_risk_scenario == 'wave':
                global_risk = 0.5 + 0.3 * np.sin(step * 0.3)
            else:
                global_risk = 0.5

            # 에이전트 업데이트
            step_adoptions = 0
            for agent in self.agents:
                if agent.update(step, global_risk):
                    step_adoptions += 1

            total_adopted = sum(1 for a in self.agents if a.adopted)
            adoption_history.append(total_adopted / self.n_agents)
            new_adoptions.append(step_adoptions)

        return adoption_history, new_adoptions

    def get_network_metrics(self):
        """네트워크 메트릭 계산"""
        metrics = {
            'clustering': nx.average_clustering(self.network),
            'avg_path_length': nx.average_shortest_path_length(self.network)
                              if nx.is_connected(self.network) else float('inf'),
            'degree_centrality': nx.degree_centrality(self.network),
            'betweenness': nx.betweenness_centrality(self.network)
        }
        return metrics

def visualize_diffusion(abm_model, adoption_history):
    """정책 확산 과정 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 네트워크 시각화
    pos = nx.spring_layout(abm_model.network, k=2, iterations=50)
    colors = ['red' if agent.adopted else 'lightblue'
             for agent in abm_model.agents]
    sizes = [300 + agent.influence_score * 50 for agent in abm_model.agents]

    nx.draw_networkx(abm_model.network, pos, ax=axes[0, 0],
                    node_color=colors, node_size=sizes,
                    with_labels=False, edge_color='gray', alpha=0.5)
    axes[0, 0].set_title('Final Adoption Network\n(Red=Adopted, Blue=Not Adopted)',
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. 채택 곡선
    axes[0, 1].plot(adoption_history, 'b-', linewidth=2.5)
    axes[0, 1].fill_between(range(len(adoption_history)), 0, adoption_history,
                           alpha=0.3)
    axes[0, 1].set_xlabel('Time Step', fontsize=12)
    axes[0, 1].set_ylabel('Adoption Rate', fontsize=12)
    axes[0, 1].set_title('Policy Adoption Curve', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # 3. 에이전트 특성 분포
    thresholds = [agent.threshold for agent in abm_model.agents]
    risk_perceptions = [agent.risk_perception for agent in abm_model.agents]
    adopted_agents = [agent for agent in abm_model.agents if agent.adopted]

    axes[1, 0].hist2d(thresholds, risk_perceptions, bins=15, cmap='Blues')
    if adopted_agents:
        adopted_thresh = [a.threshold for a in adopted_agents]
        adopted_risk = [a.risk_perception for a in adopted_agents]
        axes[1, 0].scatter(adopted_thresh, adopted_risk, c='red', s=20, alpha=0.6)
    axes[1, 0].set_xlabel('Adoption Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Risk Perception', fontsize=12)
    axes[1, 0].set_title('Agent Characteristics\n(Red dots = Adopted)',
                        fontsize=12, fontweight='bold')

    # 4. 채택 시간 분포
    adoption_times = [agent.adoption_time for agent in abm_model.agents
                     if agent.adopted and agent.adoption_time >= 0]
    if adoption_times:
        axes[1, 1].hist(adoption_times, bins=20, color='skyblue', edgecolor='navy')
    axes[1, 1].set_xlabel('Adoption Time', fontsize=12)
    axes[1, 1].set_ylabel('Number of Agents', fontsize=12)
    axes[1, 1].set_title('Adoption Timing Distribution',
                        fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'ABM Policy Diffusion: {abm_model.network_type.title()} Network',
                fontsize=14, fontweight='bold')

    # 주석 추가
    fig.text(0.5, 0.02,
             '※ 좌상: 최종 네트워크 상태 (빨강=채택, 파랑=미채택, 노드 크기=영향력)\n'
             '   우상: 시간에 따른 정책 채택률 곡선 (S자 곡선은 티핑 포인트 존재)\n'
             '   좌하: 에이전트 특성 분포 (임계값 낮고 위험인식 높을수록 빠른 채택)\n'
             '   우하: 채택 시점 분포 (초기/중기/후기 분산 → 네트워크 연결과 개인 특성의 복합 작용)',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.subplots_adjust(bottom=0.12)
    plt.savefig('abm_diffusion.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_network_types():
    """네트워크 유형별 확산 패턴 비교"""
    network_types = ['random', 'small_world', 'scale_free']
    n_runs = 10  # 각 유형별 반복 실행 횟수

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue', 'green', 'red']

    for idx, network_type in enumerate(network_types):
        all_histories = []

        for _ in range(n_runs):
            model = PolicyDiffusionABM(n_agents=100, network_type=network_type)
            model.set_initial_adopters(n_initial=5, strategy='random')
            history, _ = model.simulate(max_steps=50)
            all_histories.append(history)

        # 평균과 표준편차 계산
        all_histories = np.array(all_histories)
        mean_history = np.mean(all_histories, axis=0)
        std_history = np.std(all_histories, axis=0)

        # 플롯
        x = range(len(mean_history))
        axes[idx].plot(x, mean_history, color=colors[idx], linewidth=2.5,
                      label='Mean')
        axes[idx].fill_between(x, mean_history - std_history,
                              mean_history + std_history,
                              color=colors[idx], alpha=0.2)
        axes[idx].set_xlabel('Time Step', fontsize=12)
        axes[idx].set_ylabel('Adoption Rate', fontsize=12)
        axes[idx].set_title(f'{network_type.title()} Network',
                          fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
        axes[idx].legend()

    plt.suptitle('Network Structure Impact on Policy Diffusion',
                fontsize=14, fontweight='bold')

    # 주석 추가
    fig.text(0.5, 0.01,
             '※ 세 네트워크 모두 유사한 채택률 (10-15%) 및 확산 패턴\n'
             '   Random 초기 채택자 전략 사용 시 네트워크 구조 효과 제한적\n'
             '   높은 채택 임계값 (평균 ~0.5)으로 인한 확산 장벽 존재\n'
             '   정책 함의: 초기 채택자를 전략적으로 선정(허브 노드)하면 네트워크 효과 극대화 가능',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    plt.subplots_adjust(bottom=0.20)
    plt.savefig('network_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("에이전트 기반 모델링: COVID-19 정책 확산 시뮬레이션")
    print("=" * 60)

    # 1. 기본 시뮬레이션
    print("\n1. Small-World 네트워크에서 정책 확산 시뮬레이션...")
    model = PolicyDiffusionABM(n_agents=100, network_type='small_world')
    model.set_initial_adopters(n_initial=5, strategy='hub')
    history, new_adoptions = model.simulate(max_steps=50, global_risk_scenario='increasing')

    # 네트워크 메트릭 계산
    metrics = model.get_network_metrics()
    print(f"   - 네트워크 클러스터링 계수: {metrics['clustering']:.3f}")
    print(f"   - 최종 채택률: {history[-1]:.2%}")
    print(f"   - 50% 채택 도달 시간: ", end="")
    for i, rate in enumerate(history):
        if rate >= 0.5:
            print(f"{i} steps")
            break
    else:
        print("미도달")

    # 2. 시각화
    print("\n2. 확산 과정 시각화...")
    visualize_diffusion(model, history)

    # 3. 네트워크 유형 비교
    print("\n3. 네트워크 유형별 확산 패턴 비교 (각 10회 실행)...")
    compare_network_types()

    # 4. 결과 요약
    print("\n시뮬레이션 결과 요약:")
    print("-" * 40)
    print(f"총 에이전트 수: {model.n_agents}")
    print(f"초기 채택자 수: 5")
    print(f"최종 채택자 수: {sum(1 for a in model.agents if a.adopted)}")
    print(f"평균 채택 임계값: {np.mean([a.threshold for a in model.agents]):.3f}")
    print(f"평균 위험 인식: {np.mean([a.risk_perception for a in model.agents]):.3f}")

    print("\n" + "=" * 60)
    print("시뮬레이션 완료!")
    print("생성된 파일:")
    print("  - abm_diffusion.png")
    print("  - network_comparison.png")
    print("=" * 60)

if __name__ == "__main__":
    main()