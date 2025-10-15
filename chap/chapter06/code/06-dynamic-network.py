"""
제6장: 그래프 이론과 정책 네트워크 분석
06-dynamic-network.py - 동적 네트워크 분석과 정책 확산 시뮬레이션
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import random
import json
from datetime import datetime, timedelta

# 한글 폰트 설정
import matplotlib.font_manager as fm

# 사용 가능한 한글 폰트 찾기
available_fonts = [f.name for f in fm.fontManager.ttflist]
korean_fonts = [
    'AppleGothic',        # macOS
    'NanumGothic',        # 범용
    'Malgun Gothic',      # Windows
    'Gulim',              # Windows
    'Batang',             # Windows
    'Dotum',              # Windows
    'Arial Unicode MS',   # 범용
    'DejaVu Sans'         # Linux/범용
]
selected_font = 'DejaVu Sans'  # 기본값

for font in korean_fonts:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False

class DynamicPolicyNetwork:
    """동적 정책 네트워크 클래스"""
    
    def __init__(self, initial_nodes=10):
        self.networks = []  # 시간별 네트워크 저장
        self.timestamps = []  # 타임스탬프 저장
        self.node_history = {}  # 노드 이력 추적
        self.edge_history = {}  # 엣지 이력 추적
        
        # 초기 네트워크 생성
        self.current_network = self._create_initial_network(initial_nodes)
        self.networks.append(self.current_network.copy())
        self.timestamps.append(datetime(2020, 1, 1))
    
    def _create_initial_network(self, n_nodes):
        """초기 네트워크 생성"""
        G = nx.barabasi_albert_graph(n_nodes, 2)
        G = G.to_directed()
        
        # 노드 속성 추가
        for node in G.nodes():
            G.nodes[node]['birth_time'] = 0
            G.nodes[node]['type'] = random.choice(['government', 'business', 'ngo'])
            G.nodes[node]['influence'] = random.uniform(0.1, 1.0)
        
        # 엣지 속성 추가
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.uniform(0.1, 1.0)
            G[edge[0]][edge[1]]['birth_time'] = 0
        
        return G
    
    def add_node(self, node_id, **attributes):
        """새로운 노드 추가"""
        time_step = len(self.networks)
        self.current_network.add_node(node_id, birth_time=time_step, **attributes)
        
        # 우선적 연결 메커니즘
        degrees = dict(self.current_network.degree())
        if degrees:
            prob = np.array(list(degrees.values()))
            prob = prob / prob.sum()
            n_connections = min(3, len(degrees))
            targets = np.random.choice(list(degrees.keys()), 
                                      size=n_connections, 
                                      p=prob, replace=False)
            
            for target in targets:
                weight = random.uniform(0.1, 1.0)
                self.current_network.add_edge(node_id, target, 
                                             weight=weight, 
                                             birth_time=time_step)
    
    def remove_node(self, node_id):
        """노드 제거"""
        if node_id in self.current_network:
            self.current_network.remove_node(node_id)
    
    def update_edge_weight(self, source, target, new_weight):
        """엣지 가중치 업데이트"""
        if self.current_network.has_edge(source, target):
            self.current_network[source][target]['weight'] = new_weight
    
    def evolve(self, n_steps=10, p_add=0.1, p_remove=0.05, p_rewire=0.1):
        """네트워크 진화 시뮬레이션"""
        
        for step in range(n_steps):
            # 현재 네트워크 복사
            G = self.current_network.copy()
            
            # 노드 추가
            if random.random() < p_add:
                new_node = max(G.nodes()) + 1 if G.nodes() else 0
                node_type = random.choice(['government', 'business', 'ngo'])
                self.add_node(new_node, type=node_type, 
                            influence=random.uniform(0.1, 1.0))
            
            # 노드 제거
            if random.random() < p_remove and len(G.nodes()) > 5:
                node_to_remove = random.choice(list(G.nodes()))
                self.remove_node(node_to_remove)
            
            # 엣지 재연결
            if random.random() < p_rewire:
                edges = list(G.edges())
                if edges:
                    edge_to_remove = random.choice(edges)
                    G.remove_edge(*edge_to_remove)
                    
                    # 새로운 엣지 추가
                    nodes = list(G.nodes())
                    if len(nodes) >= 2:
                        source = random.choice(nodes)
                        target = random.choice([n for n in nodes if n != source])
                        if not G.has_edge(source, target):
                            G.add_edge(source, target, 
                                     weight=random.uniform(0.1, 1.0),
                                     birth_time=len(self.networks))
            
            # 네트워크 저장
            self.networks.append(self.current_network.copy())
            self.timestamps.append(self.timestamps[-1] + timedelta(days=30))
    
    def calculate_temporal_metrics(self):
        """시간별 네트워크 메트릭 계산"""
        metrics = []
        
        for i, G in enumerate(self.networks):
            metric = {
                'time': i,
                'timestamp': self.timestamps[i],
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_degree': np.mean([d for n, d in G.degree()]) if G.nodes() else 0,
                'clustering': nx.average_clustering(G.to_undirected()) if G.nodes() else 0
            }
            
            # 중심성 변화
            if G.nodes():
                degree_cent = nx.degree_centrality(G)
                metric['max_centrality'] = max(degree_cent.values()) if degree_cent else 0
                metric['centrality_std'] = np.std(list(degree_cent.values())) if degree_cent else 0
            else:
                metric['max_centrality'] = 0
                metric['centrality_std'] = 0
            
            metrics.append(metric)
        
        return pd.DataFrame(metrics)

class PolicyDiffusionModel:
    """정책 확산 모델"""
    
    def __init__(self, network):
        self.network = network
        self.states = {node: 'susceptible' for node in network.nodes()}
        self.adoption_times = {}
        self.time = 0
    
    def set_initial_adopters(self, adopters):
        """초기 채택자 설정"""
        for adopter in adopters:
            if adopter in self.network:
                self.states[adopter] = 'adopted'
                self.adoption_times[adopter] = 0
    
    def threshold_model(self, threshold=0.3):
        """임계값 모델 기반 확산"""
        new_adopters = []
        
        for node in self.network.nodes():
            if self.states[node] == 'susceptible':
                neighbors = list(self.network.neighbors(node))
                if neighbors:
                    adopted_neighbors = sum(1 for n in neighbors 
                                          if self.states[n] == 'adopted')
                    adoption_rate = adopted_neighbors / len(neighbors)
                    
                    if adoption_rate >= threshold:
                        new_adopters.append(node)
        
        # 상태 업데이트
        for adopter in new_adopters:
            self.states[adopter] = 'adopted'
            self.adoption_times[adopter] = self.time
        
        return new_adopters
    
    def si_model(self, infection_rate=0.1):
        """SI (Susceptible-Infected) 전염병 모델"""
        new_adopters = []
        
        for node in self.network.nodes():
            if self.states[node] == 'adopted':
                neighbors = list(self.network.neighbors(node))
                for neighbor in neighbors:
                    if self.states[neighbor] == 'susceptible':
                        if random.random() < infection_rate:
                            new_adopters.append(neighbor)
        
        # 중복 제거
        new_adopters = list(set(new_adopters))
        
        # 상태 업데이트
        for adopter in new_adopters:
            self.states[adopter] = 'adopted'
            self.adoption_times[adopter] = self.time
        
        return new_adopters
    
    def run_diffusion(self, model_type='threshold', max_steps=50, **params):
        """확산 시뮬레이션 실행"""
        history = []
        
        for step in range(max_steps):
            self.time = step
            
            if model_type == 'threshold':
                new_adopters = self.threshold_model(params.get('threshold', 0.3))
            elif model_type == 'si':
                new_adopters = self.si_model(params.get('infection_rate', 0.1))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            adopted_count = sum(1 for state in self.states.values() 
                              if state == 'adopted')
            history.append({
                'step': step,
                'adopted': adopted_count,
                'new_adopters': len(new_adopters),
                'adoption_rate': adopted_count / len(self.network.nodes())
            })
            
            # 더 이상 확산되지 않으면 종료
            if len(new_adopters) == 0:
                break
        
        return pd.DataFrame(history)

def visualize_network_evolution(dynamic_network, save_path='chap/chapter06/outputs/'):
    """네트워크 진화 시각화"""
    
    metrics_df = dynamic_network.calculate_temporal_metrics()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 노드 수 변화
    axes[0, 0].plot(metrics_df['time'], metrics_df['n_nodes'], 'b-', marker='o')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].set_title('Network Size Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 엣지 수 변화
    axes[0, 1].plot(metrics_df['time'], metrics_df['n_edges'], 'g-', marker='s')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Number of Edges')
    axes[0, 1].set_title('Edge Count Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 밀도 변화
    axes[0, 2].plot(metrics_df['time'], metrics_df['density'], 'r-', marker='^')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Network Density')
    axes[0, 2].set_title('Density Evolution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 평균 차수 변화
    axes[1, 0].plot(metrics_df['time'], metrics_df['avg_degree'], 'purple', marker='d')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Average Degree')
    axes[1, 0].set_title('Average Degree Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 클러스터링 계수 변화
    axes[1, 1].plot(metrics_df['time'], metrics_df['clustering'], 'orange', marker='*')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Clustering Coefficient')
    axes[1, 1].set_title('Clustering Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 최대 중심성 변화
    axes[1, 2].plot(metrics_df['time'], metrics_df['max_centrality'], 'brown', marker='h')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Max Centrality')
    axes[1, 2].set_title('Maximum Centrality Evolution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Dynamic Network Evolution Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}network_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_df

def visualize_diffusion(diffusion_history, model_type='threshold'):
    """정책 확산 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 채택자 수 증가 곡선
    axes[0].plot(diffusion_history['step'], diffusion_history['adopted'], 
                'b-', linewidth=2, marker='o')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Number of Adopters')
    axes[0].set_title(f'Policy Diffusion Curve ({model_type} model)')
    axes[0].grid(True, alpha=0.3)
    
    # 신규 채택자 수
    axes[1].bar(diffusion_history['step'], diffusion_history['new_adopters'], 
               color='green', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('New Adopters')
    axes[1].set_title('New Adopters per Time Step')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'chap/chapter06/outputs/diffusion_{model_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("동적 네트워크 분석 및 정책 확산 시뮬레이션")
    print("=" * 60)
    
    # 1. 동적 네트워크 생성 및 진화
    print("\n[1] 동적 네트워크 진화 시뮬레이션")
    dynamic_net = DynamicPolicyNetwork(initial_nodes=15)
    dynamic_net.evolve(n_steps=20, p_add=0.15, p_remove=0.05, p_rewire=0.1)
    
    # 네트워크 진화 시각화
    metrics_df = visualize_network_evolution(dynamic_net)
    print(f"네트워크 진화 완료: {len(dynamic_net.networks)} 시점")
    
    # 2. 정책 확산 시뮬레이션 - 임계값 모델
    print("\n[2] 정책 확산 시뮬레이션 - 임계값 모델")
    final_network = dynamic_net.networks[-1]
    
    diffusion_threshold = PolicyDiffusionModel(final_network)
    
    # 중심성이 높은 노드를 초기 채택자로 선택
    if final_network.nodes():
        centrality = nx.degree_centrality(final_network)
        initial_adopters = sorted(centrality.keys(), 
                                 key=centrality.get, reverse=True)[:2]
        diffusion_threshold.set_initial_adopters(initial_adopters)
        
        # 확산 실행
        threshold_history = diffusion_threshold.run_diffusion(
            model_type='threshold', threshold=0.25
        )
        visualize_diffusion(threshold_history, 'threshold')
        
        print(f"임계값 모델 확산 완료:")
        print(f"  - 최종 채택률: {threshold_history['adoption_rate'].iloc[-1]:.2%}")
        print(f"  - 확산 단계: {len(threshold_history)}")
    
    # 3. 정책 확산 시뮬레이션 - SI 모델
    print("\n[3] 정책 확산 시뮬레이션 - SI 모델")
    diffusion_si = PolicyDiffusionModel(final_network)
    
    if final_network.nodes():
        diffusion_si.set_initial_adopters(initial_adopters)
        
        # 확산 실행
        si_history = diffusion_si.run_diffusion(
            model_type='si', infection_rate=0.15
        )
        visualize_diffusion(si_history, 'si')
        
        print(f"SI 모델 확산 완료:")
        print(f"  - 최종 채택률: {si_history['adoption_rate'].iloc[-1]:.2%}")
        print(f"  - 확산 단계: {len(si_history)}")
    
    # 결과 저장
    metrics_df.to_csv('chap/chapter06/outputs/network_evolution_metrics.csv', index=False)
    print("\n분석 결과가 저장되었습니다.")
    
    return dynamic_net, threshold_history, si_history

if __name__ == "__main__":
    dynamic_network, threshold_results, si_results = main()