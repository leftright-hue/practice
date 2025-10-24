#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제6장: 탄소중립 2050 정책 확산 분석
실제 정책 사례 기반 동적 네트워크 분석
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import matplotlib.font_manager as fm

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
selected_font = 'DejaVu Sans'

for font in korean_fonts:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False

class CarbonNeutralPolicyAnalysis:
    """탄소중립 2050 정책 확산 분석 클래스"""

    def __init__(self, adoption_file, network_file):
        """
        Args:
            adoption_file: 부처별 정책 채택 데이터
            network_file: 부처 간 협력 네트워크 데이터
        """
        self.adoption_df = pd.read_csv(adoption_file)
        self.network_df = pd.read_csv(network_file)

        # 날짜 파싱
        self.adoption_df['adoption_date'] = pd.to_datetime(self.adoption_df['adoption_date'])
        self.network_df['start_date'] = pd.to_datetime(self.network_df['start_date'])

        # 정책 확산 기간 계산
        self.start_date = self.adoption_df['adoption_date'].min()
        self.end_date = self.adoption_df['adoption_date'].max()
        self.duration_days = (self.end_date - self.start_date).days

    def build_temporal_network(self):
        """시간에 따른 네트워크 구축"""
        G = nx.DiGraph()

        # 노드 추가 (부처별 속성 포함)
        for _, row in self.adoption_df.iterrows():
            G.add_node(
                row['ministry'],
                adoption_date=row['adoption_date'],
                role=row['role'],
                leadership=row['leadership_score'],
                budget=row['budget_allocation'],
                intensity=row['policy_intensity']
            )

        # 엣지 추가 (협력 관계)
        for _, row in self.network_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                relationship=row['relationship_type'],
                influence=row['influence_strength'],
                start_date=row['start_date'],
                projects=row['collaboration_projects']
            )

        return G

    def identify_diffusion_stages(self):
        """정책 확산 단계 식별"""
        stages = []

        # 월별로 그룹화
        self.adoption_df['year_month'] = self.adoption_df['adoption_date'].dt.to_period('M')
        monthly_adoption = self.adoption_df.groupby('year_month').agg({
            'ministry': list,
            'leadership_score': 'mean',
            'budget_allocation': 'sum',
            'policy_intensity': 'mean'
        }).reset_index()

        # 누적 채택 계산
        total_ministries = len(self.adoption_df)
        cumulative_count = 0

        for idx, row in monthly_adoption.iterrows():
            cumulative_count += len(row['ministry'])
            adoption_rate = cumulative_count / total_ministries * 100

            stages.append({
                'period': str(row['year_month']),
                'new_adopters': len(row['ministry']),
                'cumulative_adopters': cumulative_count,
                'adoption_rate': adoption_rate,
                'ministries': ', '.join(row['ministry']),
                'avg_leadership': row['leadership_score'],
                'total_budget': row['budget_allocation'],
                'avg_intensity': row['policy_intensity']
            })

        return pd.DataFrame(stages)

    def identify_innovators_and_followers(self):
        """혁신자와 추종자 분류"""
        sorted_df = self.adoption_df.sort_values('adoption_date')
        n = len(sorted_df)

        # Rogers의 혁신 확산 이론 기반 분류
        categories = []
        for idx, (_, row) in enumerate(sorted_df.iterrows()):
            percentile = (idx + 1) / n * 100

            if percentile <= 2.5:
                category = '혁신자 (Innovators)'
            elif percentile <= 16:
                category = '조기 채택자 (Early Adopters)'
            elif percentile <= 50:
                category = '조기 다수 (Early Majority)'
            elif percentile <= 84:
                category = '후기 다수 (Late Majority)'
            else:
                category = '지각 채택자 (Laggards)'

            categories.append({
                'ministry': row['ministry'],
                'adoption_date': row['adoption_date'],
                'category': category,
                'leadership_score': row['leadership_score'],
                'role': row['role']
            })

        return pd.DataFrame(categories)

    def analyze_influence_paths(self, G):
        """영향력 경로 분석"""
        # 환경부를 시작점으로 각 부처까지의 경로 분석
        source = '환경부'
        paths_analysis = []

        for target in G.nodes():
            if target == source:
                continue

            try:
                # 최단 경로
                shortest_path = nx.shortest_path(G, source, target)
                path_length = len(shortest_path) - 1

                # 경로상 영향력 합산
                total_influence = 0
                for i in range(len(shortest_path) - 1):
                    if G.has_edge(shortest_path[i], shortest_path[i+1]):
                        total_influence += G[shortest_path[i]][shortest_path[i+1]].get('influence', 0)

                paths_analysis.append({
                    'target_ministry': target,
                    'path_length': path_length,
                    'path': ' → '.join(shortest_path),
                    'total_influence': total_influence,
                    'adoption_date': G.nodes[target]['adoption_date']
                })
            except nx.NetworkXNoPath:
                paths_analysis.append({
                    'target_ministry': target,
                    'path_length': np.inf,
                    'path': '경로 없음',
                    'total_influence': 0,
                    'adoption_date': G.nodes[target]['adoption_date']
                })

        return pd.DataFrame(paths_analysis).sort_values('path_length')

    def calculate_network_metrics(self, G):
        """네트워크 지표 계산"""
        metrics = {}

        # 기본 지표
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()

        # 중심성 분석
        in_centrality = nx.in_degree_centrality(G)
        out_centrality = nx.out_degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        # 상위 5개 부처
        metrics['top_in_centrality'] = sorted(in_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['top_out_centrality'] = sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['top_betweenness'] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

        return metrics

    def visualize_adoption_timeline(self, stages_df, save_path='chap/chapter06/outputs/'):
        """정책 채택 타임라인 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 누적 채택률
        axes[0, 0].plot(range(len(stages_df)), stages_df['adoption_rate'],
                       'b-', linewidth=2, marker='o')
        axes[0, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% 채택')
        axes[0, 0].set_xlabel('기간 (월)')
        axes[0, 0].set_ylabel('누적 채택률 (%)')
        axes[0, 0].set_title('탄소중립 2050 정책 채택률 추이')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 2. 월별 신규 채택 부처 수
        axes[0, 1].bar(range(len(stages_df)), stages_df['new_adopters'], color='green', alpha=0.7)
        axes[0, 1].set_xlabel('기간 (월)')
        axes[0, 1].set_ylabel('신규 채택 부처 수')
        axes[0, 1].set_title('월별 신규 채택 부처')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 누적 예산 배분
        axes[1, 0].plot(range(len(stages_df)), stages_df['total_budget'].cumsum() / 100,
                       'orange', linewidth=2, marker='s')
        axes[1, 0].set_xlabel('기간 (월)')
        axes[1, 0].set_ylabel('누적 예산 (억원)')
        axes[1, 0].set_title('탄소중립 정책 누적 예산 배분')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 평균 정책 강도
        axes[1, 1].plot(range(len(stages_df)), stages_df['avg_intensity'],
                       'm-', linewidth=2, marker='^')
        axes[1, 1].set_xlabel('기간 (월)')
        axes[1, 1].set_ylabel('평균 정책 강도')
        axes[1, 1].set_title('시기별 평균 정책 강도')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}carbon_neutral_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_network(self, G, categories_df, save_path='chap/chapter06/outputs/'):
        """정책 확산 네트워크 시각화"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # 카테고리별 색상 매핑
        category_colors = {
            '혁신자 (Innovators)': '#ff4444',
            '조기 채택자 (Early Adopters)': '#ff8800',
            '조기 다수 (Early Majority)': '#ffcc00',
            '후기 다수 (Late Majority)': '#88cc00',
            '지각 채택자 (Laggards)': '#0088cc'
        }

        category_map = dict(zip(categories_df['ministry'], categories_df['category']))
        node_colors = [category_colors.get(category_map.get(node, ''), 'gray') for node in G.nodes()]

        # 노드 크기는 리더십 점수 기반
        node_sizes = [G.nodes[node].get('leadership', 5) * 100 for node in G.nodes()]

        # 레이아웃
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # 엣지 그리기 (영향력에 따라 두께 조절)
        edges = G.edges()
        edge_widths = [G[u][v].get('influence', 1) * 0.3 for u, v in edges]

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3,
                              edge_color='gray', arrows=True, arrowsize=20)

        # 노드 그리기
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              alpha=0.9, edgecolors='black', linewidths=1.5)

        # 라벨
        labels = {node: node.replace('부', '') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold',
                               font_family=selected_font)

        # 범례
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat)
                          for cat, color in category_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.title('탄소중립 2050 정책 확산 네트워크\n(노드 크기 = 리더십 점수, 엣지 두께 = 영향력)',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_path}carbon_neutral_network.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_influence_heatmap(self, save_path='chap/chapter06/outputs/'):
        """부처 간 영향력 히트맵"""
        # 영향력 매트릭스 구성
        ministries = sorted(self.adoption_df['ministry'].unique())
        n = len(ministries)
        influence_matrix = np.zeros((n, n))

        ministry_idx = {m: i for i, m in enumerate(ministries)}

        for _, row in self.network_df.iterrows():
            i = ministry_idx[row['source']]
            j = ministry_idx[row['target']]
            influence_matrix[i, j] = row['influence_strength']

        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=(14, 12))

        labels = [m.replace('부', '') for m in ministries]
        sns.heatmap(influence_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': '영향력 강도'}, ax=ax)

        plt.title('탄소중립 2050: 부처 간 영향력 매트릭스', fontsize=14, fontweight='bold')
        plt.xlabel('대상 부처 (영향 받는 쪽)', fontsize=12)
        plt.ylabel('원천 부처 (영향 주는 쪽)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_path}carbon_neutral_influence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("탄소중립 2050 정책 확산 분석")
    print("=" * 70)

    # 데이터 로드
    adoption_file = 'chap/chapter06/data/carbon_neutral_adoption.csv'
    network_file = 'chap/chapter06/data/carbon_neutral_network.csv'

    analyzer = CarbonNeutralPolicyAnalysis(adoption_file, network_file)

    print(f"\n정책 확산 기간: {analyzer.start_date.date()} ~ {analyzer.end_date.date()}")
    print(f"총 기간: {analyzer.duration_days}일 ({analyzer.duration_days/30:.1f}개월)")
    print(f"참여 부처: {len(analyzer.adoption_df)}개")

    # 1. 네트워크 구축
    print("\n[1] 정책 확산 네트워크 구축")
    G = analyzer.build_temporal_network()
    print(f"노드 수: {G.number_of_nodes()}개 부처")
    print(f"엣지 수: {G.number_of_edges()}개 협력 관계")

    # 2. 확산 단계 분석
    print("\n[2] 정책 확산 단계 분석")
    stages_df = analyzer.identify_diffusion_stages()
    print(stages_df[['period', 'new_adopters', 'adoption_rate', 'total_budget']].to_string(index=False))

    # 3. 혁신자-추종자 분류
    print("\n[3] 혁신자-추종자 분류 (Rogers 확산 이론)")
    categories_df = analyzer.identify_innovators_and_followers()
    for category in categories_df['category'].unique():
        ministries = categories_df[categories_df['category'] == category]['ministry'].tolist()
        print(f"\n{category}:")
        print(f"  {', '.join(ministries)}")

    # 4. 영향력 경로 분석
    print("\n[4] 환경부로부터 영향력 전파 경로")
    paths_df = analyzer.analyze_influence_paths(G)
    print(paths_df[['target_ministry', 'path_length', 'path']].head(10).to_string(index=False))

    # 5. 네트워크 지표
    print("\n[5] 네트워크 지표")
    metrics = analyzer.calculate_network_metrics(G)
    print(f"네트워크 밀도: {metrics['density']:.3f}")
    print(f"평균 연결도: {metrics['avg_degree']:.2f}")
    print(f"\n상위 영향력 행사 부처 (Out-Centrality):")
    for ministry, score in metrics['top_out_centrality']:
        print(f"  {ministry}: {score:.3f}")
    print(f"\n상위 영향력 수용 부처 (In-Centrality):")
    for ministry, score in metrics['top_in_centrality']:
        print(f"  {ministry}: {score:.3f}")
    print(f"\n상위 매개 중심성 부처 (Betweenness):")
    for ministry, score in metrics['top_betweenness']:
        print(f"  {ministry}: {score:.3f}")

    # 6. 시각화
    print("\n[6] 시각화 생성 중...")
    analyzer.visualize_adoption_timeline(stages_df)
    analyzer.visualize_network(G, categories_df)
    analyzer.visualize_influence_heatmap()

    # 7. 결과 저장
    print("\n[7] 분석 결과 저장")
    stages_df.to_csv('chap/chapter06/outputs/carbon_neutral_stages.csv', index=False)
    categories_df.to_csv('chap/chapter06/outputs/carbon_neutral_categories.csv', index=False)
    paths_df.to_csv('chap/chapter06/outputs/carbon_neutral_paths.csv', index=False)

    print("\n✅ 분석 완료! 결과가 chap/chapter06/outputs/에 저장되었습니다.")

    return analyzer, G, stages_df, categories_df, paths_df

if __name__ == "__main__":
    analyzer, G, stages_df, categories_df, paths_df = main()
