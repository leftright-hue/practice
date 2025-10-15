#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제6장 그래프 이론과 정책 네트워크 분석
06-community-detection.py: 커뮤니티 탐지와 정책 연합 분석

Louvain 알고리즘과 기타 커뮤니티 탐지 기법을 사용하여
정부 부처 간 정책 연합을 식별하고 분석합니다.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.unicode_minus'] = False

# community 패키지가 없는 경우를 대비한 대안 구현
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    print("Warning: python-louvain 패키지가 없습니다. 대안 방법을 사용합니다.")
    LOUVAIN_AVAILABLE = False

def simple_greedy_communities(G):
    """
    간단한 탐욕적 커뮤니티 탐지 (community 패키지 대안)

    Args:
        G (nx.Graph): 무방향 그래프

    Returns:
        dict: 노드별 커뮤니티 ID
    """
    # NetworkX 내장 greedy modularity communities 사용
    communities = nx.community.greedy_modularity_communities(G)

    # 노드별 커뮤니티 ID 매핑
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i

    return partition

def calculate_modularity(G, partition):
    """
    모듈러리티 계산

    Args:
        G (nx.Graph): 무방향 그래프
        partition (dict): 노드별 커뮤니티 할당

    Returns:
        float: 모듈러리티 값
    """
    return nx.community.modularity(G, [set(nodes for nodes, comm_id in partition.items()
                                          if comm_id == i)
                                      for i in set(partition.values())])

def detect_policy_coalitions(G):
    """
    다양한 방법을 사용한 정책 연합 탐지

    Args:
        G (nx.DiGraph): 정부 네트워크

    Returns:
        dict: 탐지 결과들
    """
    print("정책 연합 탐지 중...")

    results = {}

    # 무방향 그래프로 변환 (대부분의 커뮤니티 탐지 알고리즘용)
    G_undirected = G.to_undirected()

    # 1. Louvain 알고리즘 (가능한 경우)
    if LOUVAIN_AVAILABLE:
        print("  - Louvain 알고리즘 실행 중...")
        louvain_partition = community_louvain.best_partition(G_undirected)
        louvain_modularity = community_louvain.modularity(louvain_partition, G_undirected)

        results['louvain'] = {
            'partition': louvain_partition,
            'modularity': louvain_modularity,
            'n_communities': len(set(louvain_partition.values()))
        }
        print(f"    Louvain 모듈러리티: {louvain_modularity:.3f}")
    else:
        print("  - Louvain 대신 Greedy 알고리즘 사용...")
        greedy_partition = simple_greedy_communities(G_undirected)
        greedy_modularity = calculate_modularity(G_undirected, greedy_partition)

        results['louvain'] = {
            'partition': greedy_partition,
            'modularity': greedy_modularity,
            'n_communities': len(set(greedy_partition.values()))
        }
        print(f"    Greedy 모듈러리티: {greedy_modularity:.3f}")

    # 2. Label Propagation
    print("  - Label Propagation 실행 중...")
    lp_communities = nx.community.label_propagation_communities(G_undirected)
    lp_partition = {}
    for i, community in enumerate(lp_communities):
        for node in community:
            lp_partition[node] = i

    lp_modularity = calculate_modularity(G_undirected, lp_partition)

    results['label_propagation'] = {
        'partition': lp_partition,
        'modularity': lp_modularity,
        'n_communities': len(set(lp_partition.values()))
    }
    print(f"    Label Propagation 모듈러리티: {lp_modularity:.3f}")

    # 3. Girvan-Newman (작은 네트워크에서만)
    if G_undirected.number_of_nodes() <= 20:
        print("  - Girvan-Newman 실행 중...")
        gn_communities = nx.community.girvan_newman(G_undirected)

        # 첫 번째 분할 사용
        try:
            first_partition = next(gn_communities)
            gn_partition = {}
            for i, community in enumerate(first_partition):
                for node in community:
                    gn_partition[node] = i

            gn_modularity = calculate_modularity(G_undirected, gn_partition)

            results['girvan_newman'] = {
                'partition': gn_partition,
                'modularity': gn_modularity,
                'n_communities': len(set(gn_partition.values()))
            }
            print(f"    Girvan-Newman 모듈러리티: {gn_modularity:.3f}")
        except:
            print("    Girvan-Newman 실행 실패")

    # 4. 기능적 분류 기반 커뮤니티 (도메인 지식 활용)
    print("  - 기능적 분류 기반 커뮤니티...")
    functional_partition = create_functional_communities(G)
    functional_modularity = calculate_modularity(G_undirected, functional_partition)

    results['functional'] = {
        'partition': functional_partition,
        'modularity': functional_modularity,
        'n_communities': len(set(functional_partition.values()))
    }
    print(f"    기능적 분류 모듈러리티: {functional_modularity:.3f}")

    return results

def create_functional_communities(G):
    """
    도메인 지식을 기반으로 한 기능적 커뮤니티 분류

    Args:
        G (nx.DiGraph): 정부 네트워크

    Returns:
        dict: 기능적 커뮤니티 분류
    """
    functional_groups = {
        # 경제 정책 그룹
        0: ['기획재정부', '산업통상자원부', '중소벤처기업부'],

        # 사회 정책 그룹
        1: ['보건복지부', '고용노동부', '여성가족부', '교육부'],

        # 기술/혁신 그룹
        2: ['과학기술정보통신부', '행정안전부'],

        # 인프라/환경 그룹
        3: ['국토교통부', '환경부', '해양수산부', '농림축산식품부'],

        # 안보/외교 그룹
        4: ['외교부', '통일부', '국방부', '법무부'],

        # 문화/체육 그룹
        5: ['문화체육관광부']
    }

    # 노드별 그룹 매핑
    partition = {}
    for group_id, ministries in functional_groups.items():
        for ministry in ministries:
            if ministry in G.nodes():
                partition[ministry] = group_id

    # 누락된 노드를 별도 그룹에 할당
    for node in G.nodes():
        if node not in partition:
            partition[node] = max(functional_groups.keys()) + 1

    return partition

def analyze_community_characteristics(G, partition, community_name):
    """
    커뮤니티 특성 분석

    Args:
        G (nx.DiGraph): 원본 네트워크
        partition (dict): 커뮤니티 분할
        community_name (str): 커뮤니티 탐지 방법명

    Returns:
        dict: 커뮤니티 분석 결과
    """
    print(f"\n{community_name} 커뮤니티 특성 분석 중...")

    # 커뮤니티별 구성원 정리
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    analysis_results = {}

    for comm_id, members in communities.items():
        # 커뮤니티 내부 그래프
        subgraph = G.subgraph(members).to_undirected()

        # 내부 연결성 분석
        internal_edges = subgraph.number_of_edges()
        possible_edges = len(members) * (len(members) - 1) / 2
        internal_density = internal_edges / possible_edges if possible_edges > 0 else 0

        # 외부 연결 분석
        external_edges = 0
        for member in members:
            for neighbor in G.neighbors(member):
                if neighbor not in members:
                    external_edges += 1

        # 총 협업 프로젝트 수
        total_projects = 0
        for source, target, data in G.edges(data=True):
            if source in members and target in members:
                total_projects += data.get('projects', 0)

        # 커뮤니티 중심성 (구성원들의 평균 중심성)
        try:
            centralities = nx.degree_centrality(G)
            avg_centrality = np.mean([centralities[member] for member in members])
        except:
            avg_centrality = 0

        analysis_results[comm_id] = {
            'members': members,
            'size': len(members),
            'internal_edges': internal_edges,
            'internal_density': internal_density,
            'external_edges': external_edges,
            'total_projects': total_projects,
            'avg_centrality': avg_centrality
        }

        print(f"커뮤니티 {comm_id + 1}:")
        print(f"  구성원 ({len(members)}개): {', '.join([m.replace('부', '').replace('청', '') for m in members])}")
        print(f"  내부 밀도: {internal_density:.3f}")
        print(f"  총 프로젝트: {total_projects}개")
        print(f"  평균 중심성: {avg_centrality:.3f}")

    return analysis_results

def identify_policy_themes(G, partition):
    """
    커뮤니티별 정책 주제 식별

    Args:
        G (nx.DiGraph): 정부 네트워크
        partition (dict): 커뮤니티 분할

    Returns:
        dict: 커뮤니티별 정책 주제
    """
    print("\n정책 주제 식별 중...")

    # 커뮤니티별 구성원과 연결 유형 분석
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    policy_themes = {}

    for comm_id, members in communities.items():
        # 커뮤니티 내 엣지 유형 분석
        edge_types = defaultdict(int)
        for source, target, data in G.edges(data=True):
            if source in members and target in members:
                edge_types[data.get('type', 'unknown')] += 1

        # 주요 정책 유형 식별
        if edge_types:
            dominant_type = max(edge_types, key=edge_types.get)
        else:
            dominant_type = 'isolated'

        # 정책 주제 라벨링
        theme_mapping = {
            'AI_budget': 'AI 예산 정책',
            'digital_education': '디지털 교육',
            'digital_government': '디지털 정부',
            'economic_policy': '경제 정책',
            'social_safety': '사회 안전망',
            'green_transition': '녹색 전환',
            'foreign_policy': '외교 정책',
            'health_AI': '헬스케어 AI'
        }

        policy_theme = theme_mapping.get(dominant_type, '기타 정책')

        policy_themes[comm_id] = {
            'theme': policy_theme,
            'dominant_type': dominant_type,
            'edge_types': dict(edge_types),
            'theme_strength': edge_types[dominant_type] / sum(edge_types.values()) if edge_types else 0
        }

        print(f"커뮤니티 {comm_id + 1}: {policy_theme} "
              f"(강도: {policy_themes[comm_id]['theme_strength']:.2f})")

    return policy_themes

def visualize_communities(G, results, save_path=None):
    """
    커뮤니티 구조 시각화

    Args:
        G (nx.DiGraph): 원본 네트워크
        results (dict): 커뮤니티 탐지 결과
        save_path (str): 저장 경로
    """
    print("\n커뮤니티 시각화 생성 중...")

    # 최고 모듈러리티를 가진 방법 선택
    best_method = max(results.keys(), key=lambda x: results[x]['modularity'])
    best_partition = results[best_method]['partition']
    best_modularity = results[best_method]['modularity']

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # 색상 팔레트
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#F0E68C', '#87CEEB', '#FFB6C1', '#98FB98']

    # 1. 최적 커뮤니티 시각화
    ax = axes[0]
    G_undirected = G.to_undirected()
    pos = nx.spring_layout(G_undirected, k=3, iterations=50, seed=42)

    # 커뮤니티별 색상으로 노드 그리기
    for node in G_undirected.nodes():
        comm_id = best_partition[node]
        nx.draw_networkx_nodes(G_undirected, pos,
                             nodelist=[node],
                             node_color=colors[comm_id % len(colors)],
                             node_size=1000,
                             alpha=0.8,
                             ax=ax)

    # 엣지 그리기
    nx.draw_networkx_edges(G_undirected, pos, alpha=0.5, width=0.5, ax=ax)

    # 라벨 추가
    labels = {node: node.replace('부', '').replace('청', '') for node in G_undirected.nodes()}
    nx.draw_networkx_labels(G_undirected, pos, labels, font_size=8, ax=ax)

    ax.set_title(f'{best_method.title()} 커뮤니티 구조\n'
                f'(모듈러리티: {best_modularity:.3f})', fontweight='bold')
    ax.axis('off')

    # 2. 커뮤니티 크기 분포
    ax = axes[1]
    community_sizes = defaultdict(int)
    for comm_id in best_partition.values():
        community_sizes[comm_id] += 1

    sizes = list(community_sizes.values())
    comm_ids = [f'C{i+1}' for i in range(len(sizes))]

    bars = ax.bar(comm_ids, sizes, color=[colors[i % len(colors)] for i in range(len(sizes))])
    ax.set_xlabel('커뮤니티')
    ax.set_ylabel('구성원 수')
    ax.set_title('커뮤니티 크기 분포', fontweight='bold')

    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom')

    # 3. 모듈러리티 비교
    ax = axes[2]
    methods = list(results.keys())
    modularities = [results[method]['modularity'] for method in methods]

    bars = ax.bar(methods, modularities, color='steelblue')
    ax.set_xlabel('탐지 방법')
    ax.set_ylabel('모듈러리티')
    ax.set_title('커뮤니티 탐지 방법별 모듈러리티 비교', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # 값 표시
    for bar, mod in zip(bars, modularities):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mod:.3f}', ha='center', va='bottom')

    # 4. 커뮤니티 간 연결 히트맵
    ax = axes[3]

    # 커뮤니티 간 연결 행렬 생성
    n_communities = len(set(best_partition.values()))
    connection_matrix = np.zeros((n_communities, n_communities))

    for source, target, data in G.edges(data=True):
        source_comm = best_partition[source]
        target_comm = best_partition[target]
        connection_matrix[source_comm][target_comm] += data.get('projects', 1)

    # 대칭 행렬로 만들기 (무방향 그래프처럼)
    connection_matrix = connection_matrix + connection_matrix.T

    # 히트맵
    sns.heatmap(connection_matrix, annot=True, fmt='.0f', cmap='Blues',
                square=True, ax=ax,
                xticklabels=[f'C{i+1}' for i in range(n_communities)],
                yticklabels=[f'C{i+1}' for i in range(n_communities)])
    ax.set_title('커뮤니티 간 협업 프로젝트 수', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/community_analysis.png",
                   dpi=300, bbox_inches='tight')
        print(f"커뮤니티 시각화 저장: {save_path}/community_analysis.png")

    plt.show()

def create_community_network_layout(G, partition, save_path=None):
    """
    커뮤니티 구조를 강조한 네트워크 레이아웃

    Args:
        G (nx.DiGraph): 원본 네트워크
        partition (dict): 커뮤니티 분할
        save_path (str): 저장 경로
    """
    print("\n커뮤니티 네트워크 레이아웃 생성 중...")

    fig, ax = plt.subplots(figsize=(16, 12))

    G_undirected = G.to_undirected()

    # 커뮤니티별 위치 설정 (circular layout for each community)
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    pos = {}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#F0E68C', '#87CEEB', '#FFB6C1', '#98FB98']

    # 커뮤니티를 원형으로 배치
    n_communities = len(communities)
    community_angles = np.linspace(0, 2*np.pi, n_communities, endpoint=False)

    for i, (comm_id, members) in enumerate(communities.items()):
        # 각 커뮤니티의 중심 위치
        center_x = 5 * np.cos(community_angles[i])
        center_y = 5 * np.sin(community_angles[i])

        # 커뮤니티 내 노드들을 작은 원형으로 배치
        if len(members) == 1:
            pos[members[0]] = (center_x, center_y)
        else:
            member_angles = np.linspace(0, 2*np.pi, len(members), endpoint=False)
            radius = 1.5
            for j, member in enumerate(members):
                pos[member] = (center_x + radius * np.cos(member_angles[j]),
                              center_y + radius * np.sin(member_angles[j]))

    # 노드 크기 (중심성 기반)
    centralities = nx.degree_centrality(G)
    node_sizes = [centralities[node] * 2000 + 300 for node in G_undirected.nodes()]

    # 커뮤니티별 색상으로 노드 그리기
    for i, (comm_id, members) in enumerate(communities.items()):
        nodelist = [node for node in members if node in G_undirected.nodes()]
        node_sizes_comm = [centralities[node] * 2000 + 300 for node in nodelist]

        nx.draw_networkx_nodes(G_undirected, pos,
                             nodelist=nodelist,
                             node_color=colors[i % len(colors)],
                             node_size=node_sizes_comm,
                             alpha=0.8,
                             edgecolors='black',
                             linewidths=1.5)

    # 엣지 그리기 (커뮤니티 내부 vs 외부)
    internal_edges = []
    external_edges = []

    for edge in G_undirected.edges():
        source, target = edge
        if partition[source] == partition[target]:
            internal_edges.append(edge)
        else:
            external_edges.append(edge)

    # 내부 엣지 (두껍고 진한 색)
    nx.draw_networkx_edges(G_undirected, pos,
                          edgelist=internal_edges,
                          alpha=0.8,
                          width=2,
                          edge_color='darkgray')

    # 외부 엣지 (얇고 연한 색)
    nx.draw_networkx_edges(G_undirected, pos,
                          edgelist=external_edges,
                          alpha=0.4,
                          width=1,
                          edge_color='lightgray',
                          style='dashed')

    # 라벨 추가
    labels = {node: node.replace('부', '').replace('청', '') for node in G_undirected.nodes()}
    nx.draw_networkx_labels(G_undirected, pos, labels, font_size=10, font_weight='bold')

    # 커뮤니티 라벨 추가
    for i, (comm_id, members) in enumerate(communities.items()):
        center_x = 5 * np.cos(community_angles[i])
        center_y = 5 * np.sin(community_angles[i])

        # 커뮤니티 이름 (정책 주제 기반)
        community_names = {
            0: '경제정책연합',
            1: '사회정책연합',
            2: '기술혁신연합',
            3: '인프라환경연합',
            4: '안보외교연합',
            5: '문화체육연합'
        }

        comm_name = community_names.get(comm_id, f'연합{comm_id+1}')

        plt.text(center_x, center_y - 2.5, comm_name,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor=colors[i % len(colors)],
                         alpha=0.7),
                fontsize=12, fontweight='bold')

    plt.title('정부 부처 정책 연합 구조\n'
             '(실선: 연합 내부 협력, 점선: 연합 간 협력)',
             fontsize=16, fontweight='bold', pad=20)

    # 범례
    legend_elements = [
        plt.Line2D([0], [0], color='darkgray', linewidth=3, label='연합 내부 협력'),
        plt.Line2D([0], [0], color='lightgray', linewidth=2, linestyle='--', label='연합 간 협력')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/community_layout.png",
                   dpi=300, bbox_inches='tight')
        print(f"커뮤니티 레이아웃 저장: {save_path}/community_layout.png")

    plt.show()

def export_community_results(results, community_analyses, policy_themes, save_path):
    """
    커뮤니티 분석 결과 내보내기

    Args:
        results (dict): 커뮤니티 탐지 결과
        community_analyses (dict): 커뮤니티 특성 분석
        policy_themes (dict): 정책 주제 분석
        save_path (str): 저장 경로
    """
    print("\n커뮤니티 분석 결과 내보내기 중...")

    # 1. 요약 결과 CSV
    summary_data = []
    for method, result in results.items():
        summary_data.append({
            'method': method,
            'modularity': result['modularity'],
            'n_communities': result['n_communities']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{save_path}/community_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

    # 2. 상세 커뮤니티 정보 Excel
    excel_path = f"{save_path}/community_detailed.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 방법별 커뮤니티 할당
        for method, result in results.items():
            partition_data = []
            for node, comm_id in result['partition'].items():
                partition_data.append({
                    'ministry': node,
                    'community_id': comm_id
                })

            partition_df = pd.DataFrame(partition_data)
            sheet_name = method.replace('_', ' ').title()[:31]
            partition_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 커뮤니티 특성 분석 결과
        if community_analyses:
            char_data = []
            for comm_id, analysis in community_analyses.items():
                char_data.append({
                    'community_id': comm_id,
                    'size': analysis['size'],
                    'internal_density': analysis['internal_density'],
                    'total_projects': analysis['total_projects'],
                    'avg_centrality': analysis['avg_centrality'],
                    'members': ', '.join(analysis['members'])
                })

            char_df = pd.DataFrame(char_data)
            char_df.to_excel(writer, sheet_name='특성분석', index=False)

    print(f"커뮤니티 분석 결과 저장: {excel_path}")

    # 3. 정책 주제 요약
    theme_path = f"{save_path}/policy_themes.txt"
    with open(theme_path, 'w', encoding='utf-8') as f:
        f.write("=== 정책 연합별 주제 분석 ===\n\n")

        if policy_themes:
            for comm_id, theme_info in policy_themes.items():
                f.write(f"연합 {comm_id + 1}: {theme_info['theme']}\n")
                f.write(f"  주도 유형: {theme_info['dominant_type']}\n")
                f.write(f"  주제 강도: {theme_info['theme_strength']:.2f}\n")
                f.write(f"  세부 유형: {theme_info['edge_types']}\n\n")

    print(f"정책 주제 분석 저장: {theme_path}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제6장: 커뮤니티 탐지와 정책 연합 분석")
    print("=" * 60)

    # 이전 단계에서 생성된 네트워크 불러오기
    try:
        from importlib import import_module
        import sys
        sys.path.append('.')
        gov_network_module = import_module('06-government-network')
        gov_network = gov_network_module.create_government_network()
        print("기존 네트워크 불러오기 성공")
    except Exception as e:
        print(f"네트워크 불러오기 실패: {e}")
        print("네트워크를 새로 생성합니다...")
        # 간단한 네트워크 생성 (대안)
        gov_network = nx.DiGraph()
        ministries = ['기획재정부', '교육부', '과학기술정보통신부', '보건복지부']
        gov_network.add_nodes_from(ministries)
        gov_network.add_edge('기획재정부', '과학기술정보통신부', projects=10, type='AI_budget')
        gov_network.add_edge('교육부', '과학기술정보통신부', projects=8, type='digital_education')

    # 1. 다양한 방법으로 커뮤니티 탐지
    detection_results = detect_policy_coalitions(gov_network)

    # 2. 최적 방법 선택 (모듈러리티 기준)
    best_method = max(detection_results.keys(),
                     key=lambda x: detection_results[x]['modularity'])
    best_partition = detection_results[best_method]['partition']

    print(f"\n최적 커뮤니티 탐지 방법: {best_method}")
    print(f"모듈러리티: {detection_results[best_method]['modularity']:.3f}")

    # 3. 커뮤니티 특성 분석
    community_analyses = analyze_community_characteristics(
        gov_network, best_partition, best_method)

    # 4. 정책 주제 식별
    policy_themes = identify_policy_themes(gov_network, best_partition)

    # 5. 시각화
    visualize_communities(gov_network, detection_results,
                         'chap/chapter06/outputs')
    create_community_network_layout(gov_network, best_partition,
                                   'chap/chapter06/outputs')

    # 6. 결과 내보내기
    export_community_results(detection_results, community_analyses,
                            policy_themes, 'chap/chapter06/data')

    print("\n커뮤니티 분석 완료! 결과가 chap/chapter06/ 디렉토리에 저장되었습니다.")

    return detection_results, community_analyses, policy_themes

if __name__ == "__main__":
    # 실행
    community_results, community_chars, themes = main()

    # 결과를 전역 변수로 저장
    globals()['community_detection_results'] = community_results
    globals()['community_characteristics'] = community_chars
    globals()['policy_themes'] = themes