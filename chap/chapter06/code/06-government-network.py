# -*- coding: utf-8 -*-
"""
제6장 그래프 이론과 정책 네트워크 분석
06-government-network.py: 한국 정부 부처 간 협업 네트워크 생성 및 기본 분석

한국 정부 19개 부처 간의 협업 네트워크를 모델링하고 기본 속성을 분석합니다.
실제 정부 부처 간 협업 사례를 기반으로 네트워크 분석 방법론을 시연합니다.
"""

# 한글 출력 인코딩 문제 해결 (Windows 환경)
import sys
import os

# Windows 콘솔 코드페이지를 UTF-8로 설정
if sys.platform == "win32":
    try:
        os.system("chcp 65001 > nul")
    except:
        pass

# Python UTF-8 모드 강제 활성화
os.environ['PYTHONUTF8'] = '1'

# stdout/stderr 인코딩을 UTF-8로 재구성
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:
    # 구 버전 Python 호환성
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import json
from collections import defaultdict

# 폰트 설정
import matplotlib
matplotlib.use('TkAgg')  # Windows에서 이미지 표시를 위한 백엔드 설정
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
def setup_korean_font():
    korean_fonts = ['Malgun Gothic', 'NanumGothic', 'Batang', 'Gulim', 'MS Gothic']
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    selected_font = None
    for font in korean_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        print(f"한글 폰트 설정됨: {selected_font}")
    else:
        plt.rcParams['font.family'] = ['DejaVu Sans']
        print("기본 폰트 사용")
    
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 실행
setup_korean_font()
plt.rcParams['figure.figsize'] = (12, 8)

def create_government_network():
    """
    한국 정부 19개 부처 간 협업 네트워크 생성 예제

    Returns:
        nx.DiGraph: 정부 부처 간 협업 네트워크
    """
    print("한국 정부 부처 간 협업 네트워크 생성 중...")

    G = nx.DiGraph()

    # 정부 부처 노드 추가 (2024년 기준 19개 부처)
    ministries = [
        '기획재정부', '교육부', '과학기술정보통신부', '외교부',
        '통일부', '법무부', '국방부', '행정안전부',
        '문화체육관광부', '농림축산식품부', '산업통상자원부', '보건복지부',
        '환경부', '고용노동부', '여성가족부', '국토교통부',
        '해양수산부', '중소벤처기업부', '국가보훈부'
    ]

    # 노드 속성과 함께 추가
    for ministry in ministries:
        G.add_node(ministry,
                  node_type='ministry',
                  establishment_year=get_ministry_info(ministry)['year'],
                  budget_scale=get_ministry_info(ministry)['budget'],
                  staff_size=get_ministry_info(ministry)['staff'])

    # 협업 관계 추가 (177개 프로젝트를 기반으로 한 주요 협업 관계)
    collaborations = [
        # AI 및 디지털 전환 관련 협업
        ('기획재정부', '과학기술정보통신부', {'weight': 0.9, 'type': 'AI_budget', 'projects': 15}),
        ('과학기술정보통신부', '교육부', {'weight': 0.7, 'type': 'digital_education', 'projects': 12}),
        ('과학기술정보통신부', '행정안전부', {'weight': 0.9, 'type': 'digital_government', 'projects': 18}),
        ('과학기술정보통신부', '산업통상자원부', {'weight': 0.6, 'type': 'digital_industry', 'projects': 8}),
        ('과학기술정보통신부', '보건복지부', {'weight': 0.8, 'type': 'health_AI', 'projects': 10}),
        ('과학기술정보통신부', '국토교통부', {'weight': 0.7, 'type': 'smart_city', 'projects': 9}),

        # 경제 정책 관련 협업
        ('기획재정부', '산업통상자원부', {'weight': 0.8, 'type': 'economic_policy', 'projects': 20}),
        ('기획재정부', '중소벤처기업부', {'weight': 0.6, 'type': 'startup_support', 'projects': 8}),
        ('산업통상자원부', '중소벤처기업부', {'weight': 0.7, 'type': 'industry_support', 'projects': 11}),
        ('기획재정부', '고용노동부', {'weight': 0.5, 'type': 'employment_policy', 'projects': 7}),

        # 사회 정책 관련 협업
        ('교육부', '고용노동부', {'weight': 0.6, 'type': 'workforce_development', 'projects': 9}),
        ('보건복지부', '고용노동부', {'weight': 0.7, 'type': 'social_safety', 'projects': 13}),
        ('보건복지부', '여성가족부', {'weight': 0.8, 'type': 'family_welfare', 'projects': 11}),
        ('고용노동부', '여성가족부', {'weight': 0.6, 'type': 'gender_employment', 'projects': 6}),

        # 환경 및 녹색 정책 관련 협업
        ('환경부', '산업통상자원부', {'weight': 0.5, 'type': 'green_transition', 'projects': 14}),
        ('환경부', '국토교통부', {'weight': 0.6, 'type': 'green_transport', 'projects': 10}),
        ('환경부', '농림축산식품부', {'weight': 0.6, 'type': 'sustainable_agriculture', 'projects': 7}),

        # 문화 및 관광 관련 협업
        ('문화체육관광부', '과학기술정보통신부', {'weight': 0.4, 'type': 'digital_culture', 'projects': 5}),
        ('문화체육관광부', '교육부', {'weight': 0.5, 'type': 'cultural_education', 'projects': 6}),

        # 안보 및 외교 관련 협업
        ('외교부', '통일부', {'weight': 0.8, 'type': 'foreign_policy', 'projects': 12}),
        ('외교부', '국방부', {'weight': 0.7, 'type': 'defense_diplomacy', 'projects': 8}),
        ('통일부', '행정안전부', {'weight': 0.4, 'type': 'unification_admin', 'projects': 3}),

        # 해양 및 수산 관련 협업
        ('해양수산부', '환경부', {'weight': 0.5, 'type': 'marine_environment', 'projects': 6}),
        ('해양수산부', '외교부', {'weight': 0.4, 'type': 'maritime_diplomacy', 'projects': 4}),

        # 행정 조정 관련 협업
        ('행정안전부', '기획재정부', {'weight': 0.6, 'type': 'government_operation', 'projects': 10}),
        ('행정안전부', '법무부', {'weight': 0.5, 'type': 'legal_administration', 'projects': 7})
    ]

    G.add_edges_from(collaborations)

    print(f"네트워크 생성 완료: {G.number_of_nodes()}개 부처, {G.number_of_edges()}개 협업 관계")
    return G

def get_ministry_info(ministry):
    """
    부처별 기본 정보 반환 (간단한 시뮬레이션)

    Args:
        ministry (str): 부처명

    Returns:
        dict: 부처 정보
    """
    ministry_data = {
        '기획재정부': {'year': 1948, 'budget': 1.0, 'staff': 2500},
        '교육부': {'year': 1948, 'budget': 0.8, 'staff': 2000},
        '과학기술정보통신부': {'year': 2017, 'budget': 0.7, 'staff': 1800},
        '외교부': {'year': 1948, 'budget': 0.3, 'staff': 1200},
        '통일부': {'year': 1969, 'budget': 0.2, 'staff': 800},
        '법무부': {'year': 1948, 'budget': 0.4, 'staff': 1500},
        '국방부': {'year': 1948, 'budget': 0.9, 'staff': 3000},
        '행정안전부': {'year': 2008, 'budget': 0.6, 'staff': 1600},
        '문화체육관광부': {'year': 1948, 'budget': 0.3, 'staff': 1000},
        '농림축산식품부': {'year': 1948, 'budget': 0.5, 'staff': 1400},
        '산업통상자원부': {'year': 1948, 'budget': 0.6, 'staff': 1700},
        '보건복지부': {'year': 1948, 'budget': 0.8, 'staff': 2200},
        '환경부': {'year': 1994, 'budget': 0.4, 'staff': 1100},
        '고용노동부': {'year': 1981, 'budget': 0.5, 'staff': 1300},
        '여성가족부': {'year': 2001, 'budget': 0.2, 'staff': 600},
        '국토교통부': {'year': 2013, 'budget': 0.7, 'staff': 1900},
        '해양수산부': {'year': 2013, 'budget': 0.3, 'staff': 900},
        '중소벤처기업부': {'year': 2017, 'budget': 0.4, 'staff': 1000}
    }

    return ministry_data.get(ministry, {'year': 2000, 'budget': 0.3, 'staff': 1000})

def analyze_network_properties(G):
    """
    네트워크의 기본 속성 분석

    Args:
        G (nx.DiGraph): 분석할 네트워크

    Returns:
        dict: 분석 결과
    """
    print("\n네트워크 기본 속성 분석 중...")

    # 기본 네트워크 통계
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    # 연결성 분석
    is_connected = nx.is_strongly_connected(G)
    n_components = nx.number_strongly_connected_components(G)
    largest_component = max(nx.strongly_connected_components(G), key=len)

    # 경로 분석 (강연결 성분 내에서)
    if len(largest_component) > 1:
        subgraph = G.subgraph(largest_component)
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
        except:
            avg_path_length = 0
            diameter = 0
    else:
        avg_path_length = 0
        diameter = 0

    # 클러스터링 계수
    try:
        clustering_coeff = nx.average_clustering(G.to_undirected())
    except:
        clustering_coeff = 0

    # 협업 프로젝트 통계
    total_projects = sum([data['projects'] for _, _, data in G.edges(data=True)])
    avg_projects_per_edge = total_projects / n_edges if n_edges > 0 else 0

    analysis_results = {
        'basic_stats': {
            'nodes': n_nodes,
            'edges': n_edges,
            'density': density,
            'total_projects': total_projects,
            'avg_projects_per_edge': avg_projects_per_edge
        },
        'connectivity': {
            'is_strongly_connected': is_connected,
            'n_components': n_components,
            'largest_component_size': len(largest_component)
        },
        'structure': {
            'avg_path_length': avg_path_length,
            'diameter': diameter,
            'clustering_coefficient': clustering_coeff
        }
    }

    return analysis_results

def visualize_network(G, save_path=None):
    """
    네트워크 시각화

    Args:
        G (nx.DiGraph): 시각화할 네트워크
        save_path (str): 저장 경로 (선택사항)
    """
    print("\n네트워크 시각화 생성 중...")

    plt.figure(figsize=(15, 10))

    # 레이아웃 설정 (spring layout with adjusted parameters)
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

    # 노드 크기 (예산 규모에 따라)
    node_sizes = []
    for node in G.nodes():
        budget = G.nodes[node]['budget_scale']
        node_sizes.append(budget * 1000 + 500)

    # 엣지 두께 (협업 강도에 따라)
    edge_widths = []
    for _, _, data in G.edges(data=True):
        edge_widths.append(data['weight'] * 3)

    # 노드 색상 (부처 유형에 따라)
    node_colors = []
    color_map = {
        '기획재정부': '#FF6B6B',      # 빨강 (경제)
        '교육부': '#4ECDC4',          # 청록 (교육)
        '과학기술정보통신부': '#45B7D1', # 파랑 (기술)
        '보건복지부': '#96CEB4',      # 연녹 (복지)
        '환경부': '#FFEAA7',          # 노랑 (환경)
        '외교부': '#DDA0DD',          # 자주 (외교)
        '국방부': '#8B4513'           # 갈색 (국방)
    }

    for node in G.nodes():
        if node in color_map:
            node_colors.append(color_map[node])
        else:
            node_colors.append('#D3D3D3')  # 회색 (기타)

    # 네트워크 그리기
    nx.draw_networkx_nodes(G, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1)

    nx.draw_networkx_edges(G, pos,
                          width=edge_widths,
                          alpha=0.6,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->')

    # 라벨 추가 (부처명 간략화)
    labels = {}
    for node in G.nodes():
        if '부' in node:
            labels[node] = node.replace('부', '')
        elif '청' in node:
            labels[node] = node.replace('청', '')
        else:
            labels[node] = node[:4]  # 처음 4글자만

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title('Korean Government Ministry Collaboration Network\n(Node size: Budget scale, Edge width: Collaboration intensity)',
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')

    # 범례 추가
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                   markersize=10, label='Economic'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1',
                   markersize=10, label='Technology'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#96CEB4',
                   markersize=10, label='Welfare'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3',
                   markersize=10, label='Others')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"네트워크 시각화 저장: {save_path}")

    plt.show()  # 이미지 화면에 표시

def export_network_data(G, base_path):
    """
    네트워크 데이터를 다양한 형식으로 내보내기

    Args:
        G (nx.DiGraph): 내보낼 네트워크
        base_path (str): 기본 저장 경로
    """
    print("\n네트워크 데이터 내보내기 중...")

    # 디렉토리 생성
    os.makedirs(base_path, exist_ok=True)

    # GraphML 형식으로 저장
    graphml_path = f"{base_path}/government_network.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"GraphML 저장: {graphml_path}")

    # JSON 형식으로 저장
    json_data = nx.node_link_data(G)
    json_path = f"{base_path}/government_network.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {json_path}")

    # CSV 형식으로 노드와 엣지 정보 저장
    # 노드 정보
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        nodes_data.append({
            'ministry': node,
            'establishment_year': attrs['establishment_year'],
            'budget_scale': attrs['budget_scale'],
            'staff_size': attrs['staff_size']
        })

    nodes_df = pd.DataFrame(nodes_data)
    nodes_csv_path = f"{base_path}/government_nodes.csv"
    nodes_df.to_csv(nodes_csv_path, index=False, encoding='utf-8')
    print(f"노드 CSV 저장: {nodes_csv_path}")

    # 엣지 정보
    edges_data = []
    for source, target, attrs in G.edges(data=True):
        edges_data.append({
            'source': source,
            'target': target,
            'weight': attrs['weight'],
            'type': attrs['type'],
            'projects': attrs['projects']
        })

    edges_df = pd.DataFrame(edges_data)
    edges_csv_path = f"{base_path}/government_edges.csv"
    edges_df.to_csv(edges_csv_path, index=False, encoding='utf-8')
    print(f"엣지 CSV 저장: {edges_csv_path}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제6장: 한국 정부 부처 간 협업 네트워크 분석")
    print("=" * 60)

    # 네트워크 생성
    gov_network = create_government_network()

    # 네트워크 속성 분석
    network_props = analyze_network_properties(gov_network)

    # 분석 결과 출력
    print("\n=== 한국 정부 부처 협업 네트워크 분석 결과 ===")
    for category, metrics in network_props.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")

    # 주요 통계 요약
    print(f"\n주요 발견사항:")
    print(f"• 총 {network_props['basic_stats']['total_projects']}개의 협업 프로젝트")
    print(f"• 평균 {network_props['basic_stats']['avg_projects_per_edge']:.1f}개 프로젝트/협업관계")
    print(f"• 네트워크 밀도: {network_props['basic_stats']['density']:.3f}")
    print(f"• 클러스터링 계수: {network_props['structure']['clustering_coefficient']:.3f}")

    # 네트워크 시각화
    visualize_network(gov_network, '../outputs/government_network.png')

    # 데이터 내보내기
    export_network_data(gov_network, '../data')

    print("\n분석 완료! 모든 결과가 practice/chapter06/ 디렉토리에 저장되었습니다.")

    return gov_network, network_props

if __name__ == "__main__":
    # 실행
    government_network, analysis_results = main()

    # 추가 분석을 위해 전역 변수로 저장
    globals()['gov_network'] = government_network
    globals()['network_analysis'] = analysis_results