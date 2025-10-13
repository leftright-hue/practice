#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제6장 그래프 이론과 정책 네트워크 분석
06-centrality-analysis.py: 네트워크 중심성 분석과 정책 영향력 측정

다양한 중심성 지표(연결, 근접, 매개, 고유벡터)를 계산하고
정부 부처의 정책 영향력을 종합적으로 분석합니다.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.unicode_minus'] = False

def calculate_all_centralities(G):
    """
    모든 중심성 지표를 계산하는 종합 함수

    Args:
        G (nx.DiGraph): 분석할 네트워크

    Returns:
        pd.DataFrame: 중심성 분석 결과
    """
    print("다양한 중심성 지표 계산 중...")

    # 기본 연결 중심성
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    # 근접 중심성과 매개 중심성 (강연결 성분에서만 계산 가능)
    largest_scc = max(nx.strongly_connected_components(G), key=len)

    if len(largest_scc) > 1:
        scc_subgraph = G.subgraph(largest_scc)
        closeness_centrality = nx.closeness_centrality(scc_subgraph)
        betweenness_centrality = nx.betweenness_centrality(scc_subgraph)

        # 전체 그래프에 대해서도 계산 (연결되지 않은 노드는 0)
        full_closeness = {}
        full_betweenness = {}

        for node in G.nodes():
            if node in closeness_centrality:
                full_closeness[node] = closeness_centrality[node]
                full_betweenness[node] = betweenness_centrality[node]
            else:
                full_closeness[node] = 0.0
                full_betweenness[node] = 0.0

        closeness_centrality = full_closeness
        betweenness_centrality = full_betweenness
    else:
        closeness_centrality = {node: 0.0 for node in G.nodes()}
        betweenness_centrality = {node: 0.0 for node in G.nodes()}

    # 고유벡터 중심성 (방향 그래프에서는 PageRank 사용)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        # 수렴하지 않는 경우 PageRank 사용
        eigenvector_centrality = nx.pagerank(G, alpha=0.85)

    # Katz 중심성
    try:
        katz_centrality = nx.katz_centrality(G, alpha=0.1)
    except:
        katz_centrality = {node: 0.0 for node in G.nodes()}

    # 실제 협업 프로젝트 수를 기반으로 한 가중 중심성
    weighted_in_degree = {}
    weighted_out_degree = {}

    for node in G.nodes():
        # 들어오는 협업의 총 프로젝트 수
        in_projects = sum([G[source][node]['projects']
                          for source in G.predecessors(node)])
        # 나가는 협업의 총 프로젝트 수
        out_projects = sum([G[node][target]['projects']
                           for target in G.successors(node)])

        weighted_in_degree[node] = in_projects
        weighted_out_degree[node] = out_projects

    # 정규화된 가중 중심성
    max_in_projects = max(weighted_in_degree.values()) if weighted_in_degree.values() else 1
    max_out_projects = max(weighted_out_degree.values()) if weighted_out_degree.values() else 1

    normalized_weighted_in = {node: val/max_in_projects for node, val in weighted_in_degree.items()}
    normalized_weighted_out = {node: val/max_out_projects for node, val in weighted_out_degree.items()}

    # 결과를 DataFrame으로 정리
    centrality_data = []
    for node in G.nodes():
        centrality_data.append({
            'ministry': node,
            'degree': degree_centrality[node],
            'in_degree': in_degree_centrality[node],
            'out_degree': out_degree_centrality[node],
            'closeness': closeness_centrality[node],
            'betweenness': betweenness_centrality[node],
            'eigenvector': eigenvector_centrality[node],
            'katz': katz_centrality[node],
            'weighted_in_degree': normalized_weighted_in[node],
            'weighted_out_degree': normalized_weighted_out[node],
            'total_in_projects': weighted_in_degree[node],
            'total_out_projects': weighted_out_degree[node]
        })

    df = pd.DataFrame(centrality_data)

    print(f"중심성 분석 완료: {len(df)}개 부처 분석")
    return df

def analyze_centrality_rankings(df):
    """
    중심성 지표별 순위 분석

    Args:
        df (pd.DataFrame): 중심성 데이터

    Returns:
        dict: 순위 분석 결과
    """
    print("\n중심성 지표별 순위 분석 중...")

    rankings = {}
    centrality_cols = ['degree', 'in_degree', 'out_degree', 'closeness',
                      'betweenness', 'eigenvector', 'weighted_in_degree', 'weighted_out_degree']

    for col in centrality_cols:
        # 내림차순으로 정렬
        ranked = df.nlargest(5, col)[['ministry', col]]
        rankings[col] = ranked

        print(f"\n{col.upper()} 상위 5개 부처:")
        for i, (idx, row) in enumerate(ranked.iterrows(), 1):
            print(f"  {i}. {row['ministry']:<20}: {row[col]:.3f}")

    return rankings

def calculate_influence_score(df):
    """
    종합 정책 영향력 점수 계산

    Args:
        df (pd.DataFrame): 중심성 데이터

    Returns:
        pd.DataFrame: 영향력 점수가 추가된 데이터
    """
    print("\n종합 정책 영향력 점수 계산 중...")

    # 가중치 설정 (정책 영향력 관점에서)
    weights = {
        'degree': 0.15,           # 전체 연결성
        'in_degree': 0.10,        # 받는 영향
        'out_degree': 0.15,       # 주는 영향
        'closeness': 0.15,        # 전체 접근성
        'betweenness': 0.25,      # 중재 능력 (정책 조정에서 중요)
        'eigenvector': 0.20       # 네트워크 지위 (영향력 있는 부처와의 연결)
    }

    # 정규화된 점수 계산
    normalized_df = df.copy()
    for col in weights.keys():
        max_val = df[col].max()
        if max_val > 0:
            normalized_df[f'{col}_norm'] = df[col] / max_val
        else:
            normalized_df[f'{col}_norm'] = 0

    # 종합 영향력 점수 계산
    influence_score = 0
    for col, weight in weights.items():
        influence_score += normalized_df[f'{col}_norm'] * weight

    df['influence_score'] = influence_score

    # 영향력 순위
    influence_ranking = df.sort_values('influence_score', ascending=False)

    print("=== 종합 정책 영향력 순위 ===")
    for i, (idx, row) in enumerate(influence_ranking.iterrows(), 1):
        print(f"{i:2d}. {row['ministry']:<20} {row['influence_score']:.3f}")

    # 영향력 그룹 분류
    high_influence = influence_ranking[influence_ranking['influence_score'] >= 0.7]
    medium_influence = influence_ranking[
        (influence_ranking['influence_score'] >= 0.4) &
        (influence_ranking['influence_score'] < 0.7)
    ]
    low_influence = influence_ranking[influence_ranking['influence_score'] < 0.4]

    print(f"\n영향력 그룹 분석:")
    print(f"• 고영향력 부처 ({len(high_influence)}개): {', '.join(high_influence['ministry'].tolist())}")
    print(f"• 중영향력 부처 ({len(medium_influence)}개): {', '.join(medium_influence['ministry'].tolist())}")
    print(f"• 저영향력 부처 ({len(low_influence)}개): {', '.join(low_influence['ministry'].tolist())}")

    return df, influence_ranking

def analyze_centrality_correlations(df):
    """
    중심성 지표 간 상관관계 분석

    Args:
        df (pd.DataFrame): 중심성 데이터

    Returns:
        pd.DataFrame: 상관계수 행렬
    """
    print("\n중심성 지표 간 상관관계 분석 중...")

    # 분석할 중심성 지표들
    centrality_cols = ['degree', 'in_degree', 'out_degree', 'closeness',
                      'betweenness', 'eigenvector', 'weighted_in_degree', 'weighted_out_degree']

    # 상관계수 계산
    correlation_matrix = df[centrality_cols].corr()

    print("=== 중심성 지표 간 Pearson 상관계수 ===")
    print(correlation_matrix.round(3))

    # 강한 상관관계 (|r| > 0.7) 식별
    strong_correlations = []
    for i in range(len(centrality_cols)):
        for j in range(i+1, len(centrality_cols)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_correlations.append({
                    'var1': centrality_cols[i],
                    'var2': centrality_cols[j],
                    'correlation': corr_val
                })

    if strong_correlations:
        print(f"\n강한 상관관계 (|r| > 0.7):")
        for corr in strong_correlations:
            print(f"• {corr['var1']} - {corr['var2']}: {corr['correlation']:.3f}")

    return correlation_matrix

def create_centrality_visualizations(df, correlation_matrix, save_path=None):
    """
    중심성 분석 시각화 생성

    Args:
        df (pd.DataFrame): 중심성 데이터
        correlation_matrix (pd.DataFrame): 상관계수 행렬
        save_path (str): 저장 경로
    """
    print("\n중심성 분석 시각화 생성 중...")

    # 1. 종합 영향력 점수 막대 그래프
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 영향력 점수 상위 10개 부처
    top_10 = df.nlargest(10, 'influence_score')

    ax1 = axes[0, 0]
    bars = ax1.barh(range(len(top_10)), top_10['influence_score'], color='steelblue')
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([name.replace('부', '').replace('청', '') for name in top_10['ministry']])
    ax1.set_xlabel('종합 영향력 점수')
    ax1.set_title('정부 부처 종합 정책 영향력 순위 (상위 10개)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 값 표시
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)

    # 2. 중심성 지표별 산점도 (매개 중심성 vs 고유벡터 중심성)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['betweenness'], df['eigenvector'],
                         s=df['degree']*2000, alpha=0.6, c=df['influence_score'],
                         cmap='viridis')

    # 주요 부처 라벨링
    top_5_influence = df.nlargest(5, 'influence_score')
    for _, row in top_5_influence.iterrows():
        ax2.annotate(row['ministry'].replace('부', '').replace('청', ''),
                    (row['betweenness'], row['eigenvector']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('매개 중심성')
    ax2.set_ylabel('고유벡터 중심성')
    ax2.set_title('중심성 지표 관계 분석\n(점 크기: 연결 중심성, 색상: 영향력 점수)', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='영향력 점수')

    # 3. 상관계수 히트맵
    ax3 = axes[1, 0]
    centrality_cols = ['degree', 'in_degree', 'out_degree', 'closeness',
                      'betweenness', 'eigenvector']
    corr_subset = correlation_matrix.loc[centrality_cols, centrality_cols]

    sns.heatmap(corr_subset, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('중심성 지표 간 상관관계', fontweight='bold')

    # 4. 협업 프로젝트 vs 중심성
    ax4 = axes[1, 1]

    # 가중 연결 중심성을 기반으로 총 협업 강도 계산
    total_collaboration = df['weighted_in_degree'] + df['weighted_out_degree']

    ax4.scatter(total_collaboration, df['influence_score'],
               s=100, alpha=0.7, color='orange')

    # 상위 5개 부처 라벨링
    for _, row in top_5_influence.iterrows():
        ministry_collab = df.loc[df['ministry'] == row['ministry'], 'weighted_in_degree'].iloc[0] + \
                         df.loc[df['ministry'] == row['ministry'], 'weighted_out_degree'].iloc[0]
        ax4.annotate(row['ministry'].replace('부', '').replace('청', ''),
                    (ministry_collab, row['influence_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax4.set_xlabel('총 협업 강도')
    ax4.set_ylabel('종합 영향력 점수')
    ax4.set_title('협업 강도 vs 정책 영향력', fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/centrality_analysis.png",
                   dpi=300, bbox_inches='tight')
        print(f"중심성 분석 시각화 저장: {save_path}/centrality_analysis.png")

    plt.show()

def create_radar_chart(df, save_path=None):
    """
    주요 부처들의 다차원 중심성 레이더 차트

    Args:
        df (pd.DataFrame): 중심성 데이터
        save_path (str): 저장 경로
    """
    print("\n레이더 차트 생성 중...")

    # 상위 5개 영향력 부처 선택
    top_ministries = df.nlargest(5, 'influence_score')

    # 레이더 차트용 지표들
    radar_cols = ['degree', 'closeness', 'betweenness', 'eigenvector', 'weighted_in_degree']
    radar_labels = ['연결성', '근접성', '매개성', '지위성', '협업강도']

    # 각 지표를 0-1로 정규화
    normalized_data = {}
    for col in radar_cols:
        max_val = df[col].max()
        if max_val > 0:
            normalized_data[col] = df[col] / max_val
        else:
            normalized_data[col] = df[col]

    # 레이더 차트 설정
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]  # 원형 완성

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i, (_, ministry) in enumerate(top_ministries.iterrows()):
        values = [normalized_data[col][ministry.name] for col in radar_cols]
        values += values[:1]  # 원형 완성

        ministry_name = ministry['ministry'].replace('부', '').replace('청', '')
        ax.plot(angles, values, 'o-', linewidth=2,
                label=ministry_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    # 레이더 차트 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)])
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('주요 부처별 다차원 중심성 분석\n(정규화된 값)',
              fontweight='bold', pad=30)

    if save_path:
        plt.savefig(f"{save_path}/centrality_radar.png",
                   dpi=300, bbox_inches='tight')
        print(f"레이더 차트 저장: {save_path}/centrality_radar.png")

    plt.show()

def export_centrality_results(df, rankings, correlation_matrix, save_path):
    """
    중심성 분석 결과를 파일로 내보내기

    Args:
        df (pd.DataFrame): 중심성 데이터
        rankings (dict): 순위 분석 결과
        correlation_matrix (pd.DataFrame): 상관계수 행렬
        save_path (str): 저장 경로
    """
    print("\n중심성 분석 결과 내보내기 중...")

    # 1. 종합 중심성 결과 CSV
    df_export = df.round(4)
    centrality_path = f"{save_path}/centrality_analysis.csv"
    df_export.to_csv(centrality_path, index=False, encoding='utf-8')
    print(f"중심성 분석 결과 저장: {centrality_path}")

    # 2. 순위별 결과 Excel
    excel_path = f"{save_path}/centrality_rankings.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 종합 결과
        df_export.to_excel(writer, sheet_name='전체결과', index=False)

        # 각 중심성별 상위 5개
        for centrality_type, ranking_df in rankings.items():
            sheet_name = centrality_type.replace('_', ' ').title()[:31]  # Excel 시트명 길이 제한
            ranking_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 상관계수 행렬
        correlation_matrix.to_excel(writer, sheet_name='상관관계')

    print(f"순위 분석 결과 저장: {excel_path}")

    # 3. 요약 통계 텍스트 파일
    summary_path = f"{save_path}/centrality_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== 정부 부처 중심성 분석 결과 요약 ===\n\n")

        # 종합 영향력 상위 5개
        f.write("종합 정책 영향력 순위 (상위 5개):\n")
        top_5 = df.nlargest(5, 'influence_score')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            f.write(f"{i}. {row['ministry']}: {row['influence_score']:.3f}\n")

        f.write(f"\n주요 발견사항:\n")
        f.write(f"• 평균 영향력 점수: {df['influence_score'].mean():.3f}\n")
        f.write(f"• 표준편차: {df['influence_score'].std():.3f}\n")
        f.write(f"• 최고점 부처: {top_5.iloc[0]['ministry']} ({top_5.iloc[0]['influence_score']:.3f})\n")
        f.write(f"• 최저점: {df['influence_score'].min():.3f}\n")

        # 중심성별 1위 부처
        f.write(f"\n중심성별 1위 부처:\n")
        centrality_winners = {}
        for col in ['degree', 'closeness', 'betweenness', 'eigenvector']:
            winner = df.loc[df[col].idxmax()]
            centrality_winners[col] = winner['ministry']
            f.write(f"• {col}: {winner['ministry']} ({winner[col]:.3f})\n")

    print(f"요약 통계 저장: {summary_path}")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("제6장: 네트워크 중심성 분석과 정책 영향력 측정")
    print("=" * 60)

    # 정부 네트워크 직접 생성 (간소화 버전)
    def create_simple_government_network():
        """간소화된 정부 부처 네트워크 생성"""
        G = nx.DiGraph()
        
        ministries = [
            '기획재정부', '교육부', '과학기술정보통신부', '외교부',
            '통일부', '법무부', '국방부', '행정안전부',
            '문화체육관광부', '농림축산식품부', '산업통상자원부', '보건복지부',
            '환경부', '고용노동부', '여성가족부', '국토교통부',
            '해양수산부', '중소벤처기업부', '국가보훈부'
        ]
        
        for ministry in ministries:
            G.add_node(ministry, node_type='ministry')
        
        # 주요 협업 관계 추가
        collaborations = [
            ('기획재정부', '과학기술정보통신부', {'weight': 15, 'projects': 15}),
            ('과학기술정보통신부', '교육부', {'weight': 12, 'projects': 12}),
            ('과학기술정보통신부', '행정안전부', {'weight': 18, 'projects': 18}),
            ('기획재정부', '산업통상자원부', {'weight': 20, 'projects': 20}),
            ('산업통상자원부', '고용노동부', {'weight': 20, 'projects': 20}),
            ('보건복지부', '고용노동부', {'weight': 13, 'projects': 13}),
            ('환경부', '산업통상자원부', {'weight': 14, 'projects': 14}),
            ('과학기술정보통신부', '산업통상자원부', {'weight': 8, 'projects': 8}),
            ('기획재정부', '고용노동부', {'weight': 7, 'projects': 7}),
            ('교육부', '고용노동부', {'weight': 9, 'projects': 9}),
            ('보건복지부', '여성가족부', {'weight': 11, 'projects': 11}),
            ('고용노동부', '여성가족부', {'weight': 6, 'projects': 6}),
            ('과학기술정보통신부', '국토교통부', {'weight': 9, 'projects': 9}),
            ('행정안전부', '국토교통부', {'weight': 8, 'projects': 8}),
            ('기획재정부', '중소벤처기업부', {'weight': 8, 'projects': 8}),
            ('산업통상자원부', '중소벤처기업부', {'weight': 11, 'projects': 11}),
            ('외교부', '통일부', {'weight': 6, 'projects': 6}),
            ('외교부', '문화체육관광부', {'weight': 5, 'projects': 5}),
            ('환경부', '국토교통부', {'weight': 7, 'projects': 7}),
            ('환경부', '해양수산부', {'weight': 8, 'projects': 8}),
            ('농림축산식품부', '해양수산부', {'weight': 6, 'projects': 6}),
            ('보건복지부', '과학기술정보통신부', {'weight': 10, 'projects': 10}),
            ('교육부', '과학기술정보통신부', {'weight': 12, 'projects': 12}),
            ('행정안전부', '과학기술정보통신부', {'weight': 18, 'projects': 18}),
            ('기획재정부', '행정안전부', {'weight': 12, 'projects': 12}),
            ('외교부', '기획재정부', {'weight': 8, 'projects': 8})
        ]
        
        for source, target, attrs in collaborations:
            G.add_edge(source, target, **attrs)
        
        return G
    
    gov_network = create_simple_government_network()
    print("기존 네트워크 불러오기 성공")

    # 1. 모든 중심성 지표 계산
    centrality_df = calculate_all_centralities(gov_network)

    # 2. 중심성별 순위 분석
    rankings = analyze_centrality_rankings(centrality_df)

    # 3. 종합 영향력 점수 계산
    centrality_df, influence_ranking = calculate_influence_score(centrality_df)

    # 4. 상관관계 분석
    correlation_matrix = analyze_centrality_correlations(centrality_df)

    # 5. 시각화
    create_centrality_visualizations(centrality_df, correlation_matrix,
                                   './outputs')
    create_radar_chart(centrality_df, './outputs')

    # 6. 결과 내보내기 (생략)
    # export_centrality_results(centrality_df, rankings, correlation_matrix,
    #                          './data')

    print("\n중심성 분석 완료! 결과가 outputs/ 디렉토리에 저장되었습니다.")

    return centrality_df, rankings, correlation_matrix

if __name__ == "__main__":
    # 실행
    centrality_results, ranking_results, corr_results = main()

    # 결과를 전역 변수로 저장
    globals()['centrality_df'] = centrality_results
    globals()['rankings'] = ranking_results
    globals()['correlation_matrix'] = corr_results