#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
탄소중립 2050 정책 확산 분석 PDF 보고서 생성
"""

from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import pandas as pd
from datetime import datetime

def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # Windows 한글 폰트 등록
        pdfmetrics.registerFont(TTFont('NanumGothic', 'C:/Windows/Fonts/NanumGothic.ttf'))
        return 'NanumGothic'
    except:
        try:
            pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
            return 'Malgun'
        except:
            return 'Helvetica'  # 기본 폰트로 fallback

def create_carbon_neutral_report():
    """탄소중립 정책 확산 분석 PDF 보고서 생성"""
    
    # 출력 파일 경로
    output_path = "../outputs/carbon_neutral_analysis_report.pdf"
    
    # PDF 문서 생성
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    # 한글 폰트 설정
    font_name = setup_korean_font()
    
    # 스타일 설정
    styles = getSampleStyleSheet()
    
    # 제목 스타일
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1,  # 중앙 정렬
        fontName=font_name if font_name != 'Helvetica' else 'Helvetica-Bold'
    )
    
    # 소제목 스타일
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        fontName=font_name if font_name != 'Helvetica' else 'Helvetica-Bold'
    )
    
    # 본문 스타일
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        fontName=font_name
    )
    
    # 1. 제목
    story.append(Paragraph("탄소중립 2050 정책 확산 분석 보고서", title_style))
    story.append(Spacer(1, 20))
    
    # 2. 개요
    story.append(Paragraph("1. 분석 개요", heading_style))
    overview_text = """
    본 보고서는 대한민국 탄소중립 2050 정책의 정부 부처 간 확산 과정을 네트워크 분석을 통해 
    분석한 결과를 제시합니다. 2020년 10월부터 2021년 8월까지 304일간의 정책 확산 과정을 
    19개 정부 부처를 대상으로 분석하였습니다.
    """
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 15))
    
    # 3. 주요 발견사항
    story.append(Paragraph("2. 주요 발견사항", heading_style))
    
    # 3.1 확산 패턴
    story.append(Paragraph("2.1 정책 확산 패턴", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
    diffusion_text = """
    • 총 확산 기간: 304일 (약 10.1개월)<br/>
    • 참여 부처: 19개 (100% 완전 확산)<br/>
    • 네트워크 밀도: 0.079 (중간 수준의 연결성)<br/>
    • 평균 연결도: 2.84개 부처
    """
    story.append(Paragraph(diffusion_text, body_style))
    story.append(Spacer(1, 10))
    
    # 3.2 핵심 주도 부처
    story.append(Paragraph("2.2 핵심 주도 부처", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
    key_ministries_text = """
    <b>환경부</b>가 정책 확산의 중심 역할을 수행했으며, 다음과 같은 지표에서 최고 점수를 기록했습니다:<br/>
    • Out-Centrality: 0.333 (정책 전파력 1위)<br/>
    • In-Centrality: 0.333 (정책 수용력 1위)<br/>
    • Betweenness: 0.196 (매개 중심성 1위)
    """
    story.append(Paragraph(key_ministries_text, body_style))
    story.append(Spacer(1, 10))
    
    # 3.3 확산 단계별 분석
    story.append(Paragraph("2.3 Rogers 혁신 확산 이론 적용", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
    rogers_text = """
    Rogers의 혁신 확산 이론에 따른 부처 분류:<br/>
    • <b>조기 채택자 (3개)</b>: 환경부, 기획재정부, 과학기술정보통신부<br/>
    • <b>조기 다수 (6개)</b>: 산업통상자원부, 국토교통부, 외교부 등<br/>
    • <b>후기 다수 (6개)</b>: 교육부, 고용노동부, 행정안전부 등<br/>
    • <b>지각 채택자 (4개)</b>: 여성가족부, 법무부, 국방부, 국가보훈부
    """
    story.append(Paragraph(rogers_text, body_style))
    story.append(Spacer(1, 15))
    
    # 4. 네트워크 분석 결과
    story.append(Paragraph("3. 네트워크 분석 결과", heading_style))
    
    # 4.1 중심성 분석
    centrality_data = [
        ['부처명', 'Out-Centrality', 'In-Centrality', 'Betweenness'],
        ['환경부', '0.333', '0.333', '0.196'],
        ['산업통상자원부', '0.167', '0.278', '0.071'],
        ['과학기술정보통신부', '0.111', '0.222', '0.051'],
        ['국토교통부', '0.111', '0.111', '0.033'],
        ['기획재정부', '0.167', '0.056', '0.000']
    ]
    
    centrality_table = Table(centrality_data)
    centrality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), font_name if font_name != 'Helvetica' else 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), font_name),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(Paragraph("3.1 상위 5개 부처 중심성 지표", ParagraphStyle('SubHeading', parent=heading_style, fontSize=12)))
    story.append(centrality_table)
    story.append(Spacer(1, 15))
    
    # 5. 정책적 시사점
    story.append(Paragraph("4. 정책적 시사점", heading_style))
    implications_text = """
    <b>4.1 환경부의 중심적 역할</b><br/>
    환경부가 탄소중립 정책 확산의 허브 역할을 성공적으로 수행했으며, 
    이는 정책의 전문성과 리더십이 확산 과정에서 중요함을 시사합니다.<br/><br/>
    
    <b>4.2 경제 부처들의 빠른 수용</b><br/>
    기획재정부, 산업통상자원부 등 경제 관련 부처들이 조기에 정책을 수용한 것은 
    탄소중립 정책의 경제적 중요성이 인식되었음을 보여줍니다.<br/><br/>
    
    <b>4.3 점진적 확산 패턴</b><br/>
    304일에 걸친 점진적 확산은 대규모 정책 변화에 필요한 충분한 준비 기간과 
    단계적 접근의 중요성을 보여줍니다.<br/><br/>
    
    <b>4.4 협력 네트워크의 중요성</b><br/>
    평균 2.84개의 연결도는 부처 간 협력이 정책 성공에 핵심적임을 시사하며, 
    향후 유사한 정책에서도 부처 간 협력 체계 구축이 중요할 것입니다.
    """
    story.append(Paragraph(implications_text, body_style))
    story.append(Spacer(1, 15))
    
    # 6. 결론
    story.append(Paragraph("5. 결론", heading_style))
    conclusion_text = """
    탄소중립 2050 정책의 확산 과정 분석을 통해 다음과 같은 결론을 도출할 수 있습니다:
    
    1. <b>전문 부처의 리더십이 핵심</b>: 환경부의 중심적 역할이 정책 확산의 성공 요인
    2. <b>경제적 중요성의 인식</b>: 경제 관련 부처들의 빠른 수용으로 정책 동력 확보
    3. <b>점진적 접근의 효과</b>: 충분한 시간을 통한 단계적 확산으로 완전한 정책 수용 달성
    4. <b>협력 네트워크의 가치</b>: 부처 간 협력이 정책 성공의 핵심 동력
    
    이러한 분석 결과는 향후 대규모 정책 도입 시 참고할 수 있는 중요한 인사이트를 제공합니다.
    """
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 20))
    
    # 7. 보고서 정보
    story.append(Paragraph("보고서 정보", heading_style))
    report_info = f"""
    • 분석 기간: 2020년 10월 1일 ~ 2021년 8월 1일<br/>
    • 분석 대상: 19개 정부 부처<br/>
    • 분석 방법: 네트워크 분석, Rogers 혁신 확산 이론<br/>
    • 보고서 생성일: {datetime.now().strftime('%Y년 %m월 %d일')}<br/>
    • 도구: Python NetworkX, 정책 확산 시뮬레이션
    """
    story.append(Paragraph(report_info, body_style))
    
    # PDF 생성
    doc.build(story)
    print(f"✅ PDF 보고서가 생성되었습니다: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_carbon_neutral_report()