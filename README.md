# Digital-Audit-Portfolio
CPA 디지털 감사 포트폴리오

# 빅펌 디지털 감사(Digital Audit) 포트폴리오

본 저장소는 2026년 CPA 합격 후 빅펌 디지털 감사 직무를 목표로,
파이썬(Pandas)과 SQL을 활용한 감사 포트폴리오를 구축하는 과정을 기록합니다.

### Part 1: 파이썬 기초

- **목표:** 파이썬 기본 문법 및 자료구조(리스트, 딕셔너리) 학습습
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - 변수, 자료형 (int, float, str, bool)
    - 연산자 (산술, 비교)
    - 조건문 (if, elif, else)
    - 자료구조 (리스트, 딕셔너리)
    - 반복문 (for, enumerate, .items())
    - 함수 (def, return)
    - f-string (문자열 포맷팅)    
- **실습 파일:**
    - [`part1_python_basics.ipynb`](./part1_python_basics.ipynb): 기본 문법 및 자료구조 상세 학습 노트 및 기초 문법 실습 퀴즈
  
### Part 2-1: 판다스 기초

- **목표:** 데이터 분석의 핵심 라이브러리인 판다스의 주요 기능 숙달
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - 데이터 구조 (Series, DataFrame)
    - 데이터 입출력 (CSV Read/Write)
    - 데이터 탐색 (head, info, describe, value_counts)
    - 데이터 전처리 (loc/iloc, fillna, drop, astype, apply)
    - 필터링 (Boolean Indexing, isin, contains)
    - 그룹핑 및 집계 (groupby, agg, pivot_table, unstack)
    - 시계열 데이터 (to_datetime, .dt.dayofweek, Timedelta)
- **실습 파일:**
    - [`part2-1_pandas_basics.ipynb`](./part2-1_pandas_basics.ipynb): 판다스 상세 학습 노트 및 실습과 퀴즈

### Part 2-2: 판다스 기초

- **목표:** : 파이썬 및 판다스 기본 문법을 활용하여, 실제 데이터 분석 문제(데이터 클리닝, 전처리, 그룹핑)를 해결하는 실습을 진행
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - 결측치 처리: fillna(median/mode), dropna(subset=[...]), fillna(method='bfill')
    - 데이터 변환: replace(), map(), select_dtypes(), .T (전치)
    - 이상치 탐지(Outlier): IQR 로직 (Q3-Q1), 고급 필터링 (~, != round())
    - 데이터 집계: groupby().agg(), groupby().sum(numeric_only=True), sort_values(), .iloc[]
    - 시계열: to_datetime(), .dt.month
- **실습 파일:**
    - [`part2-2_pandas_basics.ipynb`](./part2-2_pandas_basics.ipynb): 판다스 실전 모의문제 풀이

### Part 2-3: 판다스 심화

- **목표:** : 판다스 스킬 다지기. loc/iloc 등 기본 복습부터 groupby 심화, 시계열(resample), 실전 처리 워크플로우까지 총정리.
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 데이터 선택: loc/iloc, select_dtype, filter, query (외부 변수 @ 활용)
    - 데이터 변환: set_index, rename, .map(lambda ...) (문자열 정제), astype
    - 데이터 집계 (심화): groupby + agg(다중 함수), apply(사용자 정의 함수), 그룹별 fillna
    - 시계열 데이터: to_datetime, set_index (시계열), resample (7d, B)
    - 데이터 범주화: pd.cut (길이 기준), pd.qcut (개수 기준)
    - 알고리즘: '조건을 만족하는 최대 연속 횟수' (cumsum/mul/diff/where/ffill/add) 로직 복습 및 실전 적용
- **실습 파일:**
    - [`part2-3_pandas_advanced.ipynb`](./part2-3_pandas_advanced.ipynb): 판다스 심화 메서드, 실전 프로세스 및 복습문제 풀이 노트
 
### Part 2-4: 판다스 심화

- **목표:** : 판다스 심화 기능을 다양한 모의문제와 실전 프로세스 복습을 통해 체화
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 판다스 퀴즈 복습: 필터링(loc), 결측치(fillna), 시계열(dt.day), 인덱싱(index[0]), 파생 변수, 정렬(sort_values), iloc
    - 핵심 메서드 복습: select_dtype, set_index, filter, rename, value_counts, unique, map(lambda), query(@)
    - GroupBy 심화: agg(다중 함수), apply(사용자 정의 함수), 그룹별 fillna
    - 시계열 심화: resample, pd.Grouper
    - 데이터 처리 실전: clip(이상치), interpolate(결측치), explode(행 쪼개기)
    - '연속 횟수' 로직을 실전 데이터(애플 주가)에 적용
- **실습 파일:**
    - [`part2-4_pandas_advanced.ipynb`](./part2-4_pandas_advanced.ipynb):: 판다스 심화 메서드, 실전 프로세스 및 복습문제 풀이 노트

### Part 3-1: 데이터 시각화 기초
- **목표:** : Matplotlib, Seaborn, Plotly를 사용한 핵심 그래프(Scatter, Line, Box)의 기본 사용법과 주요 파라미터(인자)를 숙달
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 시각화 기본: Figure, Axes 개념
    - Matplotlib & Seaborn
        sns.scatterplot (산점도): hue, palette, style, markers, s, size
        sns.regplot (회귀선): ci, order(다항식), scatter_kws, line_kws
        sns.lineplot (선 그래프): hue, style, scatterplot과 겹치기
        sns.boxplot (박스 플롯): order
        sns.stripplot, sns.swarmplot (분포 확인)
    - Plotly Express (px):
        px.scatter: color, symbol, size
        Regression Plot : px.scatter 함수 내 trendline='ols' 인자를 사용
        px.line: line_dash
        px.box: points='all', category_orders
- **실습 파일:**
    - [`part3-1_visualization_basics.ipynb`](./part3-1_visualization_basics.ipynb):: Matplotlib, Seaborn, Plotly 기초 학습 노트
