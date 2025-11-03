# Digital-Audit-Portfolio
CPA 디지털 감사 포트폴리오

# 빅펌 디지털 감사(Digital Audit) 포트폴리오

본 저장소는 2026년 CPA 합격 후 빅펌 디지털 감사 직무를 목표로,
파이썬(Pandas)과 SQL을 활용한 감사 포트폴리오를 구축하는 과정을 기록합니다.

### Part 1: 파이썬 기초 문법

- **목표:** 파이썬 기본 문법 및 자료구조(리스트, 딕셔너리) 복습
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
    - [`part1_python_basics.ipynb`](./part1_python_basics.ipynb): 기본 문법 및 자료구조 상세 학습 노트 및 기초 문법 실습 퀴즈 (리스트 필터링, 함수 만들기, 딕셔너리 반복)
  
### Part 2-1: 판다스 핵심

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
    - [`part2-1_pandas_practice.ipynb`](./part2-1_pandas_practice.ipynb): 판다스 상세 학습 노트 및 실습과 퀴즈

### Part 2-2: 판다스 실전 모의문제

- **목표:** : Part 1, 2에서 배운 파이썬 및 판다스 기본 문법을 활용하여, 실제 데이터 분석 문제(데이터 클리닝, 전처리, 그룹핑)를 해결하는 실습을 진행
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - 결측치 처리: fillna(median/mode), dropna(subset=[...]), fillna(method='bfill')
    - 데이터 변환: replace(), map(), select_dtypes(), .T (전치)
    - 이상치 탐지(Outlier): IQR 로직 (Q3-Q1), 고급 필터링 (~, != round())
    - 데이터 집계: groupby().agg(), groupby().sum(numeric_only=True), sort_values(), .iloc[]
    - 시계열: to_datetime(), .dt.month
- **실습 파일:**
    - [`part2-2_pandas_practice.ipynb`](./part2-2_pandas_practice.ipynb): 판다스 실전 모의문제 풀이

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
    - [`part2-3_pandas_deep_dive.ipynb`](./part2-3_pandas_deep_dive.ipynb): 판다스 심화 메서드, 실전 프로세스 및 복습문제 3종 풀이 노트
