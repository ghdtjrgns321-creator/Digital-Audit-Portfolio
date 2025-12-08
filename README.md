# Digital-Audit-Portfolio
CPA 디지털 감사 포트폴리오

# 빅펌 디지털 감사(Digital Audit) 포트폴리오

본 저장소는 2026년 CPA 합격 후 빅펌 디지털 감사 직무를 목표로,
파이썬(Pandas)과 SQL을 활용한 감사 포트폴리오를 구축하는 과정을 기록합니다.

### Part 1: 파이썬 및 Numpy 기초
- **목표:** : 데이터 분석을 위한 파이썬 핵심 문법, 자료구조(리스트, 딕셔너리 등), 제어문 및 수치 계산 라이브러리(Numpy)의 기초 숙달
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 기본 문법:
        - 입출력: print(), 주석(#)
        - 자료형: int, float, str, bool, type() 확인
        - 형변환: int(), str()
        - 연산자: 산술(+, -, *, /, //, %, **), 비교(>, <, ==, !=)
    - 문자열 처리:
        - 포맷팅: f-string (추천), .format()
        - 메서드: .split(), .replace(), .upper(), .lower()
    - 자료구조 (Data Structures):
        - 리스트(List): 생성, 추가(append), 정렬(sort), 인덱싱/슬라이싱([0], [::-1]), 내장함수(sum, len)
        - 딕셔너리(Dict): Key-Value 구조, 수정, 조회, 메서드(keys, values, items)
        - 튜플(Tuple) & 세트(Set): 불변성 및 중복 제거 특성
    - 제어문 (Control Flow):
        - 조건문: if, elif, else
        - 반복문: for, while, range, enumerate (인덱스 동시 추출), zip (리스트 묶기), List Comprehension
    - 함수 (Function):
        - 정의: def, 매개변수, return
        - 고급: lambda (익명 함수), map, filter
    - Numpy (수치 계산):
        - 배열(Array): np.array, 차원(ndim), 크기(shape), 랭크(rank)
        - 생성 및 변형: np.arange, .reshape, 전치행렬(.T)
        - 수학 연산: 내적(dot), 행렬 곱(matmul), 최댓값 인덱스(argmax)
        - 올림/내림: np.ceil, np.floor, np.trunc
- **실습 파일:**
    - [`part1_python_basics.md`](./part1_python_basics.md):: 파이썬 및 Numpy 상세 학습 노트
  
### Part 2-1: 판다스 기초
- **목표:** 데이터 분석의 핵심 라이브러리인 판다스의 주요 기능 숙달, 실제 데이터 분석 문제(데이터 클리닝, 전처리, 그룹핑)를 해결하는 실습 진행
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - 데이터 구조 (Series, DataFrame)
    - 데이터 입출력 (CSV Read/Write)
    - 데이터 탐색 (head, info, describe, value_counts)
    - 데이터 전처리 (loc/iloc, fillna, drop, astype, apply)
    - 필터링 (Boolean Indexing, isin, contains)
    - 그룹핑 및 집계 (groupby, agg, pivot_table, unstack)
    - 시계열 데이터 (to_datetime, .dt.dayofweek, Timedelta)
    - 결측치 처리: fillna(median/mode), dropna(subset=[...]), fillna(method='bfill')
    - 데이터 변환: replace(), map(), select_dtypes(), .T (전치)
    - 이상치 탐지(Outlier): IQR 로직 (Q3-Q1), 고급 필터링 (~, != round())
    - 데이터 집계: groupby().agg(), groupby().sum(numeric_only=True), sort_values(), .iloc[]
- **실습 파일:**
    - [`part2-1_pandas_basics.md`](./part2-1_pandas_basics.md): 판다스 상세 학습 노트
    - [`part2-1_pandas_basics_quiz1.ipynb`](./part2-1_pandas_basics_quiz1.ipynb): 판다스 실습과 퀴즈1
    - [`part2-1_pandas_basics_quiz2.ipynb`](./part2-1_pandas_basics_quiz2.ipynb): 판다스 실습과 퀴즈2
      
### Part 2-2: 판다스 심화
- **목표:** : 판다스 스킬 다지기. loc/iloc 등 기본 복습부터 groupby 심화, 시계열(resample), 실전 처리 워크플로우, 다양한 모의문제까지 총정리.
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 데이터 선택: loc/iloc, select_dtype, filter, query (외부 변수 @ 활용)
    - 데이터 변환: set_index, rename, .map(lambda ...) (문자열 정제), astype
    - 데이터 집계 (심화): groupby + agg(다중 함수), apply(사용자 정의 함수), 그룹별 fillna
    - 시계열 데이터: to_datetime, set_index (시계열), resample (7d, B)
    - 데이터 범주화: pd.cut (길이 기준), pd.qcut (개수 기준)
    - 알고리즘: '조건을 만족하는 최대 연속 횟수' (cumsum/mul/diff/where/ffill/add) 로직 복습 및 실전 적용
    - 판다스 퀴즈 복습: 필터링(loc), 결측치(fillna), 시계열(dt.day), 인덱싱(index[0]), 파생 변수, 정렬(sort_values), iloc
    - 핵심 메서드 복습: select_dtype, set_index, filter, rename, value_counts, unique, map(lambda), query(@)
    - GroupBy 심화: agg(다중 함수), apply(사용자 정의 함수), 그룹별 fillna
    - 시계열 심화: resample, pd.Grouper
    - 데이터 처리 실전: clip(이상치), interpolate(결측치), explode(행 쪼개기)
    - '연속 횟수' 로직을 실전 데이터(애플 주가)에 적용
- **실습 파일:**
    - [`part2-2_pandas_advanced.md`](./part2-2_pandas_advanced.md): 판다스 심화 메서드, 실전 프로세스 학습 노트
    - [`part2-2_pandas_advanced_quiz.ipynb`](./part2-2_pandas_advanced_quiz.ipynb): 판다스 심화 메서드, 실전 프로세스 퀴즈
      
### Part 3-1: 데이터 시각화 기초
- **목표:** : Matplotlib, Seaborn, Plotly를 사용한 핵심 그래프(Scatter, Line, Box, Histogram, Heatmap)의 기본 사용법과 주요 파라미터(인자)를 숙달
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 시각화 기본: Figure, Axes 개념
    - Matplotlib & Seaborn (Axes-level):
        - sns.scatterplot (산점도): hue, palette, style, markers, s, size
        - sns.regplot (회귀선): ci, order(다항식), scatter_kws, line_kws
        - sns.lineplot (선 그래프): hue, style, scatterplot과 겹치기
        - sns.boxplot (박스 플롯): order
        - sns.stripplot, sns.swarmplot (분포 확인)
        - sns.histplot (히스토그램): bins, hue, multiple='stack'
        - sns.heatmap (히트맵): annot, fmt, vmax/vmin, cmap
    - Seaborn (Figure-level):
        - sns.lmplot, sns.relplot (Figure-level의 개념)
        - col, row, col_wrap (그룹별 subplot 나누기)
        - sns.FacetGrid + .map_dataframe() (Axes-level을 Figure-level처럼 쓰기)
    - Plotly Express (px):
        - px.scatter: color, symbol, size
        - Regression Plot : px.scatter 함수 내 trendline='ols' 인자를 사용
        - px.line: line_dash
        - px.box: points='all', category_orders
        - px.histogram: nbins, color, barmode
        - px.imshow (히트맵): text_auto, color_continuous_scale
        - facet_col, facet_row (Plotly의 subplot 기능)
    - 그래프 튜닝: tick_params, set_title, suptitle, grid, tight_layout, update_layout
- **실습 파일:**
    - [`part3-1_visualization_basics.md`](./part3-1_visualization_basics.md):: 시각화 기초 학습 노트
    - [`part3-1_visualization_basics_quiz.ipynb`](./part3-1_visualization_basics_quiz.ipynb):: 시각화 기초 퀴즈 + 판다스 복습 퀴즈

### Part 3-2: 데이터 시각화 심화
- **목표:** : Matplotlib, Seaborn, Plotly를 활용하여 고급 그래프(Histogram, Heatmap)를 익히고, 여러 개의 그래프를 동시에 그리는 FacetGrid 및 facet_col의 원리를 이해하며, 그래프를 세부 튜닝하는 방법을 숙달
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 고급 그래프 (Seaborn/Plotly):
        - sns.histplot / px.histogram (히스토그램): bins, hue/color, multiple/barmode
        - sns.heatmap / px.imshow (히트맵): annot/text_auto, fmt, cmap/color_continuous_scale
    - 다중 그래프 (Facet):
        - Seaborn: Axes-level vs Figure-level (lmplot) 비교, FacetGrid + .map_dataframe()
        - Plotly: facet_col, facet_row, facet_col_wrap
    - 그래프 세부 튜닝 (Matplotlib/Plotly):
        - 축(Axis) 튜닝: tick_params (회전), update_xaxes (포맷), set_yscale('log'), grid
        - 다중 축(Y-axis): ax.twinx(), make_subplots(specs=[...])
        - 범례(Legend): ax.legend(bbox_to_anchor=...), update_layout(legend_x=...)
        - 텍스트/주석: ax.text(), ax.annotate(), fig.add_annotation()
        - 선/테두리: ax.axhline(), fig.add_hline(), ax.spines
    - 고급 튜닝 (FacetGrid / Plotly):
        - for 문 / def 함수를 활용한 '그룹별' 튜닝 (e.g., 그룹별 '평균선' 추가, '이상치' 강조)
    - 색상 (Color):
        - palette (Seaborn) / color_continuous_scale (Plotly)
        - '수동' 색상 지정 (List, Dictionary)
        - '커스텀' Color Map 만들기 (LinearSegmentedColormap)
- **실습 파일:**
    - [`part3-2_visualization_advanced.md`](./part3-2_visualization_advanced.md):: 시각화 심화(히스토그램/히트맵/Facet/튜닝) 상세 학습 노트
      
### Part 4: 파이썬을 활용한 통계분석
- **목표:** : 파이썬(Scipy, Statsmodels)을 활용한 가설 검정(Hypothesis Testing)의 기본 개념과 T-test, ANOVA, 카이제곱 검정(Chi-square), 회귀 분석(OLS)을 숙달
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 가설 검정: 귀무가설, 대립가설, p-value
    - T-test (1-sample, Paired, Independent): stats.ttest_1samp, ttest_rel, ttest_ind
    - 비모수 검정 (T-test): stats.wilcoxon, stats.mannwhitneyu
    - 전제 조건 검정: stats.shapiro (정규성), stats.levene (등분산성)
    - 범주형 분석 (Chi-square): stats.chisquare (적합도), stats.chi2_contingency (독립성/동질성), pd.crosstab
    - 상관/회귀 분석: .corr() (Pearson, Spearman), statsmodels.formula.api.ols, model.summary(), model.predict(), model.resid
    - 분산 분석 (ANOVA): stats.f_oneway (일원), ols('y ~ C(f1) * C(f2)') (이원)
    - 사후 검정: pairwise_tukeyhsd (Tukey), MultiComparison (Bonferroni)
    - 비모수 검정 (ANOVA): stats.kruskal (Kruskal-Wallis)
- **실습 파일:**
    - [`part4_statistics.md`](./part4_statistics.md):: 통계 분석 상세 학습 노트
    - [`part4_statistics_quiz.md`](./part4_statistics_quiz.md):: 판다스/시각화 복습퀴즈

### Part 5-1: 머신러닝 
- **목표:** : 머신러닝의 전체 프로세스(EDA, 전처리, 모델링, 평가)를 이해하고, 고급 전처리 기법(스케일링, 인코딩, 차원 축소, 피처 엔지니어링)과 다양한 모델(RF, XGB, LGBM, SVM 등)을 실전에 적용하며, 모델 성능을 극대화하는 검증/튜닝 (K-Fold, GridSearchCV, Early Stopping) 방법을 숙달
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 머신러닝 프로세스: EDA ➡️ 데이터 전처리 ➡️ 검증 데이터 분할 ➡️ 모델 학습/평가 ➡️ 예측/제출
    - EDA (탐색적 데이터 분석): .info(), .describe(), .isnull().sum(), sns.histplot (분포 확인)
    - 데이터 전처리 (ML):
        - 결측치 처리: dropna(), fillna() (mode, mean, 'X')
        - 타겟 분리: .pop()
        - 인코딩 (Encoding): pd.get_dummies (One-Hot), LabelEncoder, (Train/Test '합치기/쪼개기')
        - 스케일링 (Scaling): MinMaxScaler, StandardScaler (SVC 필수), RobustScaler
    - 피처 엔지니어링 (Feature Engineering):
        - PCA (차원 축소)
        - SelectKBest (특성 선택)
        - PolynomialFeatures (다항 회귀)
        - np.log1p (로그 변환) (치우친 데이터)
    - 검증 데이터 분할: train_test_split (stratify=target 중요!), KFold, StratifiedKFold, cross_validate
    - 모델 학습 (분류): DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier, KNeighborsClassifier, SVC, LogisticRegression, GradientBoostingClassifier
    - 모델 학습 (회귀): LinearRegression (coef_, intercept_), RandomForestRegressor, LGBMRegressor, XGBRegressor
    - 모델 학습 (군집): KMeans
    - 모델 튜닝 (Optimization):
        - GridSearchCV (하이퍼파라미터 튜닝)
        - Early Stopping (XGB/LGBM 조기 종료)
        - VotingClassifier (앙상블)
    - 모델 평가 (분류):
        - .predict() (최종 결과) / predict_proba() (확률 ➡️ ROC-AUC용)
        - confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        - 함수화, 평가표 자동화
    - 모델 평가 (회귀): RMSE, MSE, MAE, r2_score
    - 모델 평가 (군집): silhouette_score (최적의 K 찾기)
    - 모델 시각화: export_graphviz (트리), plot_importance (XGB/LGBM), sns.barplot (LR 계수)
- **실습 파일:**
    - [`part5-1_ml.md`](./part5-1_ml.md):: 머신러닝 기초/심화 프로세스 상세 학습 노트
 
### Part 5-2: 머신러닝 실습
- **목표:** : Part 5-1(ML 이론)에서 배운 머신러닝 프로세스(EDA, 전처리, 모델링, 평가)를 분류(Classification), 회귀(Regression), 군집(Clustering) 3가지 핵심 실전 프로젝트에 적용
- **학습 환경:**: Google Colab
- **주요 학습 내용:**
    - 분류(Classification) 실습 (Project 1):
        - EDA, .pop(), .drop(), LabelEncoder/pd.get_dummies(One-Hot) 비교
        - train_test_split (검증 데이터 분리)
        - 모델링: RandomForestClassifier
        - 평가: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    - 회귀(Regression) 실습 (Project 2):
        - EDA, .drop(), fillna(0), LabelEncoder
        - 모델링 (5종 비교): LinearRegression, Ridge, Lasso, RandomForestRegressor, XGBRegressor
        - 평가: R2, MAE, MSE, RMSE, RMSLE, MAPE (사용자 정의 함수)
    - 군집(Clustering) 실습 (Project 3):
        - KMeans (비지도 학습)
        - 평가: silhouette_score (최적의 K 찾기)
- **실습 파일:**
    - [`part5_2_ml_practice1_분류.ipynb`](./part5_2_ml_practice1_분류.ipynb):: 머신러닝 실전 프로젝트 1 (분류)
    - [`part5_2_ml_practice2_회귀.ipynb`](./part5_2_ml_practice2_회귀.ipynb):: 머신러닝 실전 프로젝트 2 (회귀)
    - [`part5_2_ml_practice3_군집.ipynb`](./part5_2_ml_practice3_군집.ipynb):: 머신러닝 실전 프로젝트 3 (군집)

### Part 6: 웹 대시보드 개발 (Streamlit)
- **목표:** : Python만으로 데이터 분석 결과를 인터랙티브 웹 애플리케이션으로 구현하는 방법을 숙달, 실전 프로젝트 수행
- **학습 환경:**: VS Code (로컬 개발), Jupyter Notebook (nbconvert), Terminal
- **주요 학습 내용:**
    - 기본 위젯: st.title/header, st.text/markdown (색상/강조), st.code, st.divider
    - 입력 위젯: st.button (primary), st.checkbox, st.toggle, st.radio, st.selectbox, st.multiselect, st.slider (시간/범위), st.text_input (password), st.file_uploader (CSV 로드)
    - 시각화 연동: st.pyplot (Matplotlib/Seaborn), st.plotly_chart (Plotly)
    - 레이아웃(Layout): st.sidebar (사이드바), st.columns (단 나누기), st.tabs (탭), st.expander (접기/펼치기)
    - 고급 기능:
        - Session State: st.session_state (사용자 상호작용 시 변수 값 유지)
        - Caching: @st.cache_data (대용량 연산 속도 최적화)
    - 실전 프로젝트: '자동차 CO2 배출량 분석 대시보드' 제작 (Pandas 필터링 + Plotly 시각화 + Streamlit 인터랙티브 위젯 결합)
- **실습 파일:**
    - [`part6_streamlit_study.md`](./part6_streamlit_study.md):: streamlit 상세 학습 노트
    - [`part6_streamlit_dashboard.py`](./part6_streamlit_dashboard.py):: 실전 프로젝트 대시보드 전체 소스 코드
    - https://digital-audit-portfolio-zkwnvfuoyhgcbwqhxyypbh.streamlit.app/ 실습 페이지

