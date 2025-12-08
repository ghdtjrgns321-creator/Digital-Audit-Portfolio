# Digital-Audit-Portfolio
CPA 디지털 감사 포트폴리오

# 빅펌 디지털 감사(Digital Audit) 포트폴리오

본 저장소는 2026년 CPA 합격 후 빅펌 디지털 감사 직무를 목표로,
파이썬(Pandas)과 SQL을 활용한 감사 포트폴리오를 구축하는 과정을 기록합니다.

---

### Part 1: 파이썬 및 Numpy 기초
- **목표:** 데이터 분석을 위한 파이썬 핵심 문법, 자료구조(리스트, 딕셔너리 등), 제어문 및 수치 계산 라이브러리(Numpy)의 기초 숙달
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - **기본 문법:**
        - 입출력: `print()`, 주석(`#`)
        - 자료형: `int`, `float`, `str`, `bool`, `type()` 확인
        - 형변환: `int()`, `str()`
        - 연산자: 산술(`+`, `-`, `*`, `/`, `//`, `%`, `**`), 비교(`>`, `<`, `==`, `!=`)
    - **문자열 처리:**
        - 포맷팅: f-string (추천), `.format()`
        - 메서드: `.split()`, `.replace()`, `.upper()`, `.lower()`
    - **자료구조 (Data Structures):**
        - 리스트(List): 생성, 추가(`append`), 정렬(`sort`), 인덱싱/슬라이싱(`[0]`, `[::-1]`), 내장함수(`sum`, `len`)
        - 딕셔너리(Dict): Key-Value 구조, 수정, 조회, 메서드(`keys`, `values`, `items`)
        - 튜플(Tuple) & 세트(Set): 불변성 및 중복 제거 특성
    - **제어문 (Control Flow):**
        - 조건문: `if`, `elif`, `else`
        - 반복문: `for`, `while`, `range`, `enumerate` (인덱스 동시 추출), `zip` (리스트 묶기), List Comprehension
    - **함수 (Function):**
        - 정의: `def`, 매개변수, `return`
        - 고급: `lambda` (익명 함수), `map`, `filter`
    - **Numpy (수치 계산):**
        - 배열(Array): `np.array`, 차원(`ndim`), 크기(`shape`), 랭크(`rank`)
        - 생성 및 변형: `np.arange`, `.reshape`, 전치행렬(`.T`)
        - 수학 연산: 내적(`dot`), 행렬 곱(`matmul`), 최댓값 인덱스(`argmax`)
        - 올림/내림: `np.ceil`, `np.floor`, `np.trunc`
- **실습 파일:**
    - [`part1_python_basics.md`](./part1_python_basics.md): 파이썬 및 Numpy 상세 학습 노트

<br>

### Part 2: 데이터 전처리 및 분석 심화 - Python Pandas
- **목표:** 데이터 분석의 필수 도구인 Pandas를 활용하여 데이터 불러오기부터 전처리, 파생변수 생성, 그룹핑, 시계열 처리, 그리고 데이터 재구조화까지 자유자재로 다루는 능력을 함양
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - **데이터 구조 및 조회:**
        - 기본 구조: 'Series'(1차원), 'DataFrame'(2차원) 생성 및 'dtype' 확인
        - 데이터 I/O: `read_csv`, `to_csv` (index=False)
        - 탐색: `head`, `tail`, `shape`, `info`, `describe`(기초통계), `value_counts`, `unique`, `corr`
    - **인덱싱 및 필터링 (Indexing & Filtering):**
        - `loc` (라벨 기준), `iloc` (인덱스 번호 기준)
        - 조건 필터링: Boolean Indexing, `query` 메서드 (외부 변수 참조 `@`), `isin`
        - 데이터 선택: `select_dtypes` (include/exclude), `filter` (like, regex)
    - **데이터 전처리 (Preprocessing):**
        - 결측치 처리: `isna().sum()`, `dropna` (subset), `fillna` (mean, mode, bfill, ffill), `interpolate`
        - 이상치 처리: `quantile` (IQR 방식), `clip` (상/하한 제한)
        - 중복 처리: `drop_duplicates`
        - 형 변환: `astype`, `pd.to_numeric` (errors='coerce')
        - 컬럼 관리: `rename`, `drop` (axis=0 행, axis=1 열), `reset_index`
    - **고급 연산 및 파생변수 (Advanced Operations):**
        - 함수 적용: `apply` (lambda), `map` (딕셔너리 매핑)
        - 구간 나누기: `pd.cut` (절대 구간), `pd.qcut` (비율 구간)
        - 윈도우/시차: `shift`, `diff`, `cumsum` (누적합), `rolling`
    - **그룹핑 및 집계 (Grouping & Aggregation):**
        - `groupby`: `mean`, `sum`, `count`, `size`, `agg` (여러 함수 적용)
        - `transform`: 그룹별 통계값을 행마다 적용 (결측치 대체, 표준화 등에 활용)
        - `filter`: 그룹별 조건 필터링
    - **데이터 재구조화 및 병합 (Reshaping & Merging):**
        - 피벗: `pivot_table` (index, columns, values, aggfunc)
        - 변형: `melt` (Wide to Long), `unstack` / `stack`, `explode` (리스트 분리)
        - 병합: `pd.concat` (축 기준 연결), `pd.merge` (SQL Join 방식)
    - **문자열 및 시계열 처리 (String & Time Series):**
        - 문자열(`str` accessor): `split`, `contains`, `replace`, `len`, `slice`
        - 시계열(`dt` accessor): `pd.to_datetime`, `year/month/day`, `dayofweek`, `to_period`
        - 시간 연산: `dt.total_seconds()` (시간 차 계산), `resample` (주/월 단위 집계)
- **실습 파일:**
    - [`part2-1_pandas_basics.md`](./part2-1_pandas_basics.md): 판다스 상세 학습 노트
    - [`part2-2_pandas_advanced.md`](./part2-2_pandas_advanced.md): 판다스 심화 메서드, 실전 프로세스 학습 노트

<br>

### Part 3: 파이썬을 활용한 통계분석
- **목표:** 파이썬(Scipy, Statsmodels)을 활용하여 가설 검정의 기초부터 T-test, 분산분석(ANOVA), 카이제곱 검정, 상관/회귀분석까지 통계적 추론 능력을 함양
- **학습 환경:** Google Colab
- **주요 학습 내용:**
    - **가설 검정 기초 (Hypothesis Testing):**
        - 개념: 모집단 vs 표본, 귀무가설($H_0$) vs 대립가설($H_1$)
        - 판단 기준: p-value < 0.05 (유의수준) $\rightarrow$ 귀무가설 기각 (대립가설 채택)
    - **T-test (평균 검정):**
        - **단일표본 (1-Sample):** `stats.ttest_1samp` (정규성 X: `wilcoxon`)
        - **대응표본 (Paired):** `stats.ttest_rel` (전/후 비교, 정규성 X: `wilcoxon`)
        - **독립표본 (Independent):** `stats.ttest_ind` (정규성 X: `mannwhitneyu`, 등분산 X: `equal_var=False`)
        - **전제 조건 검정:** 정규성(`shapiro`), 등분산성(`levene`)
    - **범주형 데이터 분석 (Categorical Analysis):**
        - **적합도 검정:** `chisquare` (관찰 빈도 vs 기대 빈도)
        - **독립성/동질성 검정:** `chi2_contingency` (교차표 `pd.crosstab` 활용)
    - **분산 분석 (ANOVA):**
        - **일원 분산 분석 (One-way):** `stats.f_oneway` 또는 `ols` 모델 활용
        - **이원 분산 분석 (Two-way):** `ols('y ~ C(A) * C(B)')` (상호작용 효과 확인)
        - **비모수 검정:** `stats.kruskal` (Kruskal-Wallis)
        - **사후 검정:** `pairwise_tukeyhsd` (Tukey), `MultiComparison` (Bonferroni)
    - **상관 및 회귀 분석 (Correlation & Regression):**
        - **상관분석:** `corr` (Pearson, Spearman, Kendall), `pearsonr` (유의성 검정)
        - **단순/다중 선형 회귀:** `ols` (Statsmodels), `summary()` 해석 (R-squared, Coef, P-value)
        - **로지스틱 회귀:** `logit`, 오즈비(Odds Ratio) 계산
        - **모델 평가:** 잔차($y - \hat{y}$), MSE, 신뢰구간/예측구간 (`get_prediction`)
        - **범주형 변수 처리:** `pd.get_dummies` (One-Hot Encoding, `drop_first=True`)
- **실습 문제 풀이 (Key Scenarios):**
    1. **커피 중량 (단일표본):** 120g 주장 검증 (단측검정)
    2. **교육 프로그램 효과 (대응표본):** 전/후 점수 비교
    3. **충전기 속도 비교 (독립표본):** New vs Old 그룹 간 차이
    4. **수학 성적 비교 (일원 ANOVA):** 4개 교육 방법 간 차이 및 사후 검정
    5. **토마토 수확량 (이원 ANOVA):** 비료와 물 주기의 상호작용 효과
    6. **교통사고 경험 (적합도):** 도시 vs 전국 분포 비교
    7. **캠프/세미나 등록 (독립성):** 캠프 종류와 세미나 등록의 연관성
    8. **주문량 예측 (다중회귀):** 할인율, 온도, 광고비에 따른 주문량 예측 및 잔차 분석
- **실습 파일:**
    - [`part3_statistics.md`](./part3_statistics.md): 통계 분석 핵심 이론 및 코드 요약 노트, 문제풀이

<br>

### Part 4: 데이터 시각화
- **목표:** Matplotlib, Seaborn, Plotly를 학습하고 단순 그래프 생성을 넘어 시각적 전달력을 극대화하는 세부 튜닝과 다차원 데이터 표현 기법을 숙달
- **학습 환경:** Google Colab, Jupyter Notebook
- **주요 학습 내용:**

#### 1. 시각화 기초 및 핵심 그래프
- **기본 개념:**
    - `Figure`: 전체 그래프가 그려지는 캔버스 (공간)
    - `Axes`: 실제 그래프가 그려지는 각 구역 (Subplot)
- **Scatterplot (산점도):**
    - **Matplotlib & Seaborn:** `sns.scatterplot(data=df, ax=ax, ...)`
        - `hue`: 색상 구분, `style`: 마커 모양 구분, `size`: 크기 구분
        - `palette`: 색상 테마, `markers`: 마커 지정, `hue_order`: 범례 순서
    - **Plotly:** `px.scatter(data_frame=df, ...)`
        - `color`: 색상, `symbol`: 모양, `size`: 크기
        - `color_discrete_sequence`/`map`: 색상 직접 지정
- **Regression Plot (회귀선):**
    - **Seaborn:** `sns.regplot` ( `ci`: 신뢰구간, `order`: 다항식 차수, `scatter_kws`/`line_kws`: 스타일)
    - **Plotly:** `px.scatter` 내 `trendline='ols'` ( `trendline_scope='overall'`로 전체 회귀 가능)
- **Line Plot (선 그래프):**
    - **Seaborn:** `sns.lineplot` (Scatter와 겹쳐 그리기 가능)
    - **Plotly:** `px.line` ( `line_dash`: 점선 스타일 )
- **Box Plot & Distribution:**
    - **Seaborn:** `sns.boxplot` + `sns.stripplot`/`sns.swarmplot` (데이터 분포 오버레이)
    - **Plotly:** `px.box` ( `points='all'`로 분포 표시, `category_orders`로 순서 지정)
- **Histogram (히스토그램):**
    - **Seaborn:** `sns.histplot` ( `bins`, `kde`, `multiple='stack'`)
    - **Plotly:** `px.histogram` ( `nbins`, `barmode='overlay'/'relative'`)
- **Heatmap (히트맵):**
    - **Seaborn:** `sns.heatmap` (Input: Pivot Table, `annot`, `fmt`, `cmap`, `vmax`/`vmin`)
    - **Plotly:** `px.imshow` ( `text_auto`, `color_continuous_scale`)

#### 2. 다차원 시각화 (Facet & Subplots)
- **Seaborn (Axes vs Figure Level):**
    - **Axes-level:** `scatterplot`, `boxplot` 등 (ax 인자 사용 가능, 유연한 배치)
    - **Figure-level:** `lmplot`, `relplot`, `catplot` (col, row 인자로 자동 분할)
    - **FacetGrid:** Axes-level 플롯을 Figure-level 처럼 사용
        - `g = sns.FacetGrid(...)` + `g.map_dataframe(...)`
- **Plotly Facet:**
    - `facet_col`, `facet_row`, `facet_col_wrap` 등을 사용하여 간편하게 서브플롯 생성
- **다중 축 (Multi-Axis):**
    - **Matplotlib:** `ax.twinx()` (2개 축), `ax.spines` 이동 (3개 이상 축)
    - **Plotly:** `make_subplots(specs=[[{"secondary_y": True}]])` 또는 `go.Scatter(yaxis='y2')`

#### 3. 그래프 세부 튜닝 (Graph Tuning)
- **텍스트 및 제목:**
    - Title: `set_title` / `suptitle` (Mat/Sea) vs `title` / `update_layout` (Plotly)
    - Text: `ax.text` / `ax.annotate` (화살표 포함) vs `fig.add_annotation`
- **축(Axis) 및 눈금(Tick):**
    - 회전: `tick_params(labelrotation=90)` vs `update_xaxes(tickangle=90)`
    - 시계열 포맷: `ConciseDateFormatter` vs `tickformat='%Y-%m-%d'`
    - 로그 스케일: `set_yscale('log')` vs `log_y=True`
- **범례(Legend) 및 테두리:**
    - 위치: `bbox_to_anchor` vs `update_layout(legend_x=...)`
    - 테두리 강조: `ax.spines` vs `update_xaxes(showline=True)`
- **기준선 (Reference Line):**
    - `ax.axhline` / `ax.axvline` vs `fig.add_hline` / `fig.add_vline`

#### 4. 고급 커스터마이징 (Advanced Customization)
- **FacetGrid 매핑 튜닝:**
    - 사용자 정의 함수(`def custom(...)`)를 만들어 `g.map(custom, ...)`으로 각 서브플롯 별 기준선, 평균선, 텍스트 등을 개별 적용
    - 조건부 강조: 특정 조건(`spec_out`) 만족 시 테두리 색상 변경 또는 화살표 표시
- **회귀식 표시:**
    - `scipy.stats.linregress`로 계수 산출 후 텍스트로 표시 (Matplotlib)
    - `px.get_trendline_results`로 OLS 결과 추출 후 표시 (Plotly)
- **색상 (Color Palette):**
    - **Seaborn:** `palette` (범주형), `cmap` (연속형), `LinearSegmentedColormap` (커스텀)
    - **Plotly:** `color_discrete_sequence`, `color_continuous_scale`
- **실습 파일:**
    - [`part4-1_visualization_basics.md`](./part3-1_visualization_basics.md): 시각화 기초 학습 노트
    - [`part4-2_visualization_advanced.md`](./part3-2_visualization_advanced.md): 시각화 심화(히스토그램/히트맵/Facet/튜닝) 상세 학습 노트

<br>

### Part 5: 머신러닝 심화 & 실전
- **목표:** 머신러닝의 전체 파이프라인(전처리-학습-튜닝-평가)을 마스터하고, 특히 데이터 누수 방지를 위한 전처리 원칙과 다양한 모델(Tree, Boosting, Ensemble)의 최적화 기법을 숙달
- **학습 환경:** Google Colab
- **주요 학습 내용:**

#### 1. 머신러닝 프로세스 & EDA
- **기본 원칙:**
    - **Shape Check:** `train.shape`와 `test.shape`, 그리고 제출할 `result.shape`의 **행(Row)의 수**는 반드시 같아야 함.
    - **접근 방식:** 데이터 + 해답 $\rightarrow$ 규칙 도출 (학습)
- **탐색적 데이터 분석 (EDA):**
    - 기본 확인: `head()`, `info()`, `describe()` (수치형), `describe(include='object')` (범주형)
    - 시각화: `histplot` (분포), `heatmap` (상관관계 `df.corr()`)
    - 타겟 확인: `value_counts(normalize=True)` (비율 확인, 불균형 여부)

#### 2. 데이터 전처리 (Preprocessing) **[핵심]**
- **결측치 처리 (Missing Values):**
    - **Train:** 삭제(`dropna`) 가능.
    - **Test:** 최빈값/평균값/중앙값으로 채우거나(`fillna`), 'X' 같은 새 범주로 대체.
- **이상치 처리 (Outliers):**
    - Train에서만 삭제 가능. Log 변환(`np.log1p`), Sqrt 변환, **PowerTransformer** (Yeo-Johnson) 활용 권장.
- **인코딩 (Encoding):**
    - **Label Encoding:** 트리 모델(RF, XGB, LGBM)에 적합. `fit`은 Train, `transform`은 Train/Test 모두.
    - **One-Hot Encoding:** 선형 모델에 적합. `pd.get_dummies` (Train/Test 열 개수 불일치 시 `concat` 후 인코딩 추천).
- **스케일링 (Scaling):**
    - **원칙:** `fit`은 **Train** 데이터로만, **Test**는 `transform`만 적용. (2차원 배열 입력 필수)
    - 종류: `MinMaxScaler`, `StandardScaler`, `RobustScaler` (이상치 강함).
    - 트리 모델은 스케일링 불필요, 선형/신경망 모델은 필수.

#### 3. 모델 학습 (Model Training)
- **알고리즘:**
    - **Tree 계열:** DecisionTree, **RandomForest** (Bagging), **XGBoost**, **LightGBM** (Boosting).
    - **Linear 계열:** LinearRegression, LogisticRegression (분류).
    - **기타:** KNN, SVC (정규화 필수), VotingClassifier (앙상블).
    - **비지도 학습:** KMeans (Elbow point 확인), DBSCAN (밀도 기반).
- **검증 분할:** `train_test_split(stratify=target)` (분류 시 클래스 비율 유지).

#### 4. 모델 튜닝 & 최적화
- **GridSearchCV:** 하이퍼파라미터 격자 탐색 (`best_params_`, `best_score_`).
- **Early Stopping:** XGBoost/LightGBM에서 과적합 방지 (`eval_set`, `early_stopping_rounds`).
- **Feature Engineering:**
    - 차원 축소: PCA (시각화 및 차원의 저주 해결).
    - 변수 선택: `SelectKBest`.
    - 파생 변수: 다항 회귀(`PolynomialFeatures`), 구간화(`cut`, `qcut`).

#### 5. 모델 평가 (Evaluation)
- **분류 (Classification):**
    - **Metrics:** Accuracy, Precision, Recall, F1-Score (불균형 데이터), ROC-AUC (확률 `predict_proba`).
    - **Confusion Matrix:** TP, TN, FP, FN 확인.
- **회귀 (Regression):**
    - **Metrics:** RMSE (`root_mean_squared_error`), MSE, MAE, R2 Score.
- **군집 (Clustering):** Silhouette Score.
- **실습 파일:**
    - [`part5-1_ml.md`](./part5-1_ml.md): 머신러닝 기초/심화 프로세스 상세 학습 노트

<br>

### Part 6: 딥러닝
- **목표:** 딥러닝의 기초 원리(퍼셉트론, 역전파)를 이해하고, TensorFlow/Keras를 활용하여 비정형 데이터(이미지, 시계열, 텍스트) 처리를 위한 핵심 네트워크(CNN, RNN, GAN)를 구현 및 최적화
- **학습 환경:** Google Colab, TensorFlow, Keras
- **주요 학습 내용:**

#### 1. 딥러닝 기초 (Fundamentals)
- **개념:** Feature를 스스로 학습(Representation Learning)하며, 비정형 데이터(이미지, 소리, 언어)에 강점.
- **핵심 요소:**
    - **퍼셉트론 & 활성화 함수:** Sigmoid(기울기 소실 문제), **ReLU**(표준), Softmax(다중 분류), Tanh.
    - **학습 원리:** 경사하강법(Gradient Descent), 오차역전파(Backpropagation), 손실함수(MSE, Cross-Entropy).
    - **Optimizer:** **Adam** (가장 많이 사용), RMSprop, SGD.
    - **Hyperparameter:** Epoch, Batch Size, Learning Rate, Dropout Rate.
- **구현 패턴 (TensorFlow):**
    - `Sequential()` 모델 생성 $\rightarrow$ `add(Dense(...))` 레이어 적층 $\rightarrow$ `compile()` $\rightarrow$ `fit()` $\rightarrow$ `evaluate()`

#### 2. 합성곱 신경망 (CNN) - 이미지 처리
- **구조:**
    - **Convolution Layer:** 필터(Kernel)를 통해 지역적 특징(Feature Map) 추출. (Padding, Stride 활용)
    - **Pooling Layer:** Max/Average Pooling으로 차원 축소 및 중요 정보 보존 (과적합 방지).
    - **Flatten:** 1차원 배열로 변환하여 Fully Connected Layer(Dense)로 전달.
- **전처리:**
    - 스케일링 (`/ 255.0`), 차원 확장 (`expand_dims`), 원핫 인코딩 (`to_categorical`).
    - **Image Augmentation:** `ImageDataGenerator` (이미지 증강을 통한 데이터 확보).
- **전이 학습 (Transfer Learning):**
    - Pre-trained Model (MobileNet, VGG, ResNet) 활용.
    - 전략: Feature Extractor(Freeze) + Fine Tuning (마지막 층만 재학습).

#### 3. 순환 신경망 (RNN & LSTM) - 시계열/NLP
- **특징:** 시퀀스(Sequence) 데이터 처리, 과거의 기억(Hidden State)을 현재 결과에 반영.
- **한계 및 극복:**
    - Vanilla RNN의 기울기 소실 문제 $\rightarrow$ **LSTM** (Forget Gate로 기억 조절), **GRU**로 발전.
- **실습:**
    - **주식 가격 예측 (Many-to-One):** `Window` 단위 시계열 데이터 구성, `MinMaxScaler` 필수.
    - **감성 분석 (NLP):** `Embedding` 레이어(단어 벡터화) + `Bidirectional LSTM`.

#### 4. 비지도 딥러닝 (Unsupervised Deep Learning)
- **Autoencoder (AE):**
    - 구조: Encoder(압축) $\rightarrow$ Latent Space(잠재 공간) $\rightarrow$ Decoder(복원).
    - 활용: 차원 축소, 노이즈 제거(Denoising), 이상치 탐지.
- **GAN (Generative Adversarial Network):**
    - 구조: **Generator**(위조지폐범) vs **Discriminator**(경찰)의 경쟁적 학습.
    - 원리: Generator는 Discriminator를 속이도록(1), Discriminator는 가짜를 0으로 맞추도록 학습.
    - 활용: 가짜 이미지 생성, 스타일 변환 등.

#### 5. 최신 트렌드 (Brief)
- **Transformer:** RNN 구조 제거, Attention 메커니즘만으로 성능 극대화.
- **Encoder-Decoder:** 번역(Seq2Seq) 등에 활용.
- **실습 파일:**
    - [`part6_deep_learning_basics.md`](./part6_deep_learning_basics.md): DeepLearning 학습노트

<br>

### Part 7: 웹 대시보드 개발 (Streamlit)
- **목표:** HTML/CSS/JS 없이 Python만으로 데이터 분석 결과를 '인터랙티브 웹 애플리케이션(대시보드)'으로 구현하고 배포하는 과정을 학습
- **학습 환경:** VS Code, Terminal
- **주요 학습 내용:**

#### 1. 기본 설정 및 텍스트 위젯
- **설치 및 실행:**
    - 설치: `pip install streamlit`
    - 실행: `streamlit run app.py` (터미널)
    - 변환: `jupyter nbconvert --to script` (ipynb -> py)
- **텍스트 출력:**
    - `st.title`, `st.header`, `st.subheader`: 제목 계층 구조
    - `st.text`, `st.write`: 기본 텍스트 출력
    - `st.markdown`: 마크다운 문법 지원 (색상, 볼드 등 `:red[]`, `:blue[]`)
    - `st.code`: 코드 블록 출력 (`language='python'`)
    - `st.divider`: 구분선

#### 2. 입력 위젯 (Interaction)
- **버튼 및 선택:**
    - `st.button`: 클릭 이벤트 처리 (`on_click`, `type='primary'`)
    - `st.checkbox`: True/False 반환 (`if`문 제어)
    - `st.toggle`: 스위치 형태의 토글
    - `st.radio`: 단일 선택
    - `st.selectbox`: 드롭다운 선택 (`index=None`, `placeholder`)
    - `st.multiselect`: 다중 선택 (리스트 반환)
- **수치 및 파일:**
    - `st.slider`: 값 또는 범위(Range) 선택 (시간, 수치 등)
    - `st.text_input`: 문자열 입력 (`type='password'`로 암호화 가능)
    - `st.file_uploader`: CSV 등 파일 업로드 (`accept_multiple_files`)

#### 3. 시각화 및 미디어 (Visualization)
- **Matplotlib & Seaborn:** `st.pyplot(fig)`
- **Plotly Express:** `st.plotly_chart(fig)` (인터랙티브 차트)
- **동적 그래프:** `selectbox`로 축(x, y, hue)을 선택받아 실시간으로 그래프 업데이트
- **이미지:** `st.image(img)` (PIL 라이브러리 연동, `width`, `caption`)

#### 4. 레이아웃 (Layout)
- **화면 구성:**
    - `st.sidebar`: 사이드바 메뉴 구성 (`with st.sidebar:`)
    - `st.columns`: 화면을 가로 단으로 분할 (`col1, col2 = st.columns(2)`)
    - `st.tabs`: 탭 구성 (`tab1, tab2 = st.tabs(['Table', 'Graph'])`)
    - `st.expander`: 접기/펼치기 가능한 영역 (상세 데이터 숨김 등)

#### 5. 고급 기능 (Advanced)
- **Session State (상태 유지):**
    - Streamlit은 상호작용 시마다 전체 코드가 재실행되어 변수가 초기화됨.
    - `st.session_state`를 사용하여 버튼 클릭 후에도 변수 값(예: 카운터)을 유지.
- **Caching (성능 최적화):**
    - `@st.cache_data`: 데이터 로드나 무거운 연산 결과를 메모리에 저장하여 재실행 시 속도 향상.

#### 6. 실전 프로젝트: 자동차 CO2 배출량 대시보드
- **기능 구현:**
    - **Data Load:** `pd.read_csv`로 데이터 로드.
    - **Sidebar:** 제조사(`multiselect`) 및 엔진 크기(`slider`) 필터링 기능 구현.
    - **Main Page:**
        1. **Box Plot:** 제조사별 엔진 크기 분포.
        2. **Scatter Plot 1:** 엔진 크기 vs 연비 (회귀선 포함).
        3. **Scatter Plot 2:** 연비 vs 탄소 배출량 상관관계 (축 선택 기능 포함).
    - **State Management:** 필터링된 데이터를 `st.session_state`와 연동하여 동적 시각화 구현.
- **실습 파일:**
    - [`part7_streamlit_study.md`](./part7_streamlit_study.md): streamlit 상세 학습 노트
    - [`part7_streamlit_dashboard.py`](./part7_streamlit_dashboard.py): 실전 프로젝트 대시보드 전체 소스 코드
    - [실습 페이지 링크](https://digital-audit-portfolio-zkwnvfuoyhgcbwqhxyypbh.streamlit.app/)

<br>

### Part 8: SQL 데이터베이스 & 고급 쿼리 분석
- **목표:** 관계형 데이터베이스(RDBMS)의 구조와 DDL/DML 기초, 복잡한 데이터 분석을 위한 고급 쿼리(Join, Subquery, Window Function)와 트랜잭션 제어 능력을 함양
- **학습 환경:** MySQL, Python (PyMySQL)
- **주요 학습 내용:**

#### 1. RDBMS 이론 및 DDL (Data Definition)
- **데이터베이스(DB):** 데이터의 집합, 중복 최소화 및 무결성 유지.
- **DDL (구조 정의):** `CREATE`, `ALTER`, `DROP`, `TRUNCATE`.
- **제약 조건:** `PRIMARY KEY` (고유 식별), `FOREIGN KEY` (관계 형성), `NOT NULL`, `AUTO_INCREMENT`.
- **인덱스(Index):** 검색 속도 향상 (Clustered vs Secondary).

#### 2. SQL 쿼리 작동 원리 및 기초 (DML Basics)
- **쿼리 실행 순서 (중요):**
    `FROM` $\rightarrow$ `ON` $\rightarrow$ `JOIN` $\rightarrow$ `WHERE` $\rightarrow$ `GROUP BY` $\rightarrow$ `HAVING` $\rightarrow$ `SELECT` $\rightarrow$ `DISTINCT` $\rightarrow$ `ORDER BY` $\rightarrow$ `LIMIT`
- **기본 문법:**
    - `SELECT`: `DISTINCT`(중복제거), `AS`(별칭).
    - `LIMIT`: 출력 행 수 제한.
    - `GROUP BY` & `HAVING`: 그룹화 및 그룹 후 조건 필터링.
    - `ORDER BY`: 정렬 (`ASC`, `DESC`).

#### 3. 주요 단일 행 함수 (Functions)
- **문자열 함수:**
    - `LENGTH()`, `UPPER()`, `LOWER()`
    - `CONCAT(str1, str2)`: 문자열 연결.
    - `SUBSTRING(str, start, len)`: 문자열 추출.
- **날짜/시간 함수:**
    - `NOW()`, `CURDATE()`, `CURTIME()`
    - `DATE_ADD()`, `DATE_SUB()`: 날짜 연산 (`INTERVAL 7 DAY`).
    - `EXTRACT(unit FROM date)`: 연/월/일 추출.
    - `TIMESTAMPDIFF(unit, start, end)`: 시간 차이 계산.
    - `DATE_FORMAT(date, format)`: 형식 변환 (`%Y-%m-%d`).
- **숫자 함수:** `ABS()`, `CEIL()`, `FLOOR()`, `ROUND()`, `SQRT()`.

#### 4. 데이터 결합 및 집합 연산 (Joins & Sets)
- **JOIN:**
    - `INNER JOIN`: 교집합 (매칭되는 데이터만).
    - `LEFT/RIGHT OUTER JOIN`: 기준 테이블 모두 + 매칭되는 데이터.
- **집합 연산:**
    - `UNION` (중복 제거 합집합), `UNION ALL` (중복 포함 합집합).
    - `INTERSECT` (교집합), `EXCEPT` (차집합).

#### 5. 고급 쿼리 기법 (Advanced Techniques)
- **서브쿼리 (Subquery):**
    - `WHERE`절 (필터링), `FROM`절 (인라인 뷰), `SELECT`절 (스칼라).
    - **상관 서브쿼리:** 메인 쿼리의 컬럼을 참조하여 수행.
- **CTE (Common Table Expression):** `WITH` 절을 사용한 임시 테이블 (가독성 향상).
- **VIEW:** 가상 테이블 생성 (`CREATE VIEW`).
- **조건 제어:** `CASE WHEN condition THEN result ELSE end`.
- **문자열 집계:** `GROUP_CONCAT()` (여러 행을 하나의 문자열로 결합).
- **트랜잭션:** `COMMIT` (저장), `ROLLBACK` (취소).

#### 6. 윈도우 함수 (Window Functions) **[핵심]**
- **개념:** `GROUP BY`와 달리 원본 행을 유지하면서 집계 및 순위 계산 수행.
- **문법:** `FUNCTION() OVER (PARTITION BY ... ORDER BY ... ROWS/RANGE ...)`
- **순위 함수:**
    - `RANK()`: 1, 2, 2, 4... (건너뜀)
    - `DENSE_RANK()`: 1, 2, 2, 3... (연속)
    - `ROW_NUMBER()`: 1, 2, 3, 4... (고유 번호)
- **데이터 참조:**
    - `LEAD()`: 다음 행 값, `LAG()`: 이전 행 값.
    - `FIRST_VALUE()`, `LAST_VALUE()`: 파티션 내 처음/마지막 값.
- **비율/분할:** `PERCENT_RANK()`, `CUME_DIST()`, `NTILE(n)`.
- **집계:** `SUM() OVER()`, `AVG() OVER()` (누적 합계 등).

#### 7. Python 연동 (PyMySQL)
- `connect` $\rightarrow$ `cursor` $\rightarrow$ `execute(SQL)` $\rightarrow$ `commit` $\rightarrow$ `close`
- **실습 파일:**
    - [`part8_sql.md`](./part8_sql.md): sql 학습노트
