# 파이썬을 활용한 통계분석

# 가설검정

## 모집단과 표본

- 모집단 : 연구 대상이 되는 전체 집단
- 표본 : 모집단의 일부

## 귀무가설과 대립가설

- 귀무가설 : 기존에 알려진 사실, 효과나 차이가 없음
- 대립가설 : 연구자가 입증 하려는 사실, 효과나 차이가 있음

## 검정결과

- 검정통계량 : 주어진 데이터와 귀무가설 간의 차이를 통계적으로 나타낸 값
- p-value(유의수준 0.05)
    - 유의수준 보다 작으면 귀무가설을 기각, 대립가설을 채택
    - 유의수준 보다 크면 귀무가설을 기각하지 못함, 귀무가설 채택
- 가설검정 프로세스
    - 통계적 가설 설정 → 유의수준 결정 → 검정 통계량 및 p-value 계산 → 결과도출
- t-test
    - 단일표본검정 : 모집단 1개 : 한 그룹
    - 대응(쌍체)표본검정 : 모집단 2개(같은 집단) : 같은 그룹
    - 독립표본검정 : 모집단 2개 : 서로 다른 그룹
- 단일표본검정
    - 관심 모집단1개 → 정규성 검정(shapiro test) → yes : 단일표본검정, no : wilcoxon 검정
- 대응표본검정
    - 관심 모집단2개 → 정규성 검정(shapiro test) → yes : 대응표본검정, no : wilcoxon 검정
- 독립표본검정
    - 관심 모짐단2개 → 정규성 검정(shapiro test) → yes : 등분산검정(levene), no : Mann-Whitney U 검정

## 단일 표본검정

나커피 유튜버는 최근 판매되는 "합격 원두(dark)" 상품의 실제 무게를 조사하였다. 제조사는 "합격 원두(dark)"의 무게를 120g라고 표기하였다. 나커피 유튜버는 이 주장이 사실인지 확인하기 위해 상품의 무게를 측정하였다. 다음은 30개의 상품 샘플의 무게 측정 결과다. 이 측정 결과를 바탕으로 제조사의 주장인 상품의 무게(120g)가 사실인지 검정해보시오. **(데이터는 정규분포를 따른다고 가정한다.)**

- 귀무가설: μ = μ0, "합격 원두(dark)" 상품의 평균 무게는 120g이다.
- 대립가설: μ ≠ μ0, "합격 원두(dark)" 상품의 평균 무게는 120g 아니다.
- μ(뮤): 현재 조사하려는 상품의 실제 평균 무게
- μ0(뮤 제로): 제조사가 주장하는 상품의 평균 무게(120g)

### 기초

from scipy import stats

stats.ttest_1samp(df[’무게’], 120))

TtestResult(statistic=np.float64(2.153709967150663), pvalue=np.float64(0.03970987897788578), df=np.int64(29)) ← pvalue가 0.0397로 0.05보다 적음 : 귀무가설 기각

- 양측검정(defalut 값은 양측검정임) : 평균 무게는 120g이 아니다

print(stats.ttest_1samp(df['무게'], 120, **alternative='two-sided'**))

TtestResult(statistic=np.float64(2.153709967150663), pvalue=np.float64(0.03970987897788578), df=np.int64(29))

- 단측검정 : 평균 무게가 120g보다 크다

print(stats.ttest_1samp(df['무게'], 120, **alternative='greater'**))

TtestResult(statistic=np.float64(2.153709967150663), pvalue=np.float64(0.01985493948894289), df=np.int64(29))

- 단측검정 : 평균 무게가 120g보다 작다

print(stats.ttest_1samp(df['무게'], 120, **alternative='less'**))

TtestResult(statistic=np.float64(2.153709967150663), pvalue=np.float64(0.9801450605110571), df=np.int64(29))

### 정규분포를 따르지 않을 경우

- 정규분포인지 아닌지 확인 : Shapiro-Wilk 검정
    - 귀무가설 : 정규분포를 따른다
    - 대립가설 : 정규분포를 따르지 않는다
    
    ★보통의 귀무가설과 대립가설이랑 다름 → p-value가 0.05 이상이여야 정규분포를 따른다
    

from scipy import stats

stats.shapiro(df[’무게’])
`ShapiroResult(statistic=np.float64(0.35728970196526855), pvalue=np.float64(2.2139240997414947e-10))` → p-value가 0.05이하 → 정규분포 안따름

- 정규분포를 따르지 않는경우 : Wilcoxon 검정
    - 귀무가설 : 평균 무게는 120g이다
    - 대립가설 : 평균 무게는 120g보다 작다

stats.wilcoxon(df['무게'] - 120, alternative='less')
`WilcoxonResult(statistic=np.float64(341.0), pvalue=np.float64(0.9874090523115628))` → p-value가 0.05이상 → 귀무가설 채택

## 대응표본검정 (전, 후의 점수변화를 비교할때)

퇴근후딴짓 크리에이터는 수험생의 점수 향상을 위해 새로운 교육 프로그램을 도입했다. 도입 전과 도입 후의 점수 차이를 확인하기 위해 동일한 수험생의 점수를 비교하였습니다. 다음은 교육 전과 후의 점수 데이터이다. 새로운 교육 프로그램이 효과가 있는지 검정하시오.(데이터는 정규분포를 따른다고 가정한다.)

μd = (before – after)의 평균

- 귀무가설: μ ≥ 0, 새로운 교육 프로그램은 효과가 없다.
- 대립가설: μ < 0, 새로운 교육 프로그램은 효과가 있다.

### 기초

from scipy import stats

stats.ttest_rel(df[’before’], df[’after’], alternative=’less’) ←  alternative = ‘less’ : before 이 더 작을것이라는 가설
`TtestResult(statistic=np.float64(-2.119860886666164), pvalue=np.float64(0.03152591671694539), df=np.int64(9))`

### if ud = (after-before)일 경우

stats.ttest_rel(df['after'], df['before'], alternative='greater') ← alternative = ‘greater’ : after가 더 클 것이라는 가설
`TtestResult(statistic=np.float64(2.119860886666164), pvalue=np.float64(0.03152591671694539), df=np.int64(9))`

### 정규분포를 따르지 않는경우

- 정규성 확인 : Shapiro-Wilk

df[’diff’] = df[’after’] - df[’before’]

from scipy import status

stats.shapiro(df[’diff’])
`ShapiroResult(statistic=np.float64(0.8106808132606462), pvalue=np.float64(0.019542902973577702))` ← p-value 0.05 이하 : 정규성분포 따르지 않는다

- 정규분포 따르지 않는 경우 : Wilcoxon 검정(비모수검정)

from scipy import status

stats.wilcoxon(df[’after’], df[’before’], alternative=’greater’)
`WilcoxonResult(statistic=np.float64(47.5), pvalue=np.float64(0.017578125))`

## 독립표본검정 : A,B 평균에 유의미한 차이가 있는지

다음은 빅데이터 분석기사 실기 시험 점수이다. A그룹과 B그룹의 평균 점수가 차이가 있는지 유의수준 0.05하에서 가설 검정하시오. (데이터는 정규분포를 따르고 **분산이 동일하다고 가정**한다.)

- 귀무가설(H0): 그룹별 시험 평균 점수는 차이가 없다. (μ1 = μ2)
- 대립가설(H1): 그룹별 시험 평균 점수는 차이가 있다. (μ1 ≠ μ2)

### 기초

from scipy import stats

stats.ttest_ind(A, B) ← (처리집단, 대조집단)

`TtestResult(statistic=np.float64(-2.051813915505951), pvalue=np.float64(0.04964542271174967), df=np.float64(28.0))`

### 두 집단간 분산이 다를 경우

stats.ttest_ind(A, B, **equal_var=False**)

### 단측검정(B그룹 시험 평균 점수가 더 높다)

stats.ttest_ind(A, B, alternative=’less’) A가더 작아야하니까 alternative = ‘less’

### 단측검정(A그룹 시험 평균 점수가 더 높다)

stats.ttest_ind(A,B, alternative=’greater’) A가 더 높아야하니까 alternative = ‘greater’

### 정규분포 따르는지 검정

from scipy import stats

stats.shapiro(A)

stats.shapiro(B)

if p-value가 0.05보다 적다(정규분포를 따르지 않는다) → Mann-Whitney 검정으로

if p-value가 0.05보다 크다(정규분포를 따른다) → Levene 검정으로 등분산 인지 확인

### Mann-Whitney U 검정

stats.mannwhitneyu(A, B, alternative=’less’)

### Levene 검정

stats.levene(A, B)

p-value가  0.05 미만 → 등분산이 다르다

독립표본 검정시 equal_var=False 설정 해야함

# 범주형 분석

## 카이제곱 검정

### 적합도 검정

- 관찰도수와 기대도수의 차이
- 빈도(count)로 변경(관찰값, 기대값)
- scipy.stats.chisquare(observed, expected)
    - observed : 관찰된 **빈도** 리스트
    - expectred : 기대 **빈도** 리스트
    

[문제] 지난 3년간 빅데이터 분석기사 점수 분포가 60점 미만: 50%, 60-70점 35%, 80점이상 15%로였다.
이번 회차부터 단답형을 제외하고, 작업형3을 추가하여 300명을 대상으로 적용한 결과 60점 미만: 150명, 60-70점: 120명, 80점이상: 30명이었다. 유의수준 0.05일 때, 새로운 시험문제 유형과 기존 시험문제 유형은 점수에 차이가 없는지 검정하시오.
-	귀무가설(H0): 새로운 시험문제는 기존 시험문제 점수와 동일하다.
-	대립가설(H1): 새로운 시험문제는 기존 시험문제 점수와 다르다.

#관찰
ob = [150, 120, 30]
#기대
ex = [0.5***300**, 0.35***300**, 0.15***300**] ← ★확률값이 아닌 빈도 값으로 들어가야함

from scipy import stats
stats.chisquare(ob, ex)
`Power_divergenceResult(statistic=7.142857142857142, pvalue=0.028115659748972056)` ← p-value 0.05이하 귀무가설 기각

### 독립성 검정

- 두 변수가 서로 독립적인지
- 교차표 테이블 작성
- scipy.stats.chi2_contingency(table, corrention=True)
    - table : 교차표
    - correction : 연속성 보정(기본값 True)

### 동질성 검정

- 두 개 이상의 집단에서 동질성을 갖는지 확인
- 검정 절차는 독립성과 같음

독립성 검정 = 동질성 검정

[문제] 빅데이터 분석기사 실기 언어 선택에 따라 합격 여부를 조사한 결과이다. 언어와 합격 여부는 독립적인가? 가설검정을 실시하시오. (유의수준 0.05)
-	귀무가설(H0): 언어와 합격 여부는 독립이다.
-	대립가설(H1): 언어과 합격 여부는 독립이지 않다.

- 교차표 데이터 만들기
- R: 합격 80명, 불합격 20명,
- Python: 합격 90명, 불합격 10명

import pandas as pd
df = pd.DataFrame({
    '합격':[80, 90],
    '불합격':[20, 10]
    },index=['R', 'P'])

from scipy import stats
stats.chi2_contingency(df)

Chi2ContingencyResult(statistic=3.1764705882352944, pvalue=0.07470593331213068, dof=1, expected_freq=array([[85., 15.],
       [85., 15.]]))

- 로우데이터 처리하기

import pandas as pd
data = {
    '언어': ['R']*100 + ['Python']*100,
    '합격여부': ['합격']*80 + ['불합격']*20 + ['합격']*90 + ['불합격']*10
}
df = pd.DataFrame(data)

df = **pd.crosstab**(df['언어'], df['합격여부'])
df

stats.chi2_contingency(df)

Chi2ContingencyResult(statistic=3.1764705882352944, pvalue=0.07470593331213068, dof=1, expected_freq=array([[15., 15.],
       [85., 85.]]))

# 회귀분석

### 상관계수

- 두 변수 간의 선형 관계의 강도와 방향 (-1≤r≤1)
    - r = 1 강한 양의 선형관계
    - r = 0 선형 관계 없음
    - r = -1 강한 음의 선형관계

# 데이터
import pandas as pd
df = pd.DataFrame({
    '키': [150, 160, 170, 175, 165],
    '몸무게': [42, 52, 75, 67, 56]})

df.corr()

print(df['키'].corr(df['몸무게']))
print(df['몸무게'].corr(df['키'])

- 피어슨

print(df.corr()) ← 기본값

- 스피어맨

print(df.corr(method='spearman'))

- 켄달타우

print(df.corr(method='kendall'))

### 상관계수에 대한 t검정

- 귀무가설 : 상관관계가 없다
- 대립가설 : 상관관계가 있다
- stats.pearsonr(x, y)
- stats.spearman(x, y)
- stats.kendalltau(x, y)

t검정

from scipy import stats

피어슨

print(stats.pearsonr(df['몸무게'], df['키']))

스피어맨

print(stats.spearmanr(df['몸무게'], df['키']))

켄달타우

print(stats.kendalltau(df['몸무게'], df['키']))

### 단순 선형 회귀분석

- OLS : 최소제곱법(Ordinary Least Squares)
- ols(’종속변수~독립변수’, data=df).fit()
- model.summary() ← 회귀 모델 통계적 요약
- model_predict() ← 예측값
- model.get_prediction() ←예측값과 예측값에 대한 신뢰구간, 예측구간
- df[’잔차’] = df[’종속변수’] - model.predict(df)

주어진 키와 몸무게 데이터로 회귀모델을 구축하고 각 소문제의 값을 구하시오.
- 키: 종속변수
- 몸무게: 독립변수

import pandas as pd

df = pd.DataFrame({
    '키': [150, 160, 170, 175, 165, 155, 172, 168, 174, 158,
          162, 173, 156, 159, 167, 163, 171, 169, 176, 161],
    '몸무게': [74, 50, 70, 64, 56, 48, 68, 60, 65, 52,
            54, 67, 49, 51, 58, 55, 69, 61, 66, 53]})

from statsmodels.formula.api import ols
model = ols('키 ~ 몸무게', data=df).fit()
print(model.summary())

![image.png](image.png)

- 결과 해석

model.rsquared ← 결정계수(0.28)

model.params[’몸무게’] ← 기울기(0.4938)

model.params[’Intercept’] ← 절편(135.8209)

model.pvalues[’몸무게’] ←p-value(0.017)

- 몸무게가 50일때 예측

newdata = pd.DataFrame({’몸무게’:[50]})

model.predict(newdata)

- 잔차 제곱합
- 잔차 = 관측(실제)값 - 예측값

df[’잔차’] = df[’키’] - model.predict(df[’몸무게’])

sum(df[’잔차’]**2)

- MSE

(df[’잔차’]**2).mean()

- 사이킷런 MSE

from sklearn.netrics import mean_squarred_error

pred = model.predict(df)

mean_squared_error(df[’키’], pred)

- 신뢰구간

→ 0.101 ~ 0.886

- 몸무게가 50일때 예측키에 대한 신뢰구간, 예측구간

newdata = pd.DataFrmae({’몸무게’ : [50]})

pred = model.get_prediction(newdata)

pred.summary_frame(alpha=0.05)

![image.png](image%201.png)

신뢰구간 : 155.695318 ~ 165.323136

예측구간 : 146.068566 ~ 174.949888

### 다중 선형 회귀분석

- ols(’종속변수~ 독립변수1 + 독립변수2’, data=df).fit()

주어진 매출액, 광고비, 플랫폼 데이터로 회귀모델을 구축하고 각 소문제의 값을 구하시오.
- 매출액: 종속변수
- 광고비, 플랫폼(유통 플랫폼 수), 투자: 독립변수

import pandas as pd
df = pd.DataFrame({
    '매출액': [300, 320, 250, 360, 315, 328, 310, 335, 326, 280,
            290, 300, 315, 328, 310, 335, 300, 400, 500, 600],
    '광고비': [70, 75, 30, 80, 72, 77, 70, 82, 70, 80,
            68, 90, 72, 77, 70, 82, 40, 20, 75, 80],
    '플랫폼': [15, 16, 14, 20, 19, 17, 16, 19, 15, 20,
            14, 5, 16, 17, 16, 14, 30, 40, 10, 50],
    '투자':[100, 0, 200, 0, 10, 0, 5, 0, 20, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0] })

from statsmodels.formula.api import ols
model = ols('매출액 ~ 광고비 + 플랫폼', data=df).fit()
print(model.summary())

![image.png](image%202.png)

- 결과 해석

model.rsquared ← 결정계수(0.512)

model.params[’광고비’] ← 기울기(1.8194)

model.params[’플랫폼’] ← 기울기(5.9288)

model.params[’Intercept’] ← 절편(101.0239)

model.pvalues[’광고비’] ←p-value(0.038)

model.pvalues[’플랫폼’] ←p-value(0.001

- 광고비 50, 플랫폼 20일 때 매출액 예측

newdata = pd.DataFrame({’광고비’:[50], ‘플랫폼’:[20]})

model.predict(newdata)

- 잔차 제곱합

(model.resid ** 2).sum()

- MSE

(model.resid **2)mean()

- 광고비, 플랫폼 회귀계수의 95% 신뢰구간

광고비 : 0.117 ~ 3.522

플랫폼 2.912 ~ 8.945

model.conf_int(alpha=0.05)

- 광고비 50, 플랫폼 20일 때 매출액 95%에 대한 신뢰구간과 예측구간

newdata = pd.DataFrame({’광고비’:[50], ‘플랫폼’:[20]})

pred = model.get_prediction(newdata)

pred.summary_frame(alpha=0.05)

신뢰구간 : 268.612221 ~ 352.52844

예측구간 : 179.700104 ~ 441.440556

### 범주형 변수

- 판다스의 pd.get_dummies(drop_first=True)로 원핫인코딩 처리

import pandas as pd
df = pd.DataFrame({
    '매출액': [300, 320, 250, 360, 315, 328, 310, 335, 326, 280,
            290, 300, 315, 328, 310, 335, 300, 400, 500, 600],
    '광고비': [70, 75, 30, 80, 72, 77, 70, 82, 70, 80,
            68, 90, 72, 77, 70, 82, 40, 20, 75, 80],
    '플랫폼': [15, 16, 14, 20, 19, 17, 16, 19, 15, 20,
            14, 5, 16, 17, 16, 14, 30, 40, 10, 50],
    '투자':[100, 0, 200, 0, 10, 0, 5, 0, 20, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '유형':['B','B','C','A','B','B','B','B','B','B'
        ,'C','B','B','B','B','B','B','A','A','A']
    })

from statsmodels.formula.api import ols
model = ols('매출액 ~ 광고비 + 유형', data=df).fit()
print(model.summary())

![image.png](image%203.png)

- 원핫인코딩

df2 = pd.,get_dummies(df)

- 다중공선성 해결 → 첫번째 열 삭제

df = pd.get_dummies(df, drop_first=True)

- 회귀 모델 학습 & 서머리 (독립변수로 광고비와 유형만 활용)

from statsmodels.formula.api import ols
model = ols('매출액 ~ 광고비 + 유형_B + 유형_C', data=df).fit()
print(model.summary())

![image.png](image%204.png)

# 분산분석

- 분산분석(ANOVA) : 여러 집단의 평균 차이를 통계적으로 유의미한지 검정
- 일원 분산 분석 : 하나의 요인에 따라 평균의 차이 검정
- 이원 분산 분석 : 두개의 요인에 따라 평균의 차이 검정

## 일원 분산 분석

- 3개 이상의 집단 간의 평균의 차이가 통계적으로 유의적인지 검정
- 하나의 요인, 집단의 수가 3개 이상
1. 사이파이
    
    f_oneway(sample1, sample2, … )
    
2. 스테츠모델즈
    
    model = ols(’종속변수 ~ 요인’, data=df).fit()
    
    print(anova_lm(model))
    

df : 자유도

sum_sq : 제곱합 (그룹 평균 간의 차이를 나타내는 제곱합)

mean_sq : 평균 제곱 (sum_sq/자유도)

F : 검정통계량

PR(>F): p-value

### 프로세스

![image.png](image%205.png)

정규성검정(샤피로) → yes: 등분산검정, no:크루스칼-윌리스 비모수검정 → 일원분산분석 → 사후 : Turkey HSD or Bonferroni

### 기본가정

- 독립성 : 각 집단의 관측치는 독립적이다.
- 정규성 : 각 집단은 정규분포를 따른다 (샤피로 검정)
- 등분산성 : 모든 집단은 동일한 분산을 가진다. (레빈 검정)

### 귀무가설과 대립가설

- 귀무가설 : 모든 집단의 평균은 같다
- 대립가설 : 적어도 한 집단은 평균이 다르다

주어진 데이터는 4가지 다른 교육 방법을 적용한 대학생들의 학점 결과이다. 이 실험에서는 비슷한 실력을 가진 학생 40명을 무작위로 4개(A, B, C, D)그룹으로 나누었고, 각 그룹은 다른 교육 방법을 적용했다. 학생들의 학점 결과에는 교육 방법에 따른 차이가 있는지 유의수준 0.5하에서 검정하시오.
- 귀무가설(H0): 네 가지 교육 방법에 의한 학생들의 학점 평균은 동일하다.
- 대립가설(H1): 적어도 두 그룹의 학점 평균은 다르다.

import pandas as pd
df = pd.DataFrame({
    'A': [3.5, 4.3, 3.8, 3.6, 4.1, 3.2, 3.9, 4.4, 3.5, 3.3],
    'B': [3.9, 4.4, 4.1, 4.2, 4.5, 3.8, 4.2, 3.9, 4.4, 4.3],
    'C': [3.2, 3.7, 3.6, 3.9, 4.3, 4.1, 3.8, 3.5, 4.4, 4.0],
    'D': [3.8, 3.4, 3.1, 3.5, 3.6, 3.9, 3.2, 3.7, 3.3, 3.4]})

from scipy import stats
stats.f_oneway(df['A'], df['B'], df['C'], df['D'])

 정규성, 등분산, 일원 분산 분석

Shapiro-Wilk(샤피로-윌크) 정규성 검정 (p-value가 0.05보다 크면 정규성 있음)
print(stats.shapiro(df['A']))
print(stats.shapiro(df['B']))
print(stats.shapiro(df['C']))
print(stats.shapiro(df['D']))

 Levene(레빈) 등분산 검정
print(stats.levene(df['A'], df['B'], df['C'], df['D']))

 일원 분산 분석
print(stats.f_oneway(df['A'], df['B'], df['C'], df['D']))

- 데이터 재구조화

df_melt = df.melt()

(아노바테이블)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('value ~ variable', data=df_melt).fit()
anova_lm(model)

![image.png](image%206.png)

### 사후검정 : 어떤 그룹들간 통계적으로 유의미한 차이가 있는지 구체적으로 파악

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

 Tukey HSD (투키)
tukey_result = pairwise_tukeyhsd(df_melt['value'], df_melt['variable'], alpha=0.05)
print(tukey_result.summary())

 Bonferroni(본페로니)
mc = MultiComparison(df_melt['value'], df_melt['variable'])
bon_result = mc.allpairtest(stats.ttest_ind, method='bonf')
print(bon_result[0])

![image.png](image%207.png)

reject = True → 차이가 있는 그룹임

### 비모수검정

import pandas as pd
from scipy import stats

 데이터
df = pd.DataFrame({
    'A': [10.5, 11.3, 10.8, 10.6, 11.1, 10.2, 10.9, 11.4, 10.5, 10.3],
    'B': [10.9, 11.4, 11.1, 11.2, 11.5, 10.8, 11.2, 10.9, 11.4, 11.3],
    'C': [10.2, 10.7, 10.6, 10.9, 11.3, 11.1, 10.8, 10.5, 11.4, 11.0],
    'D': [13.8, 10.4, 10.1, 10.5, 10.6, 10.9, 10.2, 10.7, 10.3, 10.4]})

 정규성 검정
print(stats.shapiro(df['A']))
print(stats.shapiro(df['B']))
print(stats.shapiro(df['C']))
print(stats.shapiro(df['D']))

 Kruskal-Wallis 검정
stats.kruskal(df['A'], df['B'], df['C'], df['D'])

## 이원 분산 분석

- 요인의수가 2개, 집단의 수가 3개 이상일 때 사용

### 기본가정

- 독립성 : 각 집단의 관측치는 독립적이다.
- 정규성 : 각 집단은 정규분포를 따른다. (샤피로 검정)
- 등분산성 : 모든 집단은 동일한 분산을 가진다. (레빈 검정)

### 귀무가설과 대립가설

- 주 효과(요인1)
    - 귀무가설: 모든 그룹의 첫 번째 요인의 평균은 동일하다.
    - 대립가설: 적어도 두 그룹은 첫 번째 요인의 평균은 다르다.
- 주 효과(요인2)
    - 귀무가설: 모든 그룹의 두 번째 요인의 평균은 동일하다.
    - 대립가설: 적어도 두 그룹은 두 번째 요인의 평균은 다르다.
- 상호작용효과
    - 귀무가설: 두 요인의 그룹 간에 상호작용은 없다.
    - 대립가설: 두 요인의 그룹 간에 상호작용이 있다.

 스테츠모델즈 (아노바 테이블)
model = ols('종속변수 ~ C(요인1) * C(요인2)', data=df).fit()
print(anova_lm(model))

![image.png](image%208.png)

가정에서 재배하고 있는 네 가지 토마토 종자(A, B, C, D)에 대해 세 가지 종류의 비료 (11, 12, 13)를 사용하여 재배된 토마토 수를 조사하였다. 종자 및 비료 종류 간의 토마토 수의 차이가 있는지 유의수준 0.05하에서 검정하시오.
(단, 정규성, 등분산성에 만족한 데이터)
- 종자 (주 효과)
    - 귀무가설(H0): 종자 간의 토마토 수에 차이가 없다.
    - 대립가설(H1): 적어도 하나의 종자에서 토마토 수에 차이가 있다.
- 비료 (주 효과)
    - 귀무가설(H0): 비료 종류 간의 토마토 수에 차이가 없다.
    - 대립가설(H1): 적어도 하나의 비료 종류에서 토마토 수에 차이가 있다.
- 상호작용 효과:
    - 귀무가설(H0): 종자와 비료 간의 상호작용은 토마토 수에 영향을 미치지 않는다.
    - 대립가설(H1): 종자와 비료 간의 상호작용은 토마토 수에 영향을 미친다.

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/lovedlim/inf/main/p3/tomato.csv")

 anova 테이블
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('토마토수 ~ 종자 + 비료 + 종자:비료', data=df).fit()
anova_lm(model)

![image.png](image%209.png)

 범주형 데이터 처리
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('토마토수 ~ C(종자) + C(비료) + C(종자):C(비료)', data=df).fit()
anova_lm(model)

![image.png](image%2010.png)

 일반표기법 format(지수표기법, '.10f')
print(format(7.254117e-10,'.10f'))
print(format(1.835039e-03,'.10f'))
print(format(2.146636e-01,'.10f'))

 formula * 활용
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

 model = ols('토마토수 ~ C(종자) + C(비료) + C(종자):C(비료)', data=df).fit()
or model = ols('토마토수 ~ C(종자) * C(비료)', data=df).fit()
anova_lm(model)

![image.png](image%2011.png)

### 사후검정

  이원 분산 분석 수행
model = ols('토마토수 ~ C(종자) + C(비료) + C(종자):C(비료)', data=df).fit()
anova_lm(model)

![image.png](image%2012.png)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey HSD
tukey_summary1 = pairwise_tukeyhsd(df['토마토수'], df['종자'], alpha=0.05)
tukey_summary2 = pairwise_tukeyhsd(df['토마토수'], df['비료'].astype(str), alpha=0.05)
print(tukey_summary1)
print(tukey_summary2)

![image.png](image%2013.png)

# Bonferroni
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(df['토마토수'], df['종자'])
bon_result = mc.allpairtest(stats.ttest_ind, method="bonf", alpha=0.05)
print(bon_result[0])

mc = MultiComparison(df['토마토수'], df['비료'])
bon_result = mc.allpairtest(stats.ttest_ind, method="bonf")
print(bon_result[0])

![image.png](image%2014.png)

### 정규성, 등분산

from scipy.stats import shapiro

cond_tree_A = df['종자'] == 'A'
cond_tree_B = df['종자'] == 'B'
cond_tree_C = df['종자'] == 'C'
cond_tree_D = df['종자'] == 'D'

cond_fert_1 = df['비료'] == 11
cond_fert_2 = df['비료'] == 12
cond_fert_3 = df['비료'] == 13

print(shapiro(df[cond_tree_A & cond_fert_1]['토마토수']))
print(shapiro(df[cond_tree_A & cond_fert_2]['토마토수']))
print(shapiro(df[cond_tree_A & cond_fert_3]['토마토수']))

print(shapiro(df[cond_tree_B & cond_fert_1]['토마토수']))
print(shapiro(df[cond_tree_B & cond_fert_2]['토마토수']))
print(shapiro(df[cond_tree_B & cond_fert_3]['토마토수']))

print(shapiro(df[cond_tree_C & cond_fert_1]['토마토수']))
print(shapiro(df[cond_tree_C & cond_fert_2]['토마토수']))
print(shapiro(df[cond_tree_C & cond_fert_3]['토마토수']))

print(shapiro(df[cond_tree_D & cond_fert_1]['토마토수']))
print(shapiro(df[cond_tree_D & cond_fert_2]['토마토수']))
print(shapiro(df[cond_tree_D & cond_fert_3]['토마토수']))

from scipy.stats import levene
print(levene(df[cond_tree_A]['토마토수'],
             df[cond_tree_B]['토마토수'],
             df[cond_tree_C]['토마토수'],
             df[cond_tree_D]['토마토수']))
print(levene(df[cond_fert_1]['토마토수'],
             df[cond_fert_2]['토마토수'],
             df[cond_fert_3]['토마토수']))