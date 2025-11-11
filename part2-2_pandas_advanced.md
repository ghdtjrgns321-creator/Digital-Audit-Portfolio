# 판다스 추가 학습

- loc, iloc 복습

df.loc[’행’,’열’] ← 인덱스 ★명★

df.iloc[행,열] ← 인덱스 ★번호★

loc[조건문] → 조건만족하는 필터링 가능

ex ) df.loc[df.[’season’]==2]

ex ) df.loc[(df.[’season’==2) & (df.[’weather] !=2)]

- select_dtypes 메서드

★dtypes 에 s 필수!!★

df.select_dtypes(include=’int’) ← 정수형만 필터링

df.select_dtypes(exclude=’int’) ← 정수형이 아닌것만 필터링

★df.select_dtypes(include = [’int’, ‘float’]) ← 조건 2개달때 리스트형으로★

- set_index 메서드 ★★★ set_index → 소괄호 써야함 ★★★

df .set_index(’datetime’) ← 기존의 0부터 시작하는 숫자 대신 datetime이라는 컬럼을 인덱스로 사용하겠다

- filter 메서드

df.filter(like=’00:00:00’, axis=0) ← axis=0행, axis=1열로 행 열 모두 필터링 가능

df.filter(items=[’humidity’,’windspeed’]) ← 포함하는 행/열 필터링

df.filter(regex=’in.s’) ← 정규표현식 필터링 가능 (정규표현식은 몰라도됨)

- rename 메서드

df.rename({변경전:변경후}), axis=1)

- value_counts 메서드

df[’Review’].value_counts()

Review

Very good 117

Good          116

Fabulous    30

Superb        16

Superb9.0    8

cf) Superb9.0을 Superb에 합치기

df.loc[df[’Review’] == ‘Superb 9.0’, ‘Review’] = ‘Superb’ ← Review

- unique 메서드

df[’Total_Review’].unique() ← 모든 고유값들 반환


cf) 모든 total review를 숫자로 바꾸기

df[’Total_Review’] = df[’Total_Review’].map(lambda x : str(x).replace(’external’, ‘’).strip())

df[’Total_Review’] = df[’Total_Review’].map(lambda x : str(x).replace(’review’, ‘’).strip())

df[’Total_Review’] = df[’Total_Review’].map(lambda x : str(x).replace(’,’, ‘’))

df[’Total_Review’].astype(’float’)

str(x) ← 문자로 바꾸기

strip() ← 공백 제거

- dropna 메서드

df = df.dropna(subset = ‘Review’, axis=0) ←Review 열에서 결측치 삭제

- quantile 메서드 + for문

quantile = [0, 0.2, 0.4, 0.6, 0.8, 1]

for i in quantile:

q = df[’Total_Review’].quantile(i)

print(f’quantile({i}) is {q}’)

- query 메서드

★query는 열이름 “” 로 묶지 않고 전체조건을 ‘’로 묶음★

문자열로 행 필터링

df.query(’bill_length_nm > 55’)

df.query(’bill_length_nm > 55 and species == “Gentoo”’)

(@를 사용하여 외부변수 참조 가능)

length = 55

species = ‘Gentoo’

df.query(’bill_length_nm >= @length and species == @species’)

df.query(’island.str.contains(”oe”), engine = ‘python’)

df.query(’species.str.endwith(”e”)’, engine = ‘python’)

filtering = [”Adelie”, “Chinstrap”]

df.query(’species.isin(@filtering)’, engine = ‘python’)

- groupby 메서드

#기본 사용법

df.groupby(’sex’)[’survived’’].mean()

sex

female 0.742038

male     0.188908

df.groupby([’sex’,’class’])[’survived’].mean()

sex          class

female    First              0.968085

                Second        0.921053

                Third             0.500000

male        First              0.387542

                 Second        0.159325

                 Third             0.135413

#변수 별 별도 통계 연산도 가능

df.groupby([’sex’,’class’])[’survived’].agg([’mean’,’count’])

df.groupby([’sex’,’class’])[[’survived’,’age’]].agg({’survived:’mean’, ‘age’:’man’})

#사용자 정의 함수도 사용 가능

def get_IQR(data):

3rd = data.quantile(.75)

1st = data.quantile(.25)

return ((3rd-1st)*1.5)

df.groupby([’sex’,’class’])[’age’].apply(get_IQR)

#결측치를 그룹별 평균값으로 대체

df.isna().sum() ← 결측치 갯수 확인

df.groupby(’species’)[[’bill_length_mm’, ‘bill_depth_mm’, ‘flipper_length_mm’, ‘body_mass_g’]].mean() ← 그룹별 평균값 확인

df.groupby(’species’)[[’bill_length_mm’, ‘bill_depth_mm’, ‘flipper_length_mm’, ‘body_mass_g’]].apply(lambda x : x.fillna(x.mean())) ← 결측치를 그룹별 평균값으로 대체

#인자로 열 외에 다른 형태 데이터 전달

    group  value

0          A           1

1          A           1

2          A           1

3          B           10

4          B           10

df.groupby([0,0,1,1,1])[’value’].sum() 

0      2

1       21

s = pd.series([False,False,True,True,True])

df.groupby(s)[’value’].sum()

False      2

True       21

- 시계열 데이터 추가학습

#시계열 데이터로 변환 + 해당 시계열 데이터를 인덱스로 활용

df[’Date’] = pd.to_datetime(df[’Date’])

df = df.set_index(’Date’)

#시계열 행 슬라이싱 가능

df[’1980-12-13’ : ‘1980-12-18’]

df[’2015-02’ : ‘2015-02’] ← 2월 데이터 전부 반환

#resample 메서드

df.resample(’7d).mean() ← 7일동안의 평균을 반환

B ← 영업일

d ← 모든일c

# 데이터 처리 실전 프로세스

## 데이터 첫 처리

1. head & info 메서드로 샘플을 눈으로 확인, 크기나 결측치 확인

df.head()

df.info()

1. 변수별로 알맞은 데이터타입으로 변경

df[’START_DATE’]_pd.to_datetime(df[’START_DATE’], errors=’coerce’)

★errors=’coerce’ → 결측치도 데이터 타입 변경하다

1. 데이터 특이점 확인(이상치, 결측치, 분포 등)
    - 데이터 고유값 확인
    
     df[’START_DATE’].unique()
    
    - 결측치 갯수 확인
    
    df[’START_DATE’].isna().sum()
    
    - 범주형 변수에서 고유값과 빈도수 확인 → value_counts
    
    df[’CATEGORY].value_counts()
    
    - 수치형 변수로 기초통계확인 → describe
    
    df[’MILES’].describe()
    
    - 이상치 삭제
    
    df = df.drop()
    
2. 목적에 맞는 파생변수 추가
    
    df[’DURATION’] = (df[’END_DATE’] - df[’START_DATE’]).dt.total_seconds() / 60 → 분단위로 표현
    
3. 그룹별로 통계값 확인
    
    df.groupby([’CATEGORY’. ‘PURPOSE’])[[’MILES’,’DURATION’]].agg([’mean’,’std’,’count’])
    
4. 변수 간 상관관계 파악
    
    corr() 메서드 사용, 시각화 사용
    
    df[[’MILES’,’DURATION’]].corr()
    

## 연속 데이터의 범주화

- 수치형 변수 범주화
    
    pd.cut(df[’Age’], bins=8) ← 8개 구간으로 쪼개기
    
    pd.cut(df[’Age’], bins=8).reset_index().groupby(’Age).size() ← Age별로 8구간으로 쪼갠거 갯수구하기
    
- 특정 수치별로 데이터 범주화
    
    bins = [10, 20, 30, 40 ,50 ,60, 70, 80]
    
    pd.cut(df[’Age’], bins=bins)
    
- qcut 함수
    
    cut함수 : 동일한 길이의 구간
    
    qcut함수 : 포함되는 데이터의 갯수를 동일하게 나눌 때(중복 데이터가 많을 경우 동일하지 않아질 수 있음)
    
    pd.qcut(df[’Age’], q=8) ← 데이터를 8개씩 구간으로 나눔
    
    ★★df_quiz['Rating_Group'] = pd.qcut(df_quiz['Rating'], q=2, labels=['Low','High'])★★
    
    ★★labels= 로 이름붙이기 가능★★
    
    pd.qcut(df[’Age’], q=8).reset_index().groupby(’Age’).size()
    
    ## 조건을 만족하는 최대 연속 횟수 구하기
    
    - 1이 몇번 연속해서 나타나는지 구하기
    
    s = pd.Series([0,0,1,1,0,1,1,1,1,0])
    
    0 0
    
    1 0
    
    2 1
    
    3 1
    
    4 0 
    
    5 1 
    
    6 1
    
    7 1
    
    8 1
    
    9 0
    
    sc = s.cumsum() ← cumsum : 누적합 구하기
    
    0 0 
    
    1 0
    
    2 1
    
    3 2
    
    4 2
    
    5 3
    
    6 4
    
    7 5
    
    8 6
    
    9 6
    
    s.mul(sc) ← mul : 곱하기 (s * sc) 
    
    0 0
    
    1 0
    
    2 1
    
    3 2
    
    4 0 
    
    5 3
    
    6 4
    
    7 5
    
    8 6
    
    9 0
    
    s.mul(sc).diff() ← diff : f(n) - f(n-1)
    
    0 nan
    
    1 0
    
    2 1
    
    3 1
    
    4 -2
    
    5 3
    
    6 1
    
    7 1
    
    8 1
    
    9 -6
    
    s.mul(sc).diff().where(lambda x: x<0) ← where(조건식, f) : 조건만족하는건 그대로, 아닌것은 f값으로, f값 적용 안했을시 결측치로 대체됨
    
    0 nan
    
    1 nan
    
    2 nan
    
    3 nan
    
    4 -2
    
    5 nan
    
    6 nan
    
    7 nan
    
    8 nan
    
    9 -6
    
    s.mul(sc).diff().where(lambda x: x<0).ffilll() ← ffill 앞쪽 결측치가 아닌 값을 뒤쪽 결측치에 전파
    
    0 nan
    
    1 nan
    
    2 nan
    
    3 nan
    
    4 -2
    
    5 -2
    
    6 -2
    
    7 -2
    
    8 -2
    
    9 -6
    
    s.mul(sc).diff().where(lambda x: x<0).ffilll().add(sc, fill_value=0) ← add : 더하기 (왼쪽 식 + sc), fill_value=0 : 결측치는 0으로 더하기
    
    0 0
    
    1 0
    
    2 1
    
    3 2
    
    4 0
    
    5 1
    
    6 2
    
    7 3
    
    8 4
    
    9 0
    
    - 애플의 주식 종가 기준으로 175불 이상이었던 날짜 중 가장 긴 날 구하기
        
        s = df[’Close’] > 175
        
        s.sum() → 22
        
        sc = s.cumsum()
        
        s.mul(sc).diff().where(lambda x : x<0).ffilll().add(sc, fill_value=0).max() → 9
        

## 이상치를 다루는 방법

- describe, 시각화를 통해 데이터 내 이상치 유무 판단

df_new = df.query(’Weigh < 350’)      ← 350넘는수치 이상치라고 판단했을경우

df_new.describe()

criteria = df[’Weigh’].quantile(0.9999) ← 0.1%를 이상치라고 판단했을 경우

df_new = df[df[’Weigh’] < criteria]

- clip 메서드

최대, 최소 값보다 크거나 작은 값을 최대,최소로 치환

df[’Weigh’’] = df[’Weigh’].clip(50, 300)

## 결측치 내삽/외삽하기

- interpolate 메서드

결측치를 주변의 값을 감안하여 보간 (method 방법에대한 풀이는 심화과정)

s.interpolate(method =”spline”, order=1, limit_direction=”foward”, limit=2)

## 정렬된 인덱스에서의 행 슬라이싱

- 날짜를 인덱스로 활용

df[’Date’] = pd.to_datetime(df[’Date’])

df = df.set_index(’Date’).sort_values(’Date’)

cf) df[’2020-01’ : ‘2020-02’] 로 슬라이싱 가능

- 문자열 인덱스 슬라이싱

df = df.reset_index().set_index(’Province/State’).sort_index()

df.index.unique() ← 인자 확인

cf) 사전식 슬라이싱 가능 df[’Ca’:’Df’] ← Delaware 는 Df보다 이전값이라서 슬라이싱됨, Df로시작되는 도시는 슬라이싱안됨(Df후임)

## Timestamp 데이터를 포함한 여러 열을 기준으로 groupby 하기

df[’Date] = pd.to_datetime(df[’Date’])

df = df.set_index(’Date’).sort_values(’Date’)

df[’Province/State’].nunique() ← 고유값 갯수 확인

states = df[’Province/State’].unique()[0:3] ← 앞에서부터 top3 열 선택

df = df[df[’Province/State’].isin(states)] ← states만 필터링해서 보기

→ 날짜/시간 형식을 포함한 두 개 이상 변수를 기준으로 groupby 하기

df.groupby([pd.Grouper(freq=’6m’), ‘Province/State’])[’Case’].mean()

pd.Grouper(freq=’6m’) ← 6개월 단위로 그루핑

## 데이터셋 내 특정 그룹별 데이터 표준화

- inspection_step 변수의 값별로 표준화 진행

df[’normalized1’] = df.groupby(’inspection_step’)[’value’].transform(lambda x : (x-x.mean() / x.std())

★★transform → 계산값들을 각 행에 뿌려줌★★ (평균값을 1행 2행 3행 에 고대로 다 뿌려줌)

- inspection_step 변수의 가장 첫 번째 값으로 표준화 진행

temp = df.sort_values([’inspection_step’, ‘date’]).drop_duplicates(’inspection_step’) → drop_duplicates 중복행 제거

temp = temp.set_index(’inspection_step’)[’value’] 

df = df.set_index(’inspection_step’)

df[’normalized2’] = df[’value’] - temp

df = df.reset_index()

## groupby 메서드를 이용한 문자열 연산

- 문자열의 더하기(이어쓰기)를 이용한 groupby 연산

df[’path’] = df.groupby(’product_id’)[’operator’].transform(lambda x : ‘_’.join(x))


df[’path’] = df[’factory’] + ‘_’ + df[’path’]

df = df.drop_duplicates(’product_id’)

df = df[[’date,’product_id’,’passfail’,’path’]]

- path 별로 pass/fail의 value_counts 실행

df.groupby(’passfail’)[’path’].value_counts()

- date 별로 pass/fail의 value_counts 실행

df.groupby([’passfail’])[’date’].value_counts()

## 하나의 행을 여러개의 행으로 쪼개기

- explode 메서드
    - 공장명을 다른 열로 분리
        
        df[’factory’] = df[’path’].map(lambda x : x[0:2])
        
        df[’path’] = df[’path’].map(lambda x : x[3:])
        
    - split method를 통해 _를 기준으로 리스트 생성
        
        df[’path’].map(lambda x : x.split(’_’)) 
        
    - explode 메서드 사용 ← 리스트를 행으로 쪼갬
        
        df = df.explode(’path’)
        
        process_map = { ‘1’ : ‘P1’, ‘2’ : ‘P1’, ‘W’ : ‘P2’ , ‘V’ : ‘P2’, ‘X’ : ‘P3’, ‘Y’ : ‘P3’}
        
        df[’process’] = df[’path’].map(process_map)
  
