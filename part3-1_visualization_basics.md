# 파이썬 데이터 시각화 기초

## 그래프의 구성 요소 및 특징

- Figure : 여러 개의 그래프들이 들어갈 수 있는 공간
- axes : figure에 그래프를 그리기 위한 컷
    
## Scatterplot

★★mat : data=df, ax=ax★★

★★plotly : data_frame = df, width=400, heigh=400★★

### matplotlib, seaborn

- Figure와 ax 생성
    
    import matplotlib.pyplot as plt
    
    import seaborn as sns
    
    import pandas as pd
    
    fig, ax = plt.subplots()
    
- sactterplot 그리기
    
    sns.scatterplot(x=’Year’, y=’No. of Internet Users’, data=df, ax=ax)
    
- hue 인자  → scatter 색 나눠서 그래프 그리기
    - palette 인자 → 색 조절 가능
    
    df.Entity.unique() ← 고유값 확인
    
    entities = [’China’, ‘India’, ‘Finland’] ← 3가지만 선택
    
    df = df.query(’Entity.isin(@entities)’)
    
    fig, ax = plt.sublots()
    
    sns.scatterplot(x=’Year’, y=’No of Internet Users’, dtat=df, ax=ax, hue=’Entity’, palette=’bright’)
    
- hue_order 인자 활용 → lengend(범례)의 순서 조절 가능
    
    sns.scatterplot(x=’Year’, y=’No of Internet Users’, data=df, ax=ax, hue=’Entity’, palette=’bright’, hue_order=[’India’, ‘Finland’, ‘China])
    
- style 인자 → 색 대신 scatter의 형태를 구분하여 변수 데이터 구분
    - makers 인자 → scatter의 형태 지정 가능
    - s인자 → scatter의 크기 조절 가능
    
    sns.scatterplot(x=’Year’, y=’No of Internet Users’, data=df, ax=ax, style=’Entity’, markers=[’o’,’^’,’X’], s=100)
    
- size → 데이터 그룹별로 size 구분
    - sizes → size의 최솟값, 최댓값 지정 가능
    
    sns.scatterplot(x=’Year’, y=’No of Internet Users’, data=df, ax=ax, size=’Entity’, sizes=(40, 200)
    
- hue, style, size 혼합사용 가능
    
    df = sns.load_dataset(’tips’)
    
    fig, ax = plt.subplots()
    
    sns.scatterplot(x=’total_bill’, y=’tip’, data=df, ax=ax, hue=’smoker’, style=’time’, size=’size’)
    

### Plotly

- Plotly express Scatterplot
    
    import [plotly.express](http://plotly.express) as px
    
    import pandas as pd
    
    fig = px.scatter(data_frame=df, x=’Year’, y=’No of Internet Users’, width =400, height=400)
    
    fig.show()

- color 인자 ← 색 나눠줌 (=seaborn의 hue)
    - color_descrete_map, color_descrete_sequence 를 통해 색 조절 가능
    
    entities = [’China’, ‘India’, ‘Finland’]
    
    df = df.loc[df[’Entity’].isin(entities)]
    
    fig = px.scatter(data_frame=df, x=’Year’, y=’No of Internet Users’, width=400, height=400, color=’Entity’, coplor_discrete_sequence=[’blue’,’black’,’red’])
    
    아니면 color_dicrete_map = {’China’ : ‘blue’, ‘Finland’ : ‘black’, ‘India’ : ‘red’}
    
    fig.show()
    
- symbol 인자 ← 모양으로 나눠줌 (=seaborn의 style)
    - symbol_sequence 인자를 활용하여 scatter의 형태 커스터마이징 가능
    
    fig = px.scatter(data_frame = df, x=’Year’, y=’No of Internet Users’, width = 400, height = 400, symbol=’Entity’, symbol_sequence=[’star’,’arrow’,’cross’])
    
- size인자 ← y값 따라서 크기 조절 가능 (seaborn과 사용법 약간 다름)
    
    fig = px.scatter(data_frame = df, x=’Year’, y=’No of Internet Users’, width = 400, height = 400, size = ‘No of Internet Users’)
    
- color, symbol, size 인자 혼합해서 사용 가능
    
    fig = px.scatter(data_frame = df, x=’total_bill’, y=’tips’, color=’smoker’, size=’size’, symbol=’time’, width=600, height=400)


## Regression Plot

### matplotlip, seaborn

- ci : confidence interval ← 95% 신뢰구간
    - ci=None 으로 없앨 수 있음
- hue인자 사용 불가능
    - scatter 인자에 False를 전달하여 regression 그래프만 표현 가능

df = sns.load_dataset(’tips’)

fig, ax= plt.subplots()

sns.regplot(x=’total_bill’, y=’tip’, data=df, ax=ax)

- order 인자 ← regression 다항식 차수 조절 가능
    
    import numpy as np
    
    x = np.arrange(0, 10, 1)
    
    y = x**3 - 9*x**2 + x + 4
    
    fig, ax = plt.subplots()
    
    sns.regplot(x=x, y=y, ax=ax, order=3) ← 3차식 그래프 생성 가능
    
- scatter_kws ← 사이즈 조절 가능(s → size)
- line_kws ←색 과 선 모양 조절 가능
    
    sns.regplot(x=x, y=y, ax=ax, order=3, scatter_kws={’s’ : 80}, line_kws={’color’:’red’ , ‘linestyle’:’- -’})

### plotly

- Plotly express의 scatter 함수에서 trendline 인자로 활용 가능 trendline=’ols’
- color인자 사용시 color 별로 회귀선 도출 가능(Seaborn 에서는 불가능)
- trendline_scope ← 전체에 대한 regression 가능 trendline_scope=’overall’
    
    df = sns.load_dataset(’tips’)
    
    fig = px.scatter(data_frame=df, x=’total_bill’, y=’tip’, width=400, height=400, color=’smokers’, trendline=’ols’)
    

## Line plot

### matplotlib & Seaborn

- scatter 대신 line으로 표현
- scatterplot과 동일하게 사용
- hue, style 사용 가능
    
    fig, ax = plt.subplots()
    
    sns.lineplot(x=’Year’, y=’No of Internet Users’, data=df, ax=ax, hue=’Entity’)
    
- scatterplot과 겹쳐 그리기 가능
    
    fig, ax = plt.subplots()
    
    sns.lineplot(x=’Year’, y=’No of Internet Users’, data=df, ax=ax, hue=’Entity’)
    
    sns.scatterplot(x=’Year’ y=’No of Internet Users’, data=df, ax=ax, hue=’Entity’, legend=False)
    

### Plotly

- Plotly scatter 과 동일하게 그림
- color 인자 사용 가능
- symbol 대신 line_dash 인자 사용
    
    fig = px.line(data_frame=df, x=’Year’, y=’No of Internet Users’, width=400, height=400, color=’Entity’, line_dash=’Entity’)
    
- symbol 인자 추가하여 scatter 동시에 표현 가능
    
    fig = px.line(data_frame=df, x=’Year’, y=’No of Internet Users’, width=400, height=400, color=’Entity’, line_dash=’Entity’, symbol=’Entity’)
    

## Boxplot

### Matplotlib & Seaborn


- IQR = Q3 - Q1
- 범위 : Q3 + 1.5IQR ~ Q1 - 1.5IQR
- scatterplot 과 유사한 방식으로 그림
    
    fig, ax = plt.subplots()
    
    sns.boxplot(x=’weekday’, y=’kwhTotal’, data=df, ax=ax)
    
- stripplot, swarmplot
    
    boxplot으로는 확인할 수 없는 통계량 표현해줌
    
    stipplot 의 그림 겹침을 분산시켜 놓은게 swarmplot 
    
    sns.stipplot(x=’weekday’, y=’kwhTotal’, data=df, ax=ax, color=’grey’, alpha=0.4) → alpha : 투명도
    
    sns. boxplot(x=’weekday’, y=kwhTotal’, data=df, ax=ax)
    
    sns.swarmplot(x=’weekday’, y=’kwhTotal’, data=df, ax=ax, color=’grey’, alpha=0.4) → alpha : 투명도
    
    sns. boxplot(x=’weekday’, y=kwhTotal’, data=df, ax=ax)
    

- order 메서드 ← boxplot 순서 정렬 가능 cf) stipplot, swarmplot 적용시 똑같이 order 적용해야함
    
    weekday_order = [’Mon’, ‘Tue’, ‘Wed’, ‘Thr’, ‘Fri’, ‘Sat’, ‘Sun’] ← 리스트형태
    
    sns.boxplot(x=’weekday’, y=’kwhTotal’, data=df, ax=ax, order=weekday_order)
    
- hue 메서드 사용 가능

### Plotly

- scatter와 유사
    
    fig = px.box(data_frame=df, x=’weekday’, y=’kwhTotal’, width=500, height=400)
    
- points 인자를 통해 boxplot+stripplot 표현 가능
    
    fig = px.box(data_frame=df, x=’weekday’, y=’kwhTotal’, width=500, height=400, points=’all’)
    
- category_orders 인자를 통해 x축 변수 순서 조절 가능
    
    weekday_order = {’weekday’ : [’Mon’, ‘Tue’, ‘Wed’, ‘Thr’, ‘Fri’, ‘Sat’, ‘Sun’]} ← 딕셔너리 형태
    
    fig = px.box(data_frame=df, x=’weekday’, y=’kwhTotal’, width=500, height=400, category_orders=weekday_order)
    
- color 인자를 통해 그룹별로 box나눠표현 가능
    
    fig = px.box(data_frame=df, x=’weekday’, y=’kwhTotal’, width=500, height=400, color=’platform’)
    

## Histogram

### Matplotlib & Seaborn

- 도수분포표를 시각화
- 수치형 변수의 분포를 표현
- bins 인자 ← 막대의 수 조절
    
    fig, ax = plt.subplots()
    
    sns.histplot(df,  x=’total_bill’, ax=ax, bins=30)
    
- hue 인자 ← 그룹별로 색 나누기 가능
- stack 인자 ← 막대를 누적하여 표현 가능 multiple=’stack’
    
    sns.histplot(df,  x=’total_bill’, ax=ax, bins=30, hue=’time’, multiple=’stack’)
    

### Plotly

- nbins 인자 ← 표현할 막대의 수 조절
    
    fig = px.histogram(data_frame=df, x=’total_bill’, width=400, nbins=20)
    
    fig.show()
    
- color ← 그룹별 색 표현 가능
- barmode ← 막대를 겹쳐/누적하여 표현 가능 barmode = ‘overly’, barmode = ‘relative’
    
    fig = px.histogram(data_frame=df, x=’total_bill’, width=400, nbins=20, color=’time’, barmode=’overly’)
    
    fig.show()
    

## Heatmap

### Matplotlib & Seaborn

- Pivot table : 2개 이상의 변수를 각각 index/column으로 지정, 또 다른 변수의 통계량을 각 value로 변환
- Heatmap : 각각의 값들에 대해 시각화 하는 방법
- annot ← 각 셀의 값 표현 가능
    
    fig, ax = plt.subplots()
    
    sns.heatmap(pivot_df, ax=ax, annot=True)

- fmt ← 각 셀에 표현되는 수치 값의 형식 변경 가능
    - fmt=’.2e’ ←소숫점아래 두자리 까지 표현 f←실수값, e←10의자승
- vmax, vmin ← color bar의 범위 설정 가능
    
    vmax=16000, vmin=0
    
- cmap ← 여러 종류의 color map 사용 가능
    1. cmap = ‘RdBu’
    2. color = sns.light_palette(’seagreen’, as_cmap=True)
        
        cmap = color
        

### Plotly

- x값과 y값을 따로 전달해줘야함
- text_auto ← 각 셀에 표현되는 수치 값 형식 변경가능
    
    fig = px.imshow(pivot_df, x=pivot_df.columns, y=pivot_df.index.astype(’str’), text_auto=’.2e’, width=400, height=400)
    
    fig.show()
    
- color_continuous_scale ← 여러 종류의 color map 사용 가능
    
    color_continuous_sclae = ‘RdBu’
    

## Seaborn의 axes-level plot과 figure-level plot

- figure-level plot
    - figure 단위로 지정하여 그리는 그래프
    - ax를 지정하여 그릴 수 있음
    - 특정 column 기준으로 groupby 가능 (column, row 나누기 가능)
    - Implot, relplot 등
- axes-level plot
    - ax 단위로 지정하여 그리는 그래프
    - ax를 지정하여 그릴 수 있음
    - 특정 column을 기준으로 groupby 불가능
    - scatterplot, boxplot, heatmap 등
- fig, ax = plt.subplots(2, 2, figsize=(12,12)) ← 2행 2열짜리 fig, 사이즈는 12,12
    

- axes-level plot을 특정 변수의 그룹별로 subplot으로 나눠 그리기

figm ax = plt.subplots(2, 2, figsize=(12,12))

sns.regplot(x=’bmi’, y=’charges’, data=df.query(’region == “southwest”’), ax=ax[0][0])

sns.regplot(x=’bmi’, y=’charges’, data=df.query(’region == “southeast”’), ax=ax[0][1])

sns.regplot(x=’bmi’, y=’charges’, data=df.query(’region == “northwest”’), ax=ax[1][0])

sns.regplot(x=’bmi’, y=’charges’, data=df.query(’region == “northeast”’), ax=ax[1][1])

- figure-level plot을 특정 변수의 그룹별로 subplot으로 나눠 그리기

sns.Implot(x=’bmi’, y=’charges’, data=df, col=’region’, col_wrap=2, sharex=False, sharey=False)

col_wrap ← 2개씩 끊어서 밑으로 내리기

sharex, sharey ← x축,y축 통일?


- Implot에서는 regplot에서 사용 불가능하던 hue 인자 사용 가능

sns.Implot(x=’bmi’, y=’charges’, data=df, col=’smoker’, row=’region’, hue=’sex’, sharex=False, sharey=False)

- axes-level plot을 figure-level plot 처럼 그리기
    - facetgrid를 이용하여 axes-level plot을 figure-level plot처럼 그리기 가능
    - axes-level plot의 특정 변수의 그룹별 column, row나눠 그리기 가능
    - facetGrid(데이터, column으로 나눌 변수, row로 나눌 변수)
        
        +map_dataframe(axes-level plot, plot의 인자들)형식 으로 사용
        

g = sns.FacetGrid(data=df, col=’region’, col_wrap=2, sharex=False, sharey=False)

g.map_dataframe(sns.boxplot, x=’smoker’, y=’charges’, hue=’sex’)


- Plotly express 함수들의 facet 사용
    - Plotly express 대부분의 함수는 facet을 나눠그릴 수 있음
    - facet_col, facet_row 인자를 이용하여 Seaborn의 figure-level plot 처럼 활용

fig = px.scatter(data_frame=df, x=’bmi’, y=’charges’, color=’sex’, facet_row=’region’, facet_col=’smoker’, width=700, height=1200, trendline=’ols’)

fig.show()

fig = px.box(data_frame=df, x=’smoker’, y=’charges’, facet_col=’region’, facet_col_wrap=2, color=’sex’, width=700, height=700)

fig.show()

## 그래프 세부 요소 튜닝하기

- Matplotlib tick label 회전, 글씨조절
    - tick label 회전 → tick_params
        
        ax.tick_params(axis = ‘x’, labelrotation=90) ← x축 tick label을 90도회전 
        
- Plotly tick label 회전, 글씨조절
    - tick label 회전
        
        fig.update_xaxes(tickfont={’size’:16}, tickangle=90)
        
- Matplotlib 그래프 제목 입력
    - 각 ax에서 set_title 메서드 사용
        
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        
        sns.boxplot(x=’company_size’, y=’salary_in_usd’, data=df, ax=ax[0], order=[’S’,’M’,’L’])
        
        ax[0].set_title(’company_size box plot’, fontsize=16)
        
        sns.histplot(x=’salary_in_usd’, data=df, ax=ax[1])
        
        ax[1].set_title(’salary histogram’, fontsize=16)
        
    - 전체 fig에 제목 넣기
        
        plt.suptitle(’This is suptitle’)
        
- Plotly 그래프 제목 입력
    - 그래프 함수 내 title 인자 사용
    - HTML요소 사용하여 굵기, 이탤릭체 등 수정 가능
    - figure의 update_layout 함수 사용하여 글꼴, 크기 등 세부사항 수정 가능
    
    fig = px.box(data_frame=df, x=’company_size’, y=’salary_in_usd’, width=400, height=400, title=’<b>company_size box plot</b>’, category_orders={’company_size’:[’S’,’M’,’L’]})
    
    fig.update_layout({’title_font_size’ : 20})
    
    fig.show()
    
- Grid 표시(눈금선)
    - Matplotlib에서 ax의 grid메서드 사용
        
        ax.grid(axis=’y’)  axis=’x’, ‘y’, ‘both’다 가능
        
    - Plotly ← default가 grid 표시
        
        grid 제거하기 → fig.update_layout(yaxis={’showgrid’ : False})
        
- Figure 내 각 ax 세부 위치 조절
    - Matplotlib → tight_layout 메스드를 사용하여 subplot간 간격 조절
        
        fig, ax = plt.subplots(2, 2)
        
        for idx in range(2):
        
        for jdx in range(2):
        
        ax[idx][jdx].set_xlabel(’x_label’)
        
        ax[idx][jdx].set_ylabel(’y_label’)
        
        ax[idx][jdx].set_title(’title’)
        
        plt.tight_layout()

        plt.tight_layout(w_pad=5, h_pad=1, pad=4)
 
        plt.tight_layout(rect=(0.5, 0, 1, 1))

- Subplot 위치 조절
    - Plotly express 그래프 함수의 facet_col_spacing, facet_row_spacing 메서드 사용
        
        fig = px.scatter(df, x=’total_bill’, y=’tip’, facet_row=’sex’, facet_col = ‘time’, width=600, height=600, facet_row_spacing=0.2, facet_col_spacing=0.1)
        
        fig.show()
