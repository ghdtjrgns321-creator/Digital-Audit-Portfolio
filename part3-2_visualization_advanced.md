# 파이썬 데이터 시각화 심화

# 시계열 그래프 Tick Label 깔끔하게 표현하기

## x축이 날짜인 시계열 그래프(Matplotlib)

- 데이터 타입 확인 후 datetime으로 변경 필요. object 타입인 경우 알아볼 수 없게 나옴

df[’Date’] = pd.to_datetime(df[’Date’])

fig, ax = plt.subplots()

sns.lineplot(x=’Date’, y=’Close’, data=df, ax=ax)

- x축 60도정도 회전시키기

fig, ax = plt.subplots()

sns.lineplot(x=’Date’, y=’Close’, ax=ax, data=df)

ax.tick_paramas(axis=’x’, labelrotation=60)

![image.png](image.png)

- 시계열 그래프 x축 깔끔하게 표시하기 위한 코드

import matplotlib as mpl

fig, ax = plt.subplots()

sns.lineplot(x=’Date’, y=’Close’, data=df, ax=ax)

ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

![image.png](image%201.png)

## x축이 날짜인 시계열 그래프(Plotly)

- Plotly의 경우 날짜/시간 관련 자동으로 파싱해줌
- figure의 update_xaxes 메서드의 tickformat 인자를 통해 날짜/시간 형식 지정 가능

fig = px.line(df, x=’Date’, y=’Close’, width=500, height=400)

fig.show()

![image.png](image%202.png)

fig = px.line(df, x=’Date’, y=’Close’, width=500, height=400)

fig.update_xaxes(tickformat=’%Y-%m-%d’)

fig.show()

![image.png](image%203.png)

# 다중 축 그래프 그리기

## 2개의 y축을 가지는 그래프 그리기(Matplotlib)

fig, ax = plt.subplots()

ax2 = ax.twinx()

sns.lineplot(x=’Date’, y=’Close’, data=df, ax=ax, color=’red’)

sns.lineplot(x=’Date’, y=’volume’, data=df, ax=ax, color=’blue’)

ax.tick_params(axis=’y’, labelcolor=’red’)

ax.yaxis.label.set_color(’red’)

ax2.tick_params(axis=’y, labelcolor=’blue’)

ax2.yaxis.label.se_color(’blue’)

![image.png](image%204.png)

## 3개의 y축을 가지는 그래프 그리기(Matplotlib)

df[’High-Low’] = df[’High’]-df[’Low’]

fig, ax = plt.subplots()

fig.subplots_adjust(right=0.75) ← 그래프 우측을 15% 줄이겠다 (3번째y축표현할 공간 남기려고)

ax2 = ax.twinx()

ax3 = ax.twinx()

ax3.spines.right.set_position((”axes”, 1.2)) ← 3번째 y축을 그래프 바깥으로(1.2만큼) 옮김

sns.lineplot(x=’Date’, y=’Close’, data=df, ax=ax, color=’red’)

sns.lineplot(x=’Date’, y=’Volume’, data=df, ax=ax, color=’blue’)

sns.lilneplot(x=’Date’, y=’Volume’, data=df, ax=ax, color=’green’)

ax.yaxis.label.set_color=(’red’) ← label(’Close’) 색 변경

ax2.yaxis.label.set_color=(’blue’)

ax3.yaxis.labe.set_color=(’green’)

ax.tick_params(axis=’y’, labelcolor=’red’) ← y축 params 변경( 220, 200, 180 … )

ax2.tick_params(axis=’y’, labelcolor=’blue’)

ax3.tick_params(axis=’y’, labelcolor=’green’)

![image.png](image%205.png)

## 2개의 y축을 가지는 그래프 그리기(Plotly)

from ploylt.sublots import make_subplots

fig = make_subplots(specs=[[{”secondary_y”: True}]]

subfig1 = px.line(df, x=’Date’, y=’Close’)

subfig1.update_traces(line_color=’red’)

subfig2 = px.line(df, x=’Date’, y=’Volume’)

subfig2.update_traces(line_color=’blue)

subfig2.update_traces(yaxis=’y2’)

fig.add_traces(subfig1.data + subfig2.data)

fig.layout.xaxis.title = ‘Date’

fig.layout.yaxis.title = ‘Close’

fig.layout.yaxis2.title = ‘Volume’

fig.layout.yaxis.color = ‘red’

fig.layout.yaxis2.color = ‘blue’

fig.update_layout(width=500, height=400)

fig.show()

![image.png](image%206.png)

## 3개의 y축을 가지는 그래프 그리기(Plotly) (심화)

import plotly.graph_objects as go 

fig=make_subplots()

fig.add_trace(

go.Scatter(

x=df[’Date’], y=df[’Close’], name=’Close’, mode=’lines’, yaxis=’y’,

 line{’color’ : ‘red’},

)

)

fig.add_trace(

go.Scatter(

x=df[’Date’], y=df[’Volume’], name=’Volume’, mode=’lines’, yaxis=’y2’,

 line{’color’ : ‘blue’},

)

)

fig.add_trace(

go.Scatter(

x=df[’Date’], y=df[’High-Low’], name=’High-Low’, mode=’lines’, yaxis=’y3’,

 line{’color’ : ‘green’},

)

)

fig.update_layout(

yaxis = dict(title = “Close”),

yaxis2 = dict(

position = 1, title = “Volume”,

side = “right”, anchor = “free”, overlaying= “y”

           ),

yaxis3 = dict(

title = “High-Low”, side= “right”, anchor =“x”

     ),

xaixs = dict(title = “Date”, domain = [.1, .85]),

width=600, height=400

)

fig.layout.yaxis.color=’red’

fig.layout.yaxis2.color=’blue’

fig.layout.yaxis3.color=’green’

fig.show()

![image.png](image%207.png)

# 범례(legend) 위치 조절하기

## Matplotlib , Seaborn

- ax의 legend 메서드를 통해 범례의 위치 설정 가능

fig, ax = plt.subplots()

sns.scatterplot(x=’Fuel Consumption Comb (L/100 km)’, y=’CO2 Emissions(g/km)’, data=df, ax=ax, hue=’Vehicle Class’, palette=’bright’)

ax.legend(bbox_to_anchor=(1.01, 1.05)) ← 범례 위치를 x축 1.01만큼 더가고 y축 1.05만큼 더가라

![image.png](image%208.png)

## Plotly

- figure의 update_layout 메서드를 통해 범례 위치 조절 가능

fig = px.scatter(df, x=’Fuel Consumption Comb (L/100 km)’, y=’CO2 Emissions(g/km)’, color=’Vehicle Class’, width=700, height=500)

fig.update_layout(legend_x=1.2, legend_y=1)

fig.show()

![image.png](image%209.png)

# 그래프의 테두리(spine) 강조하기

## Matplotlib, Seaborn

- 여러 개의 subplot을 가지는 figure에서 특정 그래프만 강조할 때 유용

fig, ax = plt.subplots()

sns.boxplot(x=’Cylinders’, y=’CO2 Emissions(g/km)’, data=df, ax=ax)

spines = [’left’, ‘right’, ‘top’, ‘bottom’]

for spine in spines:

ax.spines[spine].set_color(’blue’)

ax.spines[spine].set_linewidth(3)

![image.png](image%2010.png)

## Plotly

fig = px.box(df, x=’Cylinders’, y=’CO2 Emissions(g/km)’, width=500, height=400)

fig.update_xaxes(showline=True, linecolor=’black’, linewidth=3, mirror=True)

fig.update_yaxes(showline=True, linecolor=’black’, linewidth=3, mirror=True)

fig.show()

![image.png](image%2011.png)

# 그래프 내 텍스트 표현

## Matplotlib, Seaborn

- text, annotation 메서드를 통해 표현 가능
- text(x좌표, y좌표, s=”문자열”) 형식으로 사용

fig, ax = plt.subplots()

sns.scatterplot(x=’Fuel Consumption Comb (L/100 km)’, y=’CO2 Emissions(g/km)’, data=df, ax=ax, hue=’Fuel Type’)

ax.text(x=10, y=130, s=’fuel type ethanol emits less CO2’, fonddict={’fontsize’:12, ‘weight’:’bold’})

cf) 상대좌표 입력 시 

ax.text(x=0.3, y=0.9, s=’fuel type ethanol emits less CO2’, fonddict={’fontsize’:12, ‘weight’:’bold’}, transform=ax.transAxes)

cf) annotate 메서드 사용 시

ax.annotate(text=’etanol is efficient’, xy=(0.95, 0.7), xytext=(0.73, 0.5), arrowprops{’color’: ‘black’, ‘width’:1}, xycoord=ax.transAxes) 

arrowprops → 화살표넣기, xy → 화살표가 가리키는 대상, xytext → 텍스트가 쓰이는 위치

![image.png](image%2012.png)

![image.png](image%2013.png)

## Plotly

- Plotly figure의 add_annotation 메서드를 이용해 표현
- add_annotation(x좌표, y좌표, 문자열) 형식으로 사용 가능
- 상대좌표 입력시 xref, yref인자에 x domain, y domain 전달 필요

fig = px.scatter(df, x=’Fuel Consumption Comb (L/100 km)’, y=’CO2 Emissions(g/km)’, width=500, height=400, color=’Fuel Type’)

fig.add_annotation(x=20, y=130, text=’<b>fuel type ethanol emits less CO2</b>’, showarrow=False)

fig.add_annotation(x-0.9, xref=’x domain’, y=0.75, yref=’y domain’, text=’ethanol is efficient’, showarrow=True, arrowhead=2)

arrowhead ← 화살표 모양

fig.show()

![image.png](image%2014.png)

# 수평선과 수직선 그리기

## Matplotlib, Seaborn

- axhline, axvline 메서드를 통해 수평선, 수직선 표현

fig, ax = plt.subplots()

sns.scatterplot(x=’date’, y=’value’, data=df, ax=ax)

ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(ax.xaxis.get_major_locator())) ← 시계열그래프에서 x축 깔끔하게

ax.axhiline(df[’lower_spec’].iloc[-1], color=’red’, linewidth=0.5)

ax.axhiline(df[’target’].iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(df[’upper_spec’].iloc[-1], color=’red’, linewidth=0.5)

![image.png](image%2015.png)

## Plotly

- add_hline, add_vline 메서드를 통해 수평선, 수직선 표현
- line_width, line_dash, line_color 인자를 통해 선 속성 변경 가능

fig = px.scatter(df, x=’date’, y=’value’, width=500, height=400)

fig.add_hline(df[’lower_spec’].iloc)[-1], line_color=’red’, line_width=0.5)

fig.add_hline(df[target’].iloc)[-1], line_color=’red’, line_width=0.5)

fig.add_hline(df[’upper_spec’].iloc)[-1], line_color=’red’, line_width=0.5)

![image.png](image%2016.png)

# FacetGrid로 나눈 각 ax별 mapping으로 그래프 세부튜닝하기

## Seaborn FacetGrid를 통해 그린 subplot들을 각 그룹별 통계값으로 튜닝하기

- 사용자 정의 함수를 통해 FacetGrid mapping으로 그룹별 subplot 튜닝 가능
- for 문을 통해 각 subplot(ax)별로 그룹별 튜닝 가능

g=sns.FacetGrid(df, sharex=False, sharey=False, col=’inspection_step’, aspect=1.6) ← aspect = 1.6 : 모양이 가로가 세로의 1.6배

g.map_dataframe(sns.scatterplot, x=’date’, y=’value’)

1. 함수로 처리하는 방법

def custom(lower_spec, target, upper_spec, ****kws): ←**kws 키워드를 가변인자로 받음

ax = plt.gca()

ax.axhline(lower_spec.iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(target.iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(upper_spec.iloc[-1], color=’red’, linewidth=0,5)

ax.xaxis.set_major_formatter(mp1.dates.ConciseDateFormatter(ax.xaxis.get_major_locator())) ← x축 이쁘게

g.map(custom, ‘lower_spec’, ‘target’, ‘upper_spec’)

![image.png](image%2017.png)

1. for 문으로 처리하는 방법

for ax in g.axes.flat:

inspection_step = ax.get_title()[-1]

temp_df = df.loc[df[’inspection_step’] == inspection_step]

ax.axhline(temp_df[’lower_spec’].iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(temp_df[’upperr_spec’].iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(temp_df[’target’].iloc[-1], color=’red’, linewidth=0.5)

ax.xaxis.set_major_formatter(mp1.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

## Seaborn FacetGrid를 통해 그린 subplot들을 각 그룹별 통계값으로 튜닝하기

- 사용자 정의 함수를 통해 FacetGrid mapping으로 그룹별 subplot 튜닝 가능
- for 문을 통해 각 subplot(ax)별로 그룹별 튜닝 가능

def custom(value, lower_spec, target, upper_spec, **kws):

ax = plt.gca()

ax.axhline(lower_spec.iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(target.iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(upper_spec.iloc[-1], color=’red’, linewidth=0,5)

ax.xaxis.set_major_formatter(mp1.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

mean = value.mean()

ax.axhline(mean, color=’blue’, linestyle = ‘- -’, linewidth=2)

#심화과정으로 각 subplot 별 데이터의 median 값 표시

trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

ax.text(x=0.02, y=mean, s=’mean: {:.1f}’.format(mean), fontdict={’fontsize’:12, ‘weight’:’bold’}, bbox={’facecolor’ : ‘white’}, transform=trans, ha=’left’) ← ha : 왼쪽정렬

g.map(custom, ‘value’, ‘lower_spec’, ‘target’, ‘upper_spec’)

![image.png](image%2018.png)

## Plotly facet을 통해 그린 subplot들을 각 그룹별 통계값으로 튜닝하기

- plotly figure layout의 annotation에 포함된 ‘text’를 활용하여 for 문으로 그룹별 튜닝

fig = px.scatter(df, x=’date’, y=’value’, facet_col=’inspection_step’, facet_col_spacing=0.05)

for idx in range(df[’inspection_step’].nunique()):

step = fig.layout.annotations[idx].text.split(’=’)[1]

fig.add_hline(

y=df.query(’inspection_step == @step’)[’lower_spec’].iloc[-1],

line_color=’red’, lind_width=0.5, row=1, col=idx+1)

fig.add_hline(

y=df.query(’inspection_step == @step’)[’upper_spec’].iloc[-1],

line_color=’red’, lind_width=0.5, row=1, col=idx+1)

fig.add_hline(

y=df.query(’inspection_step == @step’)[’target’].iloc[-1],

line_color=’red’, lind_width=0.5, row=1, col=idx+1)

fig.update_yaxes(matches=None)

fig.update_yaxes(showticklabels=True)

fig.show()

![image.png](image%2019.png)

  # 각 subplot 별 데이터의 median 값 표시

med = df.query(’inspection_step == @step’)[’value’].median()

fig.add_hline(y=med, line_color=’black’, line_width=3, line_dash=’dot’, row=1, col=idx_1)

fig.add_annotation(text=’median’: {:.1f}’.format(med), showarrow=False, bordercolor=’black’, borderwidth=1, bgcolor=’rgb(256,256,256)’, x=0.02, y=med, xref=’x domain’, row=1, col=idx+1,)

![image.png](image%2020.png)

# 특정 조건을 만족하는 subplot 강조하기

## FacetGrid에서 강조

### Seaborn FacetGrid에서 조건 만족시 테두리를 강조

- 특정 조건을 만족하는 여부를 나타내는 불리언 column 생성

df[’spec_out’] = (df[’value’] > df[’upper_spec’] | df[’value’] < df[’lower_spec’])

- 불리언 column 값을 받아 테두리를 강조하는 사용자 정의 함수 생성

def custom(value, lower_spec, target, upper_spec, **kws):

ax = plt.gca()

ax.axhline(lower_spec.iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(target.iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(upper_spec.iloc[-1], color=’red’, linewidth=0,5)

ax.xaxis.set_major_formatter(mp1.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

def if_spec_out(spec_out, **kws):

if spec_out.sum() > 0;

ax = plt.gca()

apines = [’left’, ‘bottom’]

for spine in spines:

ax.spines[spine].set_color(’blue’)

ax.spines[spine].set_linewidth(3)

g = sns.FacetFrid(df, sharex=Fasle, sharey=False, col=’inspection_step’, aspect=1.6)

g.map_dataframe(sns.scatterplot, x=’date’, y=’value’)

g.map(custom, ‘lower_spec’, ‘target’, ‘upper_spec’)

g.map(if_spec_out, ‘spec_out’)

![image.png](image%2021.png)

### Seaborn FacetGrid에서 조건 만족시 화살표 표시넣기

g = sns.FacetGrid(df, sharex=False, sharey=False, col=’inspection_step’, aspect=1.6)

g.map_dataframe(sns.scatterplot, x=’date’, y=’value’)

for ax in g.axes.flat:

title = ax.get_title()[-1] ← -1 : A, B, C 표시

temp_df = df.query(’inspection_step == @title’)

ax.axhline(temp_df[’lower_spec’].iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(temp_df[’upper_spec’].iloc[-1], color=’red’, linewidth=0.5)

ax.axhline(temp_df[’target_spec’].iloc[-1], color=’red’, linewidth=0.5)

ax.xaxis.set_major_formatter(mp1.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

spec_out_df = temp_df.query(’spec_out != 0’)

if len(spec_out_df)>0:

for idx in range(len(spec_out_df)):

ax.annotate(

xy=(spec_out_df.iloc[idx][’date’], spec_out_df.iloc[idx][’value’]), 

xytext=(spec_out_df.iloc[idx][’date’], spec_out_df.iloc[idx][’value’]*1.01),

text=’spec_out’, arrowprops={’color’:’red’,’width’:2}, color=’red’, weight=’bold’)

![image.png](image%2022.png)

## Plotly Facet에서 특정 조건을 만족하는 sublot 강조

fig = px.scatter(data_frame=df, x=’date’, y=’value’, facet_col=’inspection_step’, facet_col_spacing=0.05)

for idx in range(df[’inspection_step’].nunique()):

step = fig.layout.annotations[idx].text.split(’=’)[1]

fig.add_hline(

y=df.qiery(’inspection_stop == @step’)[’lower_spec’].iloc[-1],

line_color=’red’, line_width=0.5, row=1, col=idx+1)

fig.add_hline(

y=df.qiery(’inspection_stop == @step’)[’upper_spec’].iloc[-1],

line_color=’red’, line_width=0.5, row=1, col=idx+1)

fig.add_hline(

y=df.qiery(’inspection_stop == @step’)[’target’].iloc[-1],

line_color=’red’, line_width=0.5, row=1, col=idx+1)

if df.query(’inspection_step == @step’)[’spec_out’].sum() > 0:

fig.update_xaxes(showline=True, linecolor=’black’, linewidth=3, mirror=True, row=1, col=idx+1)

fig.update_yaxes(showline=True, linecolor=’black’, linewidth=3, mirror=True, row=1, col=idx+1)

fig.update_yaxes(maches=None, showticklabels=True)

fig.show()

![image.png](image%2023.png)

## Plotly 조건 만족시 화살표 표시넣기

fig = px.scatter(data_frame=df, x=’date’, y=’value’, facet_col=’inspection_step’, facet_col_spacing=0.05)

for idx in range(df[’inspection_step’].nunique()):

step = fig.layout.annotations[idx].text.split(’=’)[1]

fig.add_hline(

y=df.qiery(’inspection_stop == @step’)[’lower_spec’].iloc[-1],

line_color=’red’, line_width=0.5, row=1, col=idx+1)

fig.add_hline(

y=df.qiery(’inspection_stop == @step’)[’upper_spec’].iloc[-1],

line_color=’red’, line_width=0.5, row=1, col=idx+1)

fig.add_hline(

y=df.qiery(’inspection_stop == @step’)[’target’].iloc[-1],

line_color=’red’, line_width=0.5, row=1, col=idx+1)

spec_out_df = df.query(’inspection_step = @step and spec_out != 0’)

if len(spec_out_df) > 0 :

for jdx in range(len(spec_out_df)):

fig.add_annotation(

text=’spec out’,

x=spec_out_df.iloc[jdx][’date’],

y=spec_out_df.iloc[jdx][’value’],

row=1, col=idx+1, arrowcolor=’red’, font={’color’:’red’})

fig.update_yaxes(matches=None, showticklabels=True)

fig.show()

# 선형회귀선의 식과 상관계수 표시하기

## Seaborn regplot을 통해 그린 선형회귀 수식 표시

- scipy stats의 lineregress 함수를 이용해 회귀선의 기울기, y절편, 결정계수 계산(결측치 제거 해야함)
- ax의 text메서드를 이용하여 텍스트로 표시
- regplot의 line_kws 인자를 이용하여 범례에 표시
1. 

df = df.dropna()

s, oi, r, p, se = linregress(df[’horsepower’], df[’weight’])

fig, ax = plt.subplots()

sns.regplot(x=’horsepower’m y=’weight’, data=df, ax=ax, line_kws={’label’ : ‘y={:.2f}x+{:.2f}, R^2={:.2f}’.format(s, i, r**2)})

ax.legend()

![image.png](image%2024.png)

1. 

fig, ax= plt.subplots()

sns.regplot(x=’horsepower’, y=’weight’, data=df, ax=ax)

ax.text(x=0.05, y=0.9, s=’y={:.2f}x+{:.2f}, R^2={:.2f}’.format(s, i, r**2), transform=ax.transAxes)

![image.png](image%2025.png)

## Plotly 를 통해 그린 선형회귀 수식 표시

- plotly scatter의 trendline 인자를 통해 그린 회귀선의 정보를 plotly express get_trendline_results 함수로 불러옴

fig = px.scatter(df, x=’horsepower’, y=’weight’, width=500, height=400, trendline=’ols’)

results = px.get_trenline_results(fig)

results = results.iloc[0][”px_fit_results”]

print(results.summaty())

![image.png](image%2026.png)

![image.png](image%2027.png)

fig.add_annotation(text=’y= {:.1f}x + {:.1f}, R^2={:.2f}’.format(results.params[1], results.params[0], results.rsquared), x=0.05, y=0.95, xref=’x domain’, yref=’y domain’, showarrow=False)

fig.show()

![image.png](image%2028.png)

# 그래프의 축 log 형식으로 변환하기

## Matplotlib, Seaborn

- 데이터가 급격하게 증가하여 일반적인 형식으로 인사이트를 얻기 어려운 경우
- 데이터의 분포가 특정 방향으로 긴 꼬리를 가져 전체를 한 눈에 파악하기 어려운 경우
- set_ysacle 메서드를 사용하여 로그 형태로 변환

fig, ax = plt.subplots()

sns.lineplot(x=’date’, y=’confiremd’, data=df, ax=ax)

ax.xaxis.set_major_formatter(mp1.dates.ConciseDateFormatter(ax.xaxis.get_manjor_locator()))

ax.set_ysclae(’log’)

![image.png](image%2029.png)

## Ployly 에서 적용

fig = px.line(data_frame=df, x=’date’, y=’confirmed’, width=500, height=400, log_y=True)

fig.show()

![image.png](image%2030.png)

# Color Palette및 Color Map 활용

## Seaborn color palette

- 특정 변수 그룹별 색깔 구분 시 hue, color 인자
    - discrete 형식, continuous 형식 등 데이터 특징에 다른 적절한 palette 적용 필요
    - Seaborn 그래프 함수의 palette 인자를 통해 튜닝 가능
- palette → bright, color blind, pastel 등

fig, ax = plt.subplots()

sns.scatterplot(x=’total_bill’, y=’tip’, data=df, ax=ax, hue=’day’, palette=’bright’)

- 색 직접 지정  (hex코드 사용 가능)

fig, ax = plt.sublots()

sns.scatterplot(x=’total_bill’, y=’tip’, data=df, ax=ax, hue=’day’, hue_order=[’Thur’, ‘Fri’, ‘Sat’, Sun’], palette =[’black’, ‘cyan’, ‘purple’, ‘salmon’])

or

sns.scatterplot(x=’total_bill’, y=’tip’, data=df, ax=ax, hue=’day’, palette ={’Thur’ : ’black’, ‘Fri’ : ‘cyan’, ‘Sat’ :‘purple’, ‘Sun’ : ‘salmon’})

- color palette 사용 (hex코드 사용 가능)

color = sns.color_palette(”coolwarm”, as_cmap=True)

fig, ax = plt.sublots()

sns.scatterplot(x=’total_bill’, y=’tip’, data=df, ax=ax, hue=’size’, palette=color)

### Seaborn Heatmap 에 color적용

pivot = df.pivot_table(index=’color’, columns=’clarity’, values=’price’)

clarity_order = [’I1’, ‘SI2’, ‘SI1’, ‘VS2’, ‘VS1’, ‘VVS2’, ‘VVS1’, ‘IF’]

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(pivot[clarity_order], annot=True, fmt=’.0f’)

![image.png](image%2031.png)

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(pivot[clarity_order], annot=True, fmt=’.0f’,

cmap = [’black’, ‘darkgrey’, ‘lightgrey’, ‘white’])

![image.png](image%2032.png)

- (참고) color map 직접 만들어서 사용

from matplotlib.colors import LinearSegmentedColormap

color = LinearSegmentedColormap.from_list(’custom color’, [(0, ‘#ffffff’), (0.5, ‘#ffffff’), (1, ‘#0000ff’)], N=256)

fig, ax = plt.subplots(figsize=(8,,6))

sns.heatmap(pivot[clarity_order], annot=True, fmt=’.0f’, linewidth=0.5, linecolor=’black’, cmap=color)

![image.png](image%2033.png)

# Plotly 에서 color 활용

fig = px.colors.qulitative.swatches() ← 컬러 예시 보는 코드

fig.show()

![image.png](image%2034.png)

fig = px.colors.sequential.swatches_continuous()

fig = px.colors.diverging.swatches_continuous()

fig = px.colors.cyclical.swatches_continuous()

- 적용해보기(수치형)

fig = px.scatter(df, x=’total_bill’, y=’tip’, width=500, height=400, color=’size’, color_continuous_scale=’balance’)

fig.show()

![image.png](image%2035.png)

- 범주형 데이터 인자에 적용하기

fig = px.scatter(df, x=’total_bill’, y=’tip’, width=500, height=400, color=’day’, color_discrete_sequence=px.colors.qualitative.Light24)

![image.png](image%2036.png)

- 색 직접 지정하기

fig = px.scatter(df, x=’total_bill’, y=’tip’, width=500, height=400, color=’day’, color_discrete_sequence=[’rgb(255, 255, 255)’, ’rgb(0, 0, 0)’, ’rgb(128, 128, 128)’, ’rgb(64, 255, 192)’]

![image.png](image%2037.png)

- 색 직접 지정하기2

fig = px.scatter(df, x=’total_bill’, y=’tip’, width=500, height=400, color=’day’, color_discrete_sequence=[’balck’, ‘white’, ‘blue’, ‘red’])

- 색 직접 지정하기3

fig = px.scatter(df, x=’total_bill’, y=’tip’, width=500, height=400, color=’day’, color_discrete_sequence={’Thur’ : ’balck’, ‘Fri’ : ‘white’, ‘Sat’ : ‘blue’, ‘Sun’ : ‘red’])

### Plotly의 heatmap 에 적용

pivot = df.pivot_table(index=’color’, columns=’clarity’, values=’price’)

clarity_order = [’I1’, ‘SI2’, ‘SI1’, ‘VS2’, ‘VS1’, ‘VVS2’, ‘VVS1’, ‘IF’]

colors = [’black’, ‘darkgrey’, ‘lightgrey’, ‘white’

fig = px.imshow(pivot[clarity_order], width=500, height=400, text_auto=’4d’, color_continuous_scale=colors)

![image.png](image%2038.png)

- (활용) color map 직접 만들어서 활용하기

fig = px.imshow(pivot[clarity_order], width=500, height=400, text_auto=’4d’)

fig.update_coloraxes(showscale=True, colorsclae=[(0.0, ‘#FFFFFF’), (0.5, ‘#FFFFFF’), (1, ‘#0000FF’),],)

![image.png](image%2039.png)