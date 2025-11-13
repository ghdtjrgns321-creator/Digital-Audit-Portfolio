# Streamlit 웹 대시보드 제작

# Streamlit 라이브러리

- 데이터 분석 및 데이터 대시보드 제작 과정
    - 데이터 수집 → 인사이트 → 배포
- 설치

!pip install streamlit

import pandas as pd

import streamlit as st

- ipynb 파일을 py 파일로 변경(터미널)

jupyter nbconvert --to script streamlit.ipynb

- 실행(터미널)

streamlit run main_streamlit_df.py

# Text위젯

- 제목을 생성하는 함수들
    - title, header, subheader 순으로 큰글씨 → 작은글씨
    
    st.title(’This is title’)
    
    st.header(’This is header’)
    
    st.subheader(’This is subheader’)
    



- Streamlit 마크다운, 텍스트, 코드

st.markdown( ‘’’ This is main text. This is how to change the color of text : red[Red,] :blue[Blue,] :green[Green.] This is **Bold** and *Italic* text’’’)


st.text( ‘’’ This is main text. This is how to change the color of text : red[Red,] :blue[Blue,] :green[Green.] This is **Bold** and *Italic* text’’’)


code = ‘’’

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.ticker import MultipleLocator, IndexLocator, FuncFormatter from matplotlib.dates import MonthLocator, DataFormatter

fig ax = plt.subplots(figsize=(5,5))

sns.scatterplot(x=’temp’, y=’count’, data=df, ax=ax)

ax.yaxis.set_minor_locator(MultipleLocator(5))

ax.xaixs.set_minor_locator(MultipleLocator(1))

‘’’

st.code(code, language=’python’)



- 페이지 나누기 : divide 함수
    - 페이지에 가로선 삽입

st.title(’Title 1’)

st.text(’Text body 1’)

st.divider()

st.title(’Title 2’)

st.text(’Text body2’)



## Button

- 첫 번째로 전달되는 문자열 인자가 버튼의 label로 표시
- 앱에서 생성된 button 클릭 시 True 가 반환, if문을 통해 버튼 입력 시 실행될 동작 코딩 가능
- on-click 인자를 통해 클릭 시 수행될 함수를 받을 수 도 있음
- type 인자에 primary를 전달하면 버튼의 색깔을 빨간 색으로 표현 가능
    - def 방식
        
        def button_write():
        
        st.write(’button activated’)
        
        st.button(’Rest’, type=’primary’)
        
        sy.button(’activate’, on_click=button_write) 
        
    - if문 방식
        
        st.button(’Restet’, type=’primary’)
        
        if st.button(’activate’):
        
        st.write(’button activated’)
        



## CheckBox

- 첫 번째로 전달되는 문자열 인자가 버튼의 label로 표시
- 체크박스 클릭 시 True가 반환, if문을 통해 버튼입력 시 실행될 동작 코딩 가능
- on_change 인자를 통해 클릭 시 수행될 함수를 받을 수 도 있음
    - if 문 사용
        
        active = st.checkbox(’I agree’)
        
        if active:
        
        st.text(’Great’)
        
    - on_change 인자 사용
        
        def checkbox_write():
        
        st.write(’Great!’)
        
        st.checkbox(’I agree’, on_change=checkbox_write)
        



## togglebox

- 첫 번재로 전달되는 문자열 인자가 토글박스의 label로 표시
- 앱에서 생성된 토글박스 클릭 시 Ture가 반환되어 if문을 통해 버튼 입력 시 실행될 동작 코딩 가능
- on_change 인자를 통해 클릭 시 수행될 함수를 받을 수 도 있음

toggle = st.toggle(’Turn on the switch!’, value=True)

if toggle:

st.text(’Switch is turned on!’)

else:

st.text(’Switch is turend off!’)


## selectbox

- 첫 번째로 전달되는 문자열 인자가 버튼의 제목으로 표시
- selectbox 클릭 시 선택된 항목이 반환되어, if문을 통해 버튼입력 시 실행될 동작 코딩 가능
- on_change 인자를 통해 값 선택 시 수행될 함수를 받을 수도 있음

option = st.selectbox(label = ‘yout selection is’, options=[’Car’, ‘Airplane’, ‘Train’, ‘Ship’],)

st.text(’you selected: {}’. format(option))

- 선택 전 빈칸에 문자를 띄우고 싶을 때

option = st.selectbox(label = ‘yout selection is’, options=[’Car’, ‘Airplane’, ‘Train’, ‘Ship’], index=None, placeholder=’selecet transportation’)




## radio button

- 첫 번째로 전달되는 문자열 인자가 버튼의 label로 표시
- options 인자로 전달된 리스트가 개별 버튼의 옵션으로 설정됨
- 앱에서 생선된 버튼 클릭 시 선택된 항목이 반환되어, if문을 통해 버튼 입력 시 실행될 동작 코딩 가능
- on_change 인자를 통해 클릭 시 수행될 함수를 받을 수도 있음

option = st.radio(’What is your favorite movie genre’, [”Comedy”, “Drama”, “Documentary”], captions = [’Laugh out loud’, ‘Get the popcorn’, ‘Never stop learning’])

if option:

st.text(’You Selected {}’.format(option))

## multiselect

- 첫 번째로 전달되는 문자열 인자가 버튼의 제목으로 표시
- options 인자로 전달된 리스트가 선택 가능한 옵션으로 설정됨
- 선택된 항목이 반환되어 if문을 통해 버튼 입력 시 실행될 동작 코딩 가능
- on_change 인자를 통해 값 선택 시 수행될 함수를 받을 수도 있음

option = st.multiselect(label=’your selection is’, options=[’Car’,’Airplane’,’Train’,’Ship’], placeholder=’select transportation’)

st.text(’yot selected: {}’.format(option))

## text input

- 첫 번째로 전달되는 문자열 인자가 위젯의 label로 표시
- placerholder 인자에 빈 칸에 기본으로 쓰여질 문자열 설정
- type 인자에 password 전달 시 입력 값 암호화
- 값 입력시 해당 입력값이 반환되어 if문을 통해 버튼 입력 시 실행될 동작 코딩 가능
- on_change 인자를 통해 입력값 변화 시 수행될 함수를 받을 수 도 있음

string = st.text_input(’Movie title’, placeholder=’write down the title of your favorite movie’)

if string:

st.text(’Your answer is ‘+string)

string = st.text_input(’Movie title’, placeholder=’write down the title of your favorite movie’, type=’password’)

if string:

st.text(’Your answer is ‘+string)

## File upload

- 첫 번째로 전달되는 문자열 인자가 버튼의 label로 표시
- type 인자를 통해 받을 수 있는 확장자명 입력
- accept_multiple_files 인자를 통해 동시에 하나 이상의 파일 업로드 가능 유무 설정
- on_change 인자를 통해 파일 업로드 시 수행될 함수를 받을 수도 있음

file = st.file_uploader(’Choose a file’, type=’csv’, accept_multiple_file=False)

if file is not None:

df = pd.read_csv(file)

st.write(df)

## Slider

- 값 or 범위 선택 가능한 slider 생성. 첫 번째로 전달되는 문자열인자가 위젯의 label로 표시
- min_value, max_value 인자를 통해 선택 가능 값의 범위 설정
- value 인자에 튜플로 묶은 2개의 값 전달 시 구간 선택 가능
- 슬라이더 조절 시 선택된 항목이 반환되어, if문을 통해 버튼 입력 시 실행될 동작 코딩 가능
- on_change  인자를 통해 슬라이더 이동 시 수행될 함수를 받을 수도 있음

score = st.slider(’Your score is …’, 0, 100, 1)

st.text(’Score: {}’.format(score))

from datetime import time

start_time, end_time = st.slider(

‘Working time is …’,

min_value=time(0), max_value=time(23), value=(time(8), time(18)),format=’HH:mm’)

st.text(’Working time: {}, {}’.format(start_time, end_time))

## 차트 및 이미지 표현하기

### Matplotlib & Seaborn을 통해 그린 그래프 표현

import matplotlib.pyplot as plt

import seaborn as sns

df = sns.load_dataset(’tips’)

fig, ax = plt.subplots()

sns.histplot(df, x=’total_bill’, ax=ax, hue=’time’)

st.pyplot(fig)

### Ployly 를 통해 그린 그래프 표현

import [plotly.express](http://plotly.express) as px

fig2 = px.box(

data_frame=df, x=’day’, y=’tip’, facet_col = ‘smoker’, facet_row = ‘sex’, width=800, height=800

)

st.plotly_chart(fig2)

### (활용) streamlit 앱에서 원하는 변수를 선택하여 그래프 그리기

- 3개의 selectbox 위젯을 통해 x,y,hue로 설정할 변수 선택
- 선택된 변수들을 통해 plotly 그래프 생성

x_options = [’day’, ‘size’]

y_options = [’total_bill’, ‘tip’]

hue_options = [’smoker’, ‘sex’]

df = sns.load_dataset(’tips’)

x_option = st.selectbox(’Select X-axis’, index=None, options=x_options)

y_option = st.selectbox(’Select Y-axis’, index=None, options=y_options)

hue_option = st.selectbox(’Select Hue’, index=None, options=hue_options)

if (x_option != None) & (y_option != None):

if hue_option != None:

fig = px.box(

data_frame=df, x=x_option, y=y_option, color=hue_option, width=500

)

else:

fig = px.box(

data_frame=df, x=x_option, y=y_option, width=500

)

st.plotly_chart(fig) 


## 이미지 생성하기

- PIL 라이브러리를 통해 image 파일 불러오기
- Streamlit image 함수를 통해 앱에 이미지 업로드
- image 함수에 width 인자를 통해 이미지 너비 조절
- image 함수에 caption 인자를 통해 캡션 추가

from PIL inport Image

img = Image.open(’datasets/images/image1.jpg’)

st.image(img, width=300, caption=’Image from Unsplash’)


## Layout 이해하기

### 사이드바 생성하기

- 메인 페이지 밖 별도 메뉴 바 생성
- streamlit sidebar를 with 구문으로 작성하여, with 구문 내 코드는 사이드바에 생성

st.title(’This is main page’)

with st.sidebar:

st.title(’This is sidebar’)

side_option = st.multiselect(

label=’your selection is’,

options=[’Car’, ‘Airplane’, ‘Train’, ‘Ship’, ‘ Bicycle’],

placeholder=’select transportation;

)


### Column 생성하기

- 메인 페이지 단 나누기
- 세로 배치보다 가로 배치가 더 효율적일 때 사용
- streamlit column 함수로 받은 변수(column)들을 with 구문으로 작성하여, with구문 내 코드는 각 column 내에 생성

col1, col2 = st.columns(2)

with col1:

st.header(’Lemonade’)

st.image(img2, width=300, caption=’Image from Unsplash’)

with col2:

st.header(’Cocktail’)

st.image(img3, width=300, caption=’Image from Unsplash’)

## tab 생성하기

- 엑셀 워크시트의 tab 처럼 각 tab 별로 독립적인 구성요소 배치 가능
- Streamlit tab 함수를 통해 받은 변수(tab)들을 with 구문으로 작성하여, with구문 내 코드는 각 tab 내에 생성

tab1, tan2 = st.tabs([’Table’, ‘Graph’])

df = pd.read_csv(datasets/medical_cost/medical_cost.csv’)

df = df.query(’region == “northwest”’)

with tab1:

st.table(df.head(5))

with tab2:

fig = px.scatter(

data_frame=df, x=’bmi’, y=’charges’

)

st.plotly_chart(fig)


### expander 생성하기

- 특정 요소를 숨기기/보이기 처리할 수 있는 위젯 생성
- 필수가 아닌 요소들을 숨겨 앱을 깔끔하게 유지하는 데 유용
- streamlit expander를 with 구문으로 작성하여, with 구문 내 코드는 expander내에 생성

df = pd.read_csv(’datasets/medical_cost/medical_cost.csv’)

df= df.query(’region == “northwest”’)

with st.expander(”See datatable”):

st.table(df.head(5))

## Streamlit Session State 이해하기

### 사용자와 interaction 시 변화하는 데이터 유지하기

- Streamlit 앱은 사용자가 특정 동작을 할 때마다 세션을 초기화 함
- 특정 동작 시 변화하는 변수들을 모두 초기화 하지 않고 유지할 필요성 있음
- (예시) 아래 결과에서 i에 1을 더해주는 버튼을 눌러도 1이상 값이 올라가지 않음

i = 0

st.header(’Session state example1’)

plus_one = st.button(label=’+1’,)

if plus_one:

i += 1

st.text(’i = {}’.format(i))


- 세션 초기화 시 변수들을 저장할 수 있는 session_state 제공
- sessiont_state에 변수와 값을 변수:값 형태의 딕셔너리로 저장

st.header(’Session state example2’)

if ‘i’ not in st.sesstion_state:

st.sesstion_state[’i’] = 0

plus_one = st.button(label=’+1’, key=’btn_plus1’)

if plus_one:

st.sesstion_state[’i’] += 1

st.text(’i = {}’.format(st.session_state[i]))


## 캐싱 이해하기

- streamlit과의 interaction 시 세션 초기화 때문에 계산 비용이 높은 함수가 있을 경우 매 번 높은 비용의 계산을 수행하여야 함
- 캐싱을 통하여 자주 변하지 않는 계산 결과나 함수를 따로 저장하여 효율화
- Streamlit cache 데코레이터를 이용하여 사용
- cache의 clear_cache 메서드를 이용하여 캐싱 값 삭제

@st.cache_data

def expensive_computation(a, b):

st.text(’Result: {}’.format(a+b))

result = st.button(’Calculate’, on_click=expensive_computation, args(3, 4,))


# Streamlit 웹 대시보드 제작 실습

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_dataset(path):
return pd.read_csv(path)

path = 'C:\sh\python\jupyter\datasets\CO2_emissions/CO2_Emissions.csv'
df = load_dataset(path)

#메인 페이지 구성
st.title(
'Data Analysis - CO2 emissions'
)
st.write(
'''
Hello there, this web page is a simple data analysis web dashboard created using the Python Streamlit library.
On this page, you can visualize the distribution of some variables or the correlation between variables.
'''
)
st.divider()

#사이드바 생성
makers = df['Vehicle Class'].unique().tolist()
with st.sidebar:
st.markdown('Filter the data you want to analyze: :tulip:')

st.multiselect(
    'Select the vehicle class you want to analyze: ',
    makers, default=['TWO-SEATER'], key='maker_filter'
)
st.slider(
    'Select the engine size you want to analyze: ',
    min_value=df['Engine Size(L)'].min(),
    max_value=df['Engine Size(L)'].max(),
    value=(df['Engine Size(L)'].quantile(0.1), df['Engine Size(L)'].quantile(0.95)),
    step=0.3, key = 'engine_filter'
)

df = df.loc[
(df['Vehicle Class'].isin(st.session_state['maker_filter'])) &
(df['Engine Size(L)'] < st.session_state['engine_filter'][1]) &
(df['Engine Size(L)'] > st.session_state['engine_filter'][0])
]

그래프1

st.subheader(
'Analysis of Engine Sizes'
)
col1, col2 = st.columns(2)
with col1:
st.write(
'''
The box plot of engine sizes by automotive manufacturer. What type of engine sizes do manufacturers produce the most for each brand?
'''
)
with col2:
fig1 = px.box(
data_frame=df.sort_values('Engine Size(L)', ascending=False),
x='Make', y='Engine Size(L)', width=300, height=400, points='all'
)
st.plotly_chart(fig1)
st.divider()

그래프2

st.subheader('Analysis of Fuel Consumption')

col3, col4 = st.columns(2)
with col3:
st.write(
'''
The scatter plot graph illustrating fuel efficiency based on engine sizes.
Which manufacturer might have lower fuel efficiency within the same engine size?
Which manufacturer might have higher fuel efficiency within the same engine size?
'''
)
st.selectbox(
'Select Y-axis: ',
[
'Fuel Consumption City (L/100 km)',
'Fuel Consumption Hwy (L/100 km)',
'Fuel Consumption Comb (L/100 km)'
],
key = 'fig2_yaxis'
)
with col4:
fig2 = px.scatter(
data_frame=df, x='Engine Size(L)', y=st.session_state['fig2_yaxis'],
width=500, color='Make', trendline='ols', trendline_scope='overall'
)
st.plotly_chart(fig2)
st.divider()

그래프3

st.subheader('Analysis of Carbon Emissions')

col5, col6 = st.columns(2)
with col5:
st.write(
'''
The scatter plot graph depicting the correlation between fuel efficiency and
carbon emissions, with color differentiation for each manufacturer.
Which manufacturer might have higher carbon emissions within the same fuel
efficiency range?
'''
)
st.selectbox(
'Select X-axis: ',
[
'Fuel Consumption City (L/100 km)',
'Fuel Consumption Hwy (L/100 km)',
'Fuel Consumption Comb (L/100 km)'
],
key='fig3_xaxis'
)
with col6:
fig3 = px.scatter(
data_frame=df, x=st.session_state['fig3_xaxis'], y='CO2 Emissions(g/km)', width=500, color='Make', trendline='ols', trendline_scope='overall'
)
st.plotly_chart(fig3)
