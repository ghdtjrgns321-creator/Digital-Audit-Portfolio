# 파이썬 기초강의

- 출력
    - 숫자 print(123)
    - 문자 따옴표, 큰따옴표 다 가능
- 주석
    - #주석, ctrl+/ (단축키)
- 산술연산자
    - +, -, *, /, //(몫), %(나머지), **(제곱)
    - 정수+실수 더하기 가능
        - print(3.44+5)
    - 문자열 더하기 가능
        - print(’he’+’llo’)
    - 올림, 내림, 버림
    
    import numpy as np
    
    올림 : np.ceil()
    
    내림 : np.floor()
    
    버림 : np.trunc()
    
- 자료형(타입)
    - 정수형 int
    - 실수형 float
    - 문자열 str
    - 참/거짓 bool
        - 타입확인 type()
            - type(True) → bool
            - type(False) → bool
- 등호 1개 (=)
    - 오른쪽 값을 왼쪽에 대입한다
- 변수 → 숫자로 시작하면 안됨
    - box = 10
    - print(box) → 10
    - box2 = 20
    - print(box+box2) → 30
    - box3 = box + box2
    - print(box3) → 30
    - box = box + box
    - print(box) → 20
    - box = box + box
    - print(box) → 40
- 형변환
    - box = 10
    - int(box) 하면 정수가 됨
    - str(box) 하면 문자열이 됨
    - int(문자) 시 에러발생
- 비교연산자
    - >, <, ==(같다), <=, >=, !=
- 조건문
    - 들여쓰기 tab, 내어쓰기 shift+tab

box = 7

if box >= 10:

print(”box는 10 보다 큽니다”)

elif box < 10 and box >= 5:

print(”box는 5이상 10 미만입니다”)

else:

print(”box는 5미만입니다”)

→ box는 5이상 10미만입니다

- print.format

.format() 사용

print("제 이름은 {}이고, 나이는 {}살입니다.".format(name, age))

- f스트링

f-string 사용

print(f"제 이름은 {name}이고, 나이는 {age}살입니다.")

숫자 = 88.5555
print(f"소숫점 2자리: {숫자:.2f}")

*# .2f = 소숫점 2자리 고정*

print(f"소숫점 1자리: {숫자:.1f}")

*# 출력:*

*# 소숫점 2자리: 88.56*

*# 소숫점 1자리: 88.6*

print(f"생존자: {생존자}명 ({비율:.2%})")

*# 출력: 생존자: 342명 (38.38%)*

*# :.2% = 소숫점 2자리까지 백분율로 표시*

*# :.1% = 소숫점 1자리까지 백분율로 표시*

- 리스트
    - 빈 리스트 만들기 : list = []
    - 리스트 → 대괄호
    - listbox = [’a’,’b’,3,4,’c’,’d’,’e’]
    - 리스트 마지막에 추가 → listbox.append(’f’)
    - 리스트 정렬 → listbox.sort() , 문자와 숫자 섞어서 소팅은 불가능함
    - 리스트 안에 값 찾기
        - list[0] → a
        - list[-1] → e
    - 리스트 안의 리스트 값 찾기
        - list = [1, 2, 3, [’a’, ‘b’, ‘c’]]
        - list[3][0] → a
- 딕셔너리
    - dictbox = {’name’ : ‘sam’, ‘level’ : 5}
    - dictbox[’name’] → sam
    - dictbox[’level’] → 5
    - dictbox[’name’] = sugar → sam이 sugar로 바뀜
    - dictbox.keys() → dict_keys([’name’,’level’])
    - dictbox.values() → dict_values([’sugar’,5])
- 리스트 안의 딕셔너리
    
    products[0]
    
    *# {'name': '노트북', 'price': 1000, 'stock': 5}*
    
    products[0]['name']
    
    *# '노트북'*
    
    products[0]['price']
    
    *# 1000*
    
- 튜플 ← 값을 못 바꿈
    - t=(1,2,3)
- 세트 ← 수학의 집합과 같음 ( 중복 제거 됨)
    
    {1,2,3,1,2,3,1,2,3,1,2,2,1,2,1,3,}
    
    → {1,2,3,}
    
- 인덱싱과 슬라이싱
    - listbox = [2,4,6,8,10]
    - 처음이 0, 마지막이 -1
    - listbox[0] → 2
    - listbox[1] → 4
    - listbox[-1] → 10
    - listbox[0:3] → 0번, 1번, 2번 3번x
    - listbox[::2] → 한칸 건너띄기
- 리스트 내장함수
    - sum(listbox) → 리스트 다 더함
    - max(listbox), min(listbox) 가능
    - len(listbox) → 5
    - round(1.2345, 2) → 1.23
- 문자열 변경
    - text = “빅데이터 분석기사 파이썬 공부”
    - text = text.replace(”공부”,”스터디”)
    - print(text) → 빅데이터 분석기사 파이썬 스터디

- 반복문 type 1
    - 5부터 10 미만까지 반복
        - for i in range(5, 10):
        - print(i)
        
        5
        
        6
        
        7
        
        8
        
        9
        
    - 5번 반복
        - for i in ragnge(5)
        - print(i)
        
        0
        
        1
        
        2
        
        3
        
        4
        
- 반복문 type 2
    - 리스트에 있는 각 원소 길이만큼 반복
        - listbox = [2, 4, 6, 8, 10]
        - for col in listbox:
        - print(col)
            
            2
            
            4
            
            6
            
            8
            
            10
            
        - listbox = [2, 4, 6, 8, 10]
        - for col in listbox:
        - print(col * 2)
            
            4
            
            8
            
            12
            
            16
            
            20
            
- 리스트에 있는 문자열과 인덱스값 출력
    - listbox = [’네모’,’세모’,’동그라미’,’별’]
    
    for index, item in enumerate(listbox):
    
    print(index)
    
    print(item)
    
    0
    
    네모
    
    1
    
    세모
    
    2
    
    동그라미
    
    3
    
    별
    
- enumerate
    
    seq = [’a’, ‘b’, ‘c’, ‘d’]
    
    list(enumerate(seq)) → [(0,’a’), (1,’b’), (2,’c’), (3,’d’)]
    
    for i, x in enumerate(seq):
    
    print(i, x)
    
    0 a
    
    1 b
    
    2 c
    
    3 d
    
- while ← 참일동안 반복
    
    i = 1
    
    while i < 5:
    
    print(’i is: {}.format(i))
    

→   i is: 1

i is: 2

i is: 3

i is: 4

- range ← 0부터 범위 지정
    
    for i in range(1, 11):
    
    print(i)
    
    1
    
    2
    
    3
    
    4
    
    5
    
    6
    
    7
    
    8
    
    9
    
    10
    
- list comprehension

x=[1,2,3,4,5]

out = []

for i in x:

out.append(i**2)

out

→ [1, 4, 9, 16, 25]

한줄로 표현 

[i**2 for i in x] → [1, 4, 9, 16, 25]

- 함수 1
    
    def hello():
    
    print(”안녕하세요!”)
    
    hello() → 안녕하세요!
    
- 함수 2
    
    def plus(x,y):
    
    print(x+y)
    
    plus(2,3) → 5
    
- 함수 3
    
    def plus(x,y):
    
    result = x+y
    
    return result
    
    x = 2
    
    y = 2
    
    total = plus(x, y) → 4
    

★★★data → 파라미터★★★ 

- 함수 4
    
    listbox = [15, 46, 78, 24, 56]
    
    def min_max(data):
    
    mi = min(data)
    
    ma = max(data)
    
    return mi, ma
    
    a, b = min_max(listbox)
    
    print(a,b) → 15 78
    
- 함수 5
    
    listbox = [15, 46, 78, 24, 56]
    
    def mean(data)
    
    return sum(data) / len(data)
    
    mean(listbox) → 43.8
    
- lambda expressions

def test(x):

return x*10

test(10) 

→ 100

을 lambda로 한줄표현

(lambda x : x*10)(10)

→ 100

- map and filter

seq = [1,2,3,4,5]

list(map(lambda x:x*2 seq))

→ [2,4,6,8,10]

list(filter(lambda x:x%2 == 0, seq))

→ [2,4]

- 메서드

st.lower() → 소문자

st.upper() → 대문자

st.split() → 괄호 안에 있는것 기준으로 쪼개기

딕셔너리에서

d.keys() ← 키값들 반환

d.items() ←키와 밸류값 같이 반환

# Numpy

    ## Numpy Array

- 1차원 - vector
- 2차원 - matrix
- 3차원 이상 - tensor
- rank - array의 dimension(차원)
- shape - 각 dimension의 size
- dtype - tensor 의 data type

- Matrix의 numpy 표현
    

    - np.array([[8,5,3], [1,2,9]]) → [[8,5,3]
        
                                                           [1, 2, 9]]

        
    - np.array([[[1,2], [3,4,]], [[5,6], [7,8]]])
    
    → [[[1 2]
    
           [3 4]]
    
           [[5 6]
    
             [7 8]]]
    
- concatenate

- slicing

## Numpy와 선형대수

### ndarray

- n_dimensional array(다차원 배열 객체)로 구성

import numpy as np

- 스칼라

x = 6

- 1차원 array(vector)

x = np.array([1, 2, 3]) *#엘리멘트 갯수가 3인 벡터*

np.argmax(x) *#가장 큰 **인덱스** 값 반환*

→ 2 (인덱스 : 0, 1, 2)

= x.argmax()

- 2차원 array(matrix)

y = np.array([[1,2,3], [2,3,4], [5,6,7], [8,9,10]])

print(y)


y.shape

→ (4, 3)

- 3차원 array(tensor)

z = np.array([[1,2,3], [2,3,4,]], [[5,6,7], [8,9,10]]])

print(z)

z.shape

(2, 2, 3)


- 슬라이싱
  
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(arr)


print(arr[:2, 1:])

print(arr[2::])

print(arr[:, :2])

print(arr[1, :2])


### 벡터의 내적(inner product)

- 내적이 되려면 두 벡터의 dimension이 같아야함.
- 내적은 각 element 의 곱을 모두 더한 것
- inner product는 dot product(점곱) 이라고도 불림

a=np.array([2,5,1])

print(a)

b= np.array([4,3,5])

print(b)

a, b의 내적 —> scalar

np.dot(a, b)

→ 28

### Matrix 곱셈 (행렬 곱셈, dot product)

- 두 행렬 A, B는 A의 열 갯수가 B의 행 갯수와 같을 때 곱할 수 있음
- 결과 행렬 C의 shape은 A의 row x B의 Column이 됨

a = np.array([[2,1], [1,4]])

b = np.array([[1,2,0],[0,1,2]])

np.matmul(a,b)

### 전치행렬 (Transposed Matrix)

list(range(9)) → [0,1,2,3,4,5,6,7,8]

np.arrange(9) → array([0,1,2,3,4,5,6,7,8])

np.arragne(9).reshape(3,3)

np.arrange(9).reshape(3,3).T *#행, 열 바뀜*
