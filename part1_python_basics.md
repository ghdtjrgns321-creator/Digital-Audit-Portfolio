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
- 변수
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

- 리스트
    - 리스트 → 대괄호
    - listbox = [’a’,’b’,3,4,’c’,’d’,’e’]
    - 리스트 마지막에 추가 → listbox.append(’f’)
    - 리스트 정렬 → listbox.sort() , 문자와 숫자 섞어서 소팅은 불가능함
- 딕셔너리
    - dictbox = {’name’ : ‘sam’, ‘level’ : 5}
    - print(dictbox[’name’]) → sam
    - print(dictbox[’level’]) → 5
    
    - dictbox[’name’] = sugar → sam이 sugar로 바뀜
    
    - dictbox.keys() → dict_keys([’name’,’level’])
    - dictbox.values() → dict_values([’sugar’,5])
- 인덱싱과 슬라이싱
    - listbox = [2,4,6,8,10]
    - 처음이 0, 마지막이 -1
    - listbox[0] → 2
    - listbox[1] → 4
    - listbox[-1] → 10
    - listbox[0:3] → 0번, 1번, 2번 3번x!!!
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
    
- f스트링
    - print(f’인덱스{index}인 값은{item}입니다.’)
    
    인덱스0인 값은네모입니다.
    
    인덱스1인 값은세모입니다.
    
    인덱스2인 값은동그라미입니다.
    
    인덱스3인 값은별입니다.
    
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

    ## Numpy Array

- 1차원 - vector
- 2차원 - matrix
- 3차원 이상 - tensor
- rank - array의 dimension(차원)
- shape - 각 dimension의 size
- dtype - tensor 의 data type

![image.png](attachment:cdd5ab24-cc12-4984-a746-c113aa516cf3:image.png)

![image.png](attachment:6cf31e7e-f7fd-4223-9232-8146ac9bb134:image.png)

- Matrix의 numpy 표현
    
    ![image.png](attachment:72c091e1-8302-49bd-ada9-de7c791d6c06:image.png)
    
    - np.array([[8,5,3], [1,2,9]]) → [[8,5,3]
        
                                                           [1, 2, 9]]
        
        ![image.png](attachment:b7bf4087-fa43-4b70-94fc-7a99df06e842:image.png)
        
    - np.array([[[1,2], [3,4,]], [[5,6], [7,8]]])
    
    → [[[1 2]
    
           [3 4]]
    
           [[5 6]
    
             [7 8]]]
    
- concatenate

![image.png](attachment:e10dfae3-ae67-4561-b42d-105db87dc598:image.png)

- slicing

![image.png](attachment:ecc0f126-ed87-465f-aa9e-2f38856bfebf:image.png)

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

![image.png](attachment:2c87e0e9-5fb4-467a-a9f7-2000752aa7a5:image.png)

y.shape

→ (4, 3)

- 3차원 array(tensor)

z = np.array([[1,2,3], [2,3,4,]], [[5,6,7], [8,9,10]]])

print(z)

![image.png](attachment:1b8fc02c-7515-4215-ba47-129b5383ae7f:image.png)

z.shape

(2, 2, 3)

![image.png](attachment:d49ac748-3137-4994-b13a-9e3bd0fbd1b3:image.png)

![image.png](attachment:f6b3d875-af54-469c-8d1f-65b944d70ed5:image.png)

- 슬라이싱

![image.png](attachment:aff9878c-7cb7-4bb4-a4c1-870f01d6d0d5:image.png)

arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(arr)

![image.png](attachment:09c523b8-4e2b-44be-af99-721abd07062e:image.png)

print(arr[:2, 1:])

![image.png](attachment:05bd4636-a070-4642-aff1-1c86cf965cd6:image.png)

print(arr[2::])

![image.png](attachment:d3c06fe4-1939-48d9-85e9-5ddb92d58c1f:image.png)

print(arr[:, :2])

![image.png](attachment:9c433dd4-a2f0-49d5-97d2-a6a077b03327:image.png)

print(arr[1, :2])

![image.png](attachment:b7af8eae-2dc7-499d-a6b4-7c38be95c263:image.png)

### 벡터의 내적(inner product)

- 내적이 되려면 두 벡터의 dimension이 같아야함.
- 내적은 각 element 의 곱을 모두 더한 것
- inner product는 dot product(점곱) 이라고도 불림

a=np.array([2,5,1])

print(a)

![image.png](attachment:36a23656-fa74-4fe7-8804-63082d55102a:image.png)

b= np.array([4,3,5])

print(b)

![image.png](attachment:a05f6fd6-f620-499c-be38-9c8ffb68238e:image.png)

a, b의 내적 —> scalar

np.dot(a, b)

→ 28

![image.png](attachment:40b1b1ad-4eba-4dc2-9f1c-5c3468efd1b8:image.png)

### Matrix 곱셈 (행렬 곱셈, dot product)

- 두 행렬 A, B는 A의 열 갯수가 B의 행 갯수와 같을 때 곱할 수 있음
- 결과 행렬 C의 shape은 A의 row x B의 Column이 됨

![image.png](attachment:4d599860-cb10-476e-8b27-fb805f86e268:image.png)

a = np.array([[2,1], [1,4]])

b = np.array([[1,2,0],[0,1,2]])

np.matmul(a,b)

![image.png](attachment:d1c59057-4b17-4d59-bfed-1dc25b61fb0a:image.png)

### 전치행렬 (Transposed Matrix)

list(range(9)) → [0,1,2,3,4,5,6,7,8]

np.arrange(9) → array([0,1,2,3,4,5,6,7,8])

np.arragne(9).reshape(3,3)

![image.png](attachment:d71d45e6-ce60-4333-932b-7a1c5391e35f:image.png)

np.arrange(9).reshape(3,3).T *#행, 열 바뀜*

![image.png](attachment:fc369f1c-d5ae-4fa2-9db7-365b82e29d2f:image.png)
