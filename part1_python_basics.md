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