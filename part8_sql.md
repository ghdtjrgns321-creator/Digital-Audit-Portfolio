# 데이터베이스와 RDBMS의 이해

## **1. RDBMS(Relational Database Management System) 이해**

### **1.1 데이터베이스란?**

- 체계화된 데이터의 모임
- 여러 응용 시스템들의 통합된 정보를 저장하여, 운영할 수 있는 공용 데이터의 묶음
- 논리적으로 연관된 하나 이상의 자료 모음으로, 데이터를 고도로 구조화함으로써 검색/갱신등의 데이터 관리를 효율화함
- DBMS: 데이터베이스를 관리하는 시스템
- 데이터베이스 장점
    1. 데이터 중복 최소화
    2. 데이터 공유
    3. 일관성, 무결성, 보안성 유지
    4. 최신의 데이터 유지
    5. 데이터의 표준화 가능
    6. 데이터의 논리적, 물리적 독립성
    7. 용이한 데이터 접근
    8. 데이터 저장 공간 절약
- 데이터베이스 단점
    1. 데이터베이스 전문가 필요
    2. 많은 비용 부담
    3. 시스템의 복잡함
- 데이터베이스 랭킹 예
    - [https://db-engines.com/en/ranking](https://db-engines.com/en/ranking)

### **1.2 RDBMS(Relational Database Management System, 관계형 데이터베이스 관리 시스템)**

- 데이터베이스의 한 종류로, 가장 많이 사용됨
- 역사가 오래되어, 가장 신뢰성이 높고, 데이터 분류, 정렬, 탐색 속도가 빠름
- 관계형 데이터베이스 = 테이블!
- 2차원 테이블(Table) 형식을 이용하여 데이터를 정의하고 설명하는 데이터 모델
- 관계형 데이터베이스에서는 데이터를 속성(Attribute)과 데이터 값(Attribute Value)으로 구조화(2차원 Table 형태로 만들어짐)
- 데이터를 구조화한다는 것은 속성(Attribute)과 데이터 값(Attribute Value) 사이에서 관계(Relation)을 찾아내고 이를 테이블 모양의 구조로 도식화함의 의미함
- 주요 용어

![image.png](image.png)

- Primary Key and Foreign Key
    - Primary Key(기본키): Primary Key는 한 테이블(Table)의 각 로우(Row)를 유일하게 식별해주는 컬럼(Column)으로,각 테이블마다 Primary Key가 존재해야 하며, NULL 값을 허용하지 않고, 각 로우(Row)마다 유일한 값이어야 한다.
    - Foreign Key(외래키 또는 외부키): Foreign Key는 한 테이블의 필드(Attribute) 중 다른 테이블의 행(Row)을 식별할 수 있는 키

![image.png](image%201.png)

### **1.3 데이터베이스 스키마(Schema)**

- 데이터베이스의 테이블 구조 및 형식, 관계 등의 정보를 형식 언어(formal language)로 기술한 것
    1. 관계형 데이터베이스를 사용하여 데이터를 저장할 때 가장 먼저 할 일은 데이터의 공통 속성을 식별하여 컬럼(Column)으로 정의하고, 테이블(Table)을 만드는 것
    2. 통상적으로 하나의 테이블이 아닌 여러 개의 테이블로 만들고, 각 테이블 구조, 형식, 관계를 정의함
    3. 이를 스키마라고 하며, 일종의 데이터베이스 설계도로 이해하면 됨
    4. 데이터베이스마다 스키마를 만드는 언어가 존재하며, 해당 스키마만 있으면 동일한 구조의 데이터베이스를 만들 수 있음(데이터베이스 백업과는 달리 데이터 구조만 동일하게 만들 수 있음)

![image.png](image%202.png)

### **1.4 SQL(Structured Query Language)**

- 관계형 데이터베이스 관리 시스템에서 데이터를 관리하기 위해 사용되는 표준 프로그래밍 언어(Language)
- 데이터베이스 스키마 생성 및 수정, 테이블 관리, 데이터 추가, 수정, 삭제, 조회 등, 데이터베이스와 관련된 거의 모든 작업을 위해 사용되는 언어
- 데이터베이스마다 문법에 약간의 차이가 있지만, 표준 SQL을 기본으로 하므로, 관계형 데이터베이스를 다루기 위해서는 필수적으로 알아야 함
- SQL은 크게 세 가지 종류로 나뉨
    - 데이터 정의 언어(DDL, Data Definition Language)
    - 데이터 처리 언어(DML, Data Manipulation Language)
    - 데이터 제어 언어(DCL, Data Control Language)

### **1.4.1 데이터 정의 언어(DDL, Data Definition Language): 데이터 구조 정의**

- 테이블(TABLE), 인덱스(INDEX) 등의 개체를 만들고 관리하는데 사용되는 명령
- CREATE, ALTER, DROP, RENAME, TRUNCATE 등이 있음

### **1.4.2 데이터 조작 언어(DML, Data Manipulation Language): 데이터 CRUD [Create(생성), Read(읽기), Update(갱신), Delete(삭제)]**

- INSERT 테이블(Table)에 하나 이상의 데이터 추가.
- UPDATE 테이블(Table)에 저장된 하나 이상의 데이터 수정.
- DELETE 테이블(Table)의 데이터 삭제.
- SELECT 테이블(Table)에 저장된 데이터 조회.

### **1.4.2 데이터 제어 언어(DCL, Data Control Language): 데이터 핸들링 권한 설정, 데이터 무결성 처리 등 수행**

- GRANT 데이터베이스 개체(테이블, 인덱스 등)에 대한 사용 권한 설정.
- BEGIN 트랜잭션(Transaction) 시작.
- COMMIT 트랜잭션(Transaction) 내의 실행 결과 적용.
- ROLLBACK 트랜잭션(Transaction)의 실행 취소.

# **3. SQL DDL(Data Definition Language) 이해 및 실습**

## **3.1 데이터베이스**

- 데이터베이스 안에는 여러 개의 데이터베이스 이름이 존재한다.
- 각 데이터베이스 이름 안에는 여러 개의 테이블이 존재한다.

![image.png](attachment:ee84d83b-4f8c-425f-9e3c-bcae846ba3de:image.png)

1. 데이터베이스 생성
    
    CREATE DATABASE dbname;
    
2. 데이터베이스 목록 보기
    
    SHOW DATABASES;
    
3. 데이터베이스 사용
    
    USE dbname;
    
4. 데이터베이스 삭제
    - IF EXISTS 옵션은 해당 데이터베이스 이름이 없더라도 오류를 발생시키지 말라는 의미의 옵션
    
    DROP DATABASE dbname; or
    
    DROP DATABASE IF EXISTS dbname;
    

## **3.2 테이블**

### **3.2.1 테이블 생성**

CREATE TABLE 테이블명 (

컬럼명 데이터형,

컬럼명 데이터형,

기본키(컬럼명)

)

### PRIMARY KEY

- 한 개 이상 지정 가능, 보통 한개 지정
- NULL 값 등록 X, 컬럼안에서 같은값이 없도록 유일해야함
- PRIMARY KEY 컬럼은 보통 NOT NULL, AUTO_INCREMENTAL(유일해짐) 선언이 되는 경우가 많음

### 숫자형 데이터 타입

![image.png](attachment:bb07a3ec-089e-4f47-957e-9f024f5e5648:image.png)

- 숫자형 타입 옵션
    - UNSIGNED : 0포함 양수만 허용
    - NOT NULL : NULL 허용X
    - AUTO_INCREMENT : 자동으로 숫자가 1씩 증가하며 저장됨. 양수만 가능. 테이블안에서 한 컬럼만 가능 (인덱스 자동생성), 보통 PRIMARY KEY에 지정함

### 문자형 데이터 타입

![image.png](attachment:391b6ee3-ca14-4e47-9015-26102bcd6293:image.png)

### 시간형 데이터 타입

![image.png](attachment:c7ba28ec-c8ad-4184-8836-eb300208e29e:image.png)

### **3.2.1 테이블 조회**

- SHOW TABLES;

![image.png](attachment:10dac3db-af41-4f80-866c-493362399b2f:image.png)

- DESC 테이블명;
    
    DESC mytable;
    

![image.png](attachment:f0f22923-4d4c-4fa2-86c5-94d7aa2f4fba:image.png)

### **3.2.2 테이블 삭제**

- DROP TABLE 테이블명; (IF EXISTS 사용가능)
    
    DROP TABLE IF EXISTS mytable;
    

### **3.2.3 테이블 구조 수정**

- 테이블에 새로운 컬럼 추가
    
    ALTER TABLE 테이블명 ADD COLUMN 추가할컬럼명 추가할컬럼데이터형;
    ALTER TABLE mytable ADD COLUMN model_type varchar(10) NOT NULL
    
- 테이블 컬럼 타입 변경
    
    ALTER TABLE 테이블명 MODIFY COLUMN 변경할컬럼명 변경할컬럼타입;
    
    ALTER TABLE mytable MODIFY COLUMN name varchar(20) NOT NULL;
    
- 테이블 컬럼 이름 변경
    
    ALTER TABLE 테이블명 CHANGE COLUMN 기존컬럼명 변경할컬럼명 변경할컬럼타입;
    
    ALTER TABLE mytable CHANGE COLUMN modelnumber model_num varchar(10) NOT NULL;
    
- 테이블 컬럼 삭제
    
    ALTER TABLE 테이블명 DROP COLUMN 삭제할컬럼명;
    
    ALTER TABLE mytable DROP COLUMN series;
    

# **4. SQL DML(Data Manipulation Language) 이해 및 실습 (focusing on CRUD)**

## **4.1. CRUD [Create(생성), Read(읽기), Update(갱신), Delete(삭제)]**

- 데이터 관리는 결국 데이터 생성, 읽기(검색), 수정(갱신), 삭제 를 한다는 의미

### **4.1.1 데이터 생성**

- 테이블에 컬럼에 맞추어 데이터를 넣는 작업
- 기본 문법 (INSERT)
    1. 테이블 전체 컬럼에 대응하는 값을 모두 넣기
    
    INSERT INTO 테이블명 VALUES(값1, 값2, …)
    
    1. 테이블 특정 컬럼에 대응하는 값만 넣기 (지정되지 않은 컬럼은 디폴트값 또는 NULL값이 들어감)
    
    INSERT INTO 테이블명 (col1, col2, …) VALUES(값1, 값2 …)
    

### **4.1.2 데이터 읽기(검색)**

- 테이블에 저장된 데이터를 읽는 작업

### **기본 문법 (SELECT)**

- **테이블 전체 컬럼의 데이터 모두 읽기**

SELECT * FROM 테이블명; ← * : 모든 테이블

- **테이블 특정 컬럼의 데이터만 읽기**

SELECT 컬럼1, 컬럼2 FROM 테이블명;

### **조건에 맞는 데이터만 검색하기 (비교 연산자 사용)**

- **WHERE 조건문**을 사용하여 특정 조건에 맞는 데이터를 검색할 수 있음
- **비교 연산자:**
    - `=` : 같다
    - `!=` 또는 `<>` : 같지 않다
    - `<` : 작다
    - `>` : 크다
    - `<=` : 작거나 같다
    - `>=` : 크거나 같다

### **조건에 맞는 데이터만 검색하기 (논리 연산자)**

- 여러 조건을 조합하여 데이터를 검색할 수 있음
    - **AND 연산자**: 모든 조건을 만족하는 데이터 검색
    
    SELECT * FROM students WHERE age >= 15 AND grade = ‘10학년’;
    
    - **OR 연산자**: 하나 이상의 조건을 만족하는 데이터 검색
    
    SELECT * FROM students WHERE age <= 14 OR grade = ‘11학년’;
    

### **조건에 맞는 데이터만 검색하기 (LIKE를 활용한 부분 일치)**

- 문자열의 일부를 검색할 때 **LIKE 연산자**를 사용
    - `%` : 0개 이상의 문자를 대체
        
        SELECT * FROM students WHERE name LIKE ‘박%’; ← 박으로 시작하는 이름 검색
        
    - `_` : 단일 문자를 대체
        
        SELECT * FROM students WHERE name LIKE ‘__’; ← 이름이 2글자 검색
        

### **4.1.3 데이터 수정**

테이블에 저장된 데이터를 수정하는 작업입니다.

### **기본 문법 (UPDATE)**

- **특정 조건에 맞는 데이터를 수정**

UPDATE students SET age = 18

WHERE name = ‘박민수’;

- **다수의 컬럼 값을 수정할 수도 있음**

UPDATE students SET grade = ‘10학년’, age = ‘18’

WHERE name = ‘박민수’;

### **4.1.4 데이터 삭제**

테이블에 저장된 데이터를 삭제하는 작업입니다.

### **기본 문법 (DELETE)**

- **특정 조건에 맞는 데이터를 삭제**

DELETE FROM students

WHERE name = ‘김철수’;

- **테이블에 저장된 모든 데이터를 삭제할 수도 있음**

DELETE FROM students;

# SQL 조건 순서

SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT

# 문자열 함수

## LENGTH(string)

- 문자열 길이를 반환

SELECT title, LENGTH(title) AS title_length

FROM film

LIMIT 10;

SELECT * FROM film

WHERE LENGTH(title) = 15;

## UPPER(string)

- 문자열을 대문자로 변환

SELECT title, UPPER(title) AS uppercased_title

FROM film

LIMIT 10;

## LOWER(string)

- 문자열을 소문자로 변환

SELECT title, LOWER(title) AS lowercased_title

FROM film

LIMIT 10;

## CONCAT(string1, string2, …)

- 두 개 이상의 문자열을 하나로 연결

SELECT CONCAT(first_name, ‘ ‘ , last_name) AS full_name

FROM actor

LIMIT 10;

SELECT UPPER(CONCAT(firtst_name, ‘ ‘ , last_name)

FROM actor

WHERE LOWER(first_name) = ‘john’;

## SUBSTRING(string, start, length)

- 문자열에서 부분 문자열을 추출

SELECT SUBSTRING(description, 1, 10) AS short_description

FROM film

LIMIT 10;

SELECT title

FROM film

WHERE SUBSTRING(description, 3, 6) = ‘Action’;

## NOW()

- 현재 날짜와 시간을 반환

SELECT NOW() AS current_date_time;

## CURDATE()

- 현재 날짜를 반환

SELECT CURDATE() AS cureent_date;

## CURTIME()

- 현재 시간을 반환

SELECT CURTIME() AS current_time;

## DATE_ADD(date, INTERVAL unit)

- 날짜에 간격을 추가
- 예 : rental 테이블에서 각 대여 시작 날짜 (rental_date)에 7일 추가
- 년: YEAR, 달: MONTH, 일: DAY, 시간: HOUR, 분: MINUTE, 초:SECOND

SELECT rental_date, DATE_ADD(renta_date, INTERVAL 7 DAY) AS return_date

FROM rental

LIMIT 10;

## DATE_SUM(date, INTERVAL unit)

- 날짜에 간격을 뺌

SELECT rental_date, DATE_SUB(rental_date, INTERVAL 7 DAY) AS earlier_date

FROM rental

LIMIT 10;

## EXTRACT(field FROM source)

- SQL의 EXTRACT함수는 날짜 필드에서 특정 부분(예: 년,월,일 등)을 추출 할 때 사용
- EXTRACT(field FROM source)
    - 모든 결제 레코드에서 년도만 추출
        
        SELECT EXTRACT(YEAR FROM payment_date) AS payment_year
        
        FROM payment;
        
    - 각 월별 결제 횟수 세기
        
        SELECT EXTRACT (MONTH FROM payment_date) AS payment_month, COUNT(*)
        
        FROM payment
        
        GROUP BY payment_month;
        
- YEAR, MONTH, DAY는 달력 날짜를, HOUR, MINUTE, SECOND 는 시간을 추출

## YEAR(), MONTH(), DAY(), HOUR(), MINUTE(), SECOND()

- EXTRACT와 유사한 작업 수행
    
    SELECT YEAR(payment_date) AS payment_year
    
    FROM payment;
    

## DAYOFWEEK()

- 요일을 반환(일요일 = 1 , 월요일 =2, … 토요일 =7)
    
    SELECT DAYOFWEEK(payment_date) AS payment_dayofweek, COUNT(*)
    
    FROM payment
    
    GROUP BY payment_dayofweek;
    

## TIMESTAMPDIFF

- 두 날짜 또는 시간 값 사이의 차이를 계산
- TIMESTAMPDIFF(unit, start_datetime, end_datetime)
- unit : 반환할 시간 단위
    - 대여와 반납 사이의 일수 차이
        
        SELECT TIMESTAMPDIFF(DAY, rental_date, returen_date) AS rental_days
        
        FROM rental
        
        LIMIT 5;
        

## DATE_FORMAT()

- 날짜 또는 시간 데이터를 특정 형식의 문자열로 변환
- DATE_FORMAT(date, format)
- 지시자
    - %Y : 4자리 연도 표시
    - %y : 연도의 마지막 두 자리 표시
    - %M : 영문 월 이름 표시
    - %m : 월을 두 자리 숫자로 표시
    - %c : 월을 한 자리 숫자로 표시
    - %D : 일을 두 자리 숫자와 영문 접미사로 표시(1st, 21th)
    - %d : 일을 두 자리 숫자로 표시
    - %H : 시간을 24시간 형식의 두 자리 숫자로 표시
    - %h : 시간을 12시간 형식의 두 자리 숫자로 표시
    - %l : 시간을 12시간의 형식의 한 자리 또는 두 자리 숫자로 표시
    - %i : 분을 두 자리 숫자로 표시
    - %s : 초를 두 자리 숫자로 표시
        - 렌탈 날짜를 ‘YYYY-MM-DD’형식으로 변환
            
            SELECT
            
            rental_date, DATE_FORMAT(rental_date, ‘%y-%m-%d %l’) AS formatted_rental_day
            
            FROM rental
            
            LIMIT 5;
            

## ABS(number)

- 절대값을 반환
    
    SELECT ABS(-payment.amount) AS absolute_amount
    
    FROM payment
    
    LIMIT 10;
    

## CEIL(number)

- 올림
    
    SELECT CEIL(amount) AS ceiling_amount
    
    FROM payment
    
    LIMIT 10;
    

## FLOOR(number)

- 내림
    
    SELECT FLOOR(amount) AS floor_amount
    
    FROM payment
    
    LIMIT 10;
    

## ROUND(number, decimals)

- 특정 소수점 자리수로 반올림
    
    SELECT ROUND(amount, 2) AS rounded_amount
    
    FROM payment
    
    LIMIT 10;
    

## SQRT(number)

- 제곱근을 반환
    
    SELECT SQRT(length) AS sqrt_length
    
    FROM film
    
    LIMIT 5;
    

# SELECT

- SELECT → 테이블을 조회하는 기본 쿼리임

## LIMIT

- 결과 중 일부 데이터만 가져오기

SELECT * FROM film LIMIT 10; ← 최상위 데이터 10개 불러오기

SELECT * FROM film WHERE length > 50 LIMIT 1; ← 조건에 맞는 데이터 중 최상위 데이터 1개 불러오기

## COUNT

- 결과 수 세기 (데이터 행의 수 세기)

SELECT COUNT(*) FROM 테이블이름; ← 테이블 전체 데이터 수 세기

SELECT COUNT(*) FROM 테이블이름 WHERE 조건문; ← 특정 조건에 맞는 테이블 데이터 수 세기

## DISTINCT

- 중복된 값 없이 출력하기 (유일한 컬럼 출력)

SELECT DISTINCT 컬럼명 FROM 테이블이름; ← 특정 컬럼에 들어가 있는 컬럼값 종류 확인

SELECT DISTINCT 컬럼명 FROM 테이블이름 WHERE 조건문; ← 특정 조건에 맞는 컬럼값 종류 확인

## 집계함수 → SELECT에 바로 적용

- SUM, AVG, MAX, MIN

SELECT SUM(컬럼명) FROM 테이블이름;

SELECT SUM(컬럼명), AVG(컬럼명) FROM 테이블이름 WHERE 조건문;

## GROUP BY

- 특정 컬럼값을 기반으로 그룹핑하기 → ★SELECT 뒤에 GROUP BY랑 동일한 열 입력해줘야함!

SELECT rating, COUNT(*) FROM film WHERE 조건문 GROUP BY rating;

## HAVING

- 집계함수를 가지고 조건비교를 할 때 사용
- GROUP BY 절과 함께 사용

SELECT provider FROM items GROUP BY provider HAVING COUNT(*) >= 100;

- HAVING 절을 포함한 복합검색 예시

SELECT provider, COUNY(*)

FROM items

WHERE provider != ‘스마일배송’

GROUP BY provider

HAVING COUNT(*) > 100

ORDER BY COUNY(*) DESC;

## ORDER BY

- 특정 컬럼값을 기준으로 데이터 정렬하기
    - DESC , ASC

SELECT * FROM film ORDER BY rating DESC

## AS

- 표시할 컬럼명 다르게 하기
- ★FROM 전에 써야함

SELECT COUNT(*) AS total FROM film ← total이라는 컬럼으로 표기

# 외래키 (FOREIGN KEY)

- 두 테이블간 관계에 있어서 데이터의 정확성을 보장하는 제약 조건을 넣는 것임 → 데이터 무결성
- 제약 조건이 들어가기 때문에 꼭 필요한 경우에만 생성
- 데이터 삽입, 데이터 삭제 시 제약 조건에 위배된다면 에러 발생
- 생성

FOREIGH KEY (컬럼명) REFERENCES 참조테이블 (참조컬럼명);

# JOIN 구문

- 두 개 이상의 테이블로부터 필요한 데이터를 연결해 하나의 포괄적인 구조로 결합
    - INNER JOIN(일반적 JOIN) : 두 테이블에 해당 필드값이 매칭되는 레코드만 가져옴
    - OUTER JOIN
        - LEFT OUTER JOIN : 왼쪽 테이블에서 모든 레코드, 오른쪽 테이블에서 매칭되는 레코드만
        - RIGHT OUTER JOIN : 왼쪽 테이블에서 매칭되는 레코드만, 오른쪽 테이블에서 모든 레코드

## INNER JOIN(JOIN)

- ON 절의 조건이 일치하는 결과만 출력

FROM 테이블1 INNER JOIN 테이블2 ON 매칭조건

- 예시

SELECT * FROM items INNER JOIN ranking ON ranking.item_code = items.teim_code

WHERE ranking.main_category = “ALL”; ← ranking.main_category : ranking테이블의 main_category컬럼

- 테이블 이름 다음에 새로운 이름을 쓰면 AS용법과 동일하게 사용 (일반적으로 이렇게 많이 씀)

SELECT * FROM items a INNER JOIN ranking b ON a.item_code = b.item_code WHERE b.main_category = “ALL”

### LEFT OUTER JOIN → 처음으로 오는게 왼쪽 테이블

SELECT * FROM customer_table C LEFT OUTER JOIN order_table O ON C.customer_id = O.customer_id

### RIGHT OUTER JOIN → 처음으로 오는게 왼쪽 테이블

SELECT * FROM customer_table C RIGHT OUTER JOIN order_table O ON C.customer_id = O.customer_id

# Subquery

- SQL문 안에 포함되어 있는 SQL 문
    - SQL문 안에서 괄호()를 사용해서 서브쿼리문을 추가 할 수 있음
- 테이블과 테이블간 검색 시 검색 범위를 좁히는 기능에 주로 사용

### 사용법

- JOIN은 출력 결과에 여러 테이블의 열이 필요한 경우 유용
- 대부분의 서브쿼리는 JOIN문으로 처리가 가능함
    
    예시) 서브카테고리가 ‘여성신발’인 상품 타이틀만 가져오기
    
    1. JOIN 사용
        
        SELECT title
        
        FROM items
        
        INNER JOIN ranking ON items.tiem_code = ranking.item_code
        
        WHERE ranking.sub_catogory = ‘여성신발’
        
    2. 서브쿼리 사용
        - 컬럼값 IN 서브쿼리 출력값 → 컬럼값과 서브쿼리 값이 같을 때
            
            SELECT title
            
            FROM items
            
            WHERE item_code IN
            
            (SELECT item_code FROM ranking WHERE sub_category = ‘여성신발’);
            
    
    참고 ) 다양한 서브쿼리 삽입 위치
    
    - 비교에 사용
        
        SELECT category_id, COUNT(*) AS film_count
        
        FROM film_category
        
        WHERE film_category.category_id >
        
        (SELECT category.category_id FROM category
        
        WHERE [category.name](http://category.name) = ‘Comedy’)
        
        GROUP BY film_category.category_id
        
    - FROM절에 사용
        
        SELECT 
        
        a, b, c
        
        FROM
        
        (SELECT * FROM atoz_table)
        

### 예제

- 평균 결제 금액보다 많은 결제를 한 고객
    
    SELECT first_name, last_name
    
    FROM customer
    
    WHERE customer_id IN (
    
    SELECT cunstomer_id
    
    FROM payment
    
    WHERE amount > (SELECT AVG(amount) FROM payment)
    
    );
    

### GROUP BY가 있는 서브쿼리

- 평균 결제 횟수보다 많은 결제를 한 고객 (first_name, last_name은 customer 테이블에 있음. 연결키는 customer_id)
    
    SELECT first_name, last_name
    
    FROM customer
    
    WHERE customer_id IN (
    
    SELECT customer_id
    
    FROM payment
    
    GROUP BY customer_id
    
    HAVING COUNT(*) > (
    
    SELECT AVG(payment_count)
    
    FROM (
    
    SELECT COUNT(*) AS payment_count
    
    FROM payment
    
    GROUP BY customer_id
    
    ) AS payment_counts
    
    )
    
    );
    

### 최대값을 가진 행 찾기

- 가장 결제를 많이 한 고객
    
    SELECT first_name, last_name
    
    FROM customer
    
    WHERE customer_id = (
    
    SELECT customer_id
    
    FROM (
    
    SELECT customer_id, COUNT(*) AS payment_count
    
    FROM payment
    
    GROUP BY customer_id
    
    ) AS payment_counts
    
    ORDER BY payment_count DESC
    
    LIMIT 1
    
    );
    

### 조인과 서브쿼리 동시 사용

- 각 고객에 대해 자신이 대여한 평균 영화 길이보다 긴 영화들의 제목
    
    SELECT C.first_name, C.last_name, F.title
    
    FROM customer C
    
    JOIN rental R on R.customer_id = C.customer_id
    
    JOIN inventory I ON I.inventory_id = R.inventory_id
    
    JOIN film F ON F.film_id = I.film_id
    
    WHERE F.length > (
    
    SELECT
    
    FROM film FIL
    
    JOIN inventory INV ON INV.film_id = FIL.film_id
    
    JOIN rental REN ON REN.inventory_id = INV.inventory_id
    
    WHERE REN.customer_id = C.customer_id
    
    );
    

### 상관 서브커리(서브쿼리가 외부 변수를 참조)

- 각 고객에 대해 자신이 결제한 평균 금액보다 큰 결제를 한 경우 결제 정보를 찾음
    
    SELECT P.customer_id, P.amount, P.payment_date
    
    FROM payment P
    
    WHERE P.amout > (
    
    SELECT AVG(amonut)
    
    FROM payment
    
    WHERE customer_id = P.customer_id
    
    );
    

# 집합연산 : UNION, UNION ALL, INTERSECT, EXCEPT

## UNION

- 두 개 이상의 SELECT문의 결과 집합을 결합
- **중복된 행 제거**
- 각 SELECT문의 열은 같은 순서를 가져야 하며, 유사한 데이터를 가져야함
    
    SELECT film_id FROM film
    
    UNION
    
    SELECT film_id FROM inventory;
    

## UNION ALL

- 두 개 이상의 SELECT문의 결과 집합을 결합
- **중복된 행 포함**
- 각 SELECT문의 열은 같은 순서를 가져야 하며, 유사한 데이터를 가져야함
    
    SELECT film_id FROM film
    
    UNION ALL
    
    SELECT film_id FROM inventory;
    

## INTERSECT

- 두개 이상의 SELECT문의 결과 집합의 **교집합을 반환**
- 각 SELECT문의 열은 같은 순서를 가져야 하며, 유사한 데이터를 가져야함
    
    SELECT film_id FROM film
    
    INTERSECT
    
    SELECT film_id FROM inventory;
    

## EXCEPT

- 두 SELECT문의 결과 집합의 **차집합을 반환**
- 각 SELECT문의 열은 같은 순서를 가져야 하며, 유사한 데이터를 가져야함
    
    SELECT film_id FROM film
    
    EXCEPT
    
    SELECT film_id FROM inventory;
    

# 트랜잭션, 커밋, 롤백

## 트랜잭션

- 트랜잭션 : 하나 이상의 SQL문을 포함하는 작업의 논리적 단위
- 트랜잭션은 단일 논리적 작업 단위로 수행되는 연산의 순서

## COMMIT

- 모든 변경 사항을 저장
- COMMIT 바로 다음에 새 트랜잭션 시작

## ROLLBACK

- 일부, 모든 변경 사항을 취소
- 현재 트랜잭션을 종료, 새 트랜잭션 시작

### 예제

START TRANSACTION;

UPDATE customer

SET first_name = “DAVE”

WHERE customer_id = 1;

COMMIT; ← CMMIT 후에는 복원 불가능, ROLLBACK해도 소용없음

# VIEW SQL

## VIEW

- 실제 테이블을 기반으로 한 가상 테이블
- 복잡한 쿼리를 단순화하고 데이터의 특정 부분에만 접근 허용 가능
- 실제 테이블에서 수정 → VIEW 에도 자동 반영됨

### VIEW 생성

CREATE VIEW view_name AS

SELECT column1, column2, …

FROM table_name

WHERE condition;

### VIEW 수정

CREATE OR REPLACE VIEW view_name AS

SELELCT column1, column, …

FROM table_name

WHERE condition;

### VIEW 삭제

DROP VIEW view_name;

# WITH, CASE WHEN SQL

## WITH절 (Common Table Expression)

- 일시적인 결과 세트를 만들어 SELECT, INSERT, UPDATE, DELETE 문에서 참조할수 있게 해줌
- 단일 SQL 쿼리 내에서만 사용 가능
- SQL쿼리가 종료하면 자동 삭젲됨
- 복잡한 쿼리를 쉽게 분해할 수 있음
    
    WITH cte_name AS (
    
    )
    

### 예시

재고에 있는 모든 영화의 이름을 가져오고 싶을 때

WITH FilmInventory AS (

SELECT DISTINC film_id FROM inventory

)

SELECT f.film_id, f.title

FROM film f

JOIN FilmInventory fi ON f.film_id = fi.film_id;

## CASE WHEN

- SQL 쿼리에 조건 로직을 추가할 수 있음
- IF-ELSE와 비슷한 역할
    
    CASE
    
    WHEN condition1 THEN result1
    
    WHEN condition2 THEN result2
    
    …
    
    ELSE result
    
    END
    

### 예제

데이터베이스의 영화를 렌탈 비용을 기준으로 카테고리화 하고 싶은 경우

SELECT title,

CASE

WHEN rental_rate < 1 THEN ‘cheap’

WHEN rental_rate BETWEEN 1 AND 3 THEN ‘moderate’

ELSE ‘expensive’

END AS PriceCategory

FROM film;

# GROUP_CONCAT

- 문자열 집계 함수
- 그룹 내의 여러 행을 하나의 문자열로 결합
    
    GROUP_CONCAT(expression)
    
    GROUP_CONCAT(expression SEPARATOR
    
    separator_string)
    
- expression : 결합할 컬럼 혹은 표현식
- seperator_string : 행을 연결할 때 사용할 문자열 (기본값은 쉼표)

### 예제

고객별로 렌탈한 영화의 제목을 문자열로 만들기

SELECT

c.customer_id,

CONCAT(c.first_name, ‘ ‘, c.last_name) AS customer_name,

GROUP_CONCAT(f.title ORDER BY f.totle ASC) AS rented_moives

FROM customer c

JOIN rental r ON c.customer_id = r.customer_id

JOIN inventory i ON r.inventory_id = i.inventory_id

JOIN film f ON i.film_id = f.film_id

GROUP BY

c.customer_id

LIMIT 5;

# 윈도우 함수

## MySQL과 윈도우 함수

- 윈도우 함수는 SQL 쿼리 내에서 데이터 집합을 세분화하여 각 부분에 대한 계산을 수행하는 함수
- 데이터 순위를 매기거나, 집계, 이동 평균, 누적 합 등을 계산할 수 있으며, 기본 집계보다 더 유연한 데이터분석이 가능함
    - 윈도우(데이터의 부분집합)내에서 작동하며 각각의 행에 대해 결과를 반환하되, 전체 쿼리 결과의 컨텍스트 내에서 실행됨

## 윈도우 함수

- 행 기준 연산 : 각 행에 대해 연산을 수행하면서 원본 행의 구조를 유지 → 각 행에 대한 상세정보를 보존하면서 계산 수행 가능
- 부분 데이터 집합 사용 : 특정 윈도우 내에서 연산을 수행하고 이는 PARTITION BY 절을 통해 세분화 될 수 있음
- 다양한 연산 지원 : 순위 매기기, 누적 집계, 이동 평균 등
- 원본 데이터셋 변경 없음
- 집계와 상세 데이터 동시 제공

## GROUP BY

- 그룹 기준 연산
- 데이터 집약
- 단순 집계 연산 제한
- 원본 데이터셋 축소
- 상세 데이터 제공은 불가능함

## RANK, DENSE_RANK, ROW_NUMBER

- 모두 OVER 절과 함께 사용되며, ORDER BY 절을 통해 순위를 매길 기준 컬럼을 지정함

### RANK()

- 순위를 매김
- 동일한 값이 있을 경우 같은 순위를 부여
- 다음 순위 건너뜀
    
    RANK() OVER (ORDER BY column_name [ASC|DESC])
    

### DENSE_RANK()

- 순위를 매김
- 동일한 값이 있을 경우 같은 순위를 부여
- 다음 순위 건너뛰지 않음
    
    DENSE_RANK() OVER (ORDER BY column_name [ASC|DESC])
    

### ROW_NUMBER()

- 순위와 상관없이 각 행에 고유한 번호를 부여함
    
    ROW_NUMBER() OVER (ORDER BY column_name [ASC|DESC])
    

## PARTITION BY, ORDER BY, ROWS/RANGE

### PARTITION BY

- 특정 컬럼을 기준으로 데이터를 부분집합으로 분할함
- 부분집합들은 윈도우 함수의 계산 범위를 정의
- 각 부분집합 내에서 독립적으로 함수가 계산됨
    
    FUNCTION() OVER (PARTITION BY column1, column2, … )
    

### ORDER BY

- 각 부분집합 내에서 데이터 행들의 정렬 순서를 지정함
- 순위나 누적합계 같은 윈도우 함수의 결과에 직접적인 영향 미침
    
    FUNCTION() OVER (PARTITION BY column1, column2 …
    
    ORDER BY column3 [ASC|DESC|)
    

### ROWS/RANGE

- ORDER BY와 함께 사용되어, 윈도우 함수가 특정 행위 범위 내에서만 데이터를 계산하도록 지정
- ROW는 물리적 행의 위치를 기준으로 범위를 설정
- RANGE는 정렬 키의 값에 따라 범위를 설정

### ROWS/RANGE와 BETWEEN 옵션

- UNBOUNDED PRECEDING : 파티션의 첫 행부터 시작
- UNBOUNDED FOLLOWING : 파티션의 마지막 행까지
- CURRENT ROW : 현재 행 포함
- n PRECEDING/FOLLOWING : 현재 행에서 n행 앞이나 뒤
    - 예시
        
        FUNCTION() OVER (PARTITION BY column1, column2 …
        
        ORDER BY column3
        
        ROWS BETWEEN 1 PRECEDING AND 1 FOLOOWING) AS cumulative_rental_date
        

### FUNCTION()

- FUNCTION OVER (PARTITION BY customer_id ORDER BY rental_date
    
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_rentals
    
- FUNCTION 부분에 다음 함수 사용
    - COUNT()
    - SUM()
    - AVG()
    - MIN()
    - MAX()
    - ROW_NUMBER()
    - RANK()
    - DENSE_RANK()
    - LEAD() - 앞에 위치한 데이터 반환
    - LAG() - 뒤에 위치한 데이터 반환
    - FIRST_VALUE() - 윈도우 내 첫번째 값
    - LAST_VALUE() - 윈도우 내 마지막 값
    - 

## ROWS/RANGE 차이

### ROWS

- 물리적인 행의 위치를 기준으로 윈도우를 정의
- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW는 현재 행을 포함하여 그 이전의 모든 행들을 집계 범위로 삼음
- 각 행을 고유하게 취급하며, 정렬된 순서에 따라 정확하게 해당 위치의 행들만 집계에 포함

### RANGE

- 현재 행의 정렬 기준 값(ORDER BY 에 지정된 값)과 동일한 값을 가진 모든 행을 윈도우에 포함
- RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW는 현재 행의 정렬 키 값과 동일한 모든 행을 포함하여 그 이전의 모든 행들까지를 집계 범위로 삼음
- 만약 정렬 키 값이 중복되는 경우 RANGE는 그 값에 해당하는 모든 행을 같은 그룹으로 간주하여 집계
    
    =트랜스포머랑 비슷하다고 생각하면됨, 누적합을 각행에 다 뿌림
    

![image.png](attachment:659fe65f-0857-440e-998b-5b7ec14bf6c0:image.png)

## SUM(revenue)와 SUM(revenue) OVER() 의 차이

### SUM(revenue)

- 일반적인 집계함수. 전체 쿼리 결과 또는 그룹화된 데이터 세트의 합계를 계산
- GROUP BY 절과 함께 사용될 경우 각 그룹의 revenue 합계를 계산
- GROUP BY 절이 없는 경우 전체 쿼리 결과의 revenue 합계를 반환

### SUM(revenue) OVER()

- OVER()는 특정 파티션 또는 정렬없이 전체 결과 세트에 대해 함수를 적용하라는 의미
- 쿼리가 반환하는 각 행에 대해 전체 결과 집합의 revenue 합계를 계산
- 모든 행에서 같은값을 반환하며 각 행은 전체 revenue의 합계를 알 수 있음

## LEAD(), LAG(), FIRST_VALUE

### LEAD

- LEAD(column, n, defalut) : 현재 행을 기준으로 n행 뒤의 값을 가져옴. n 미지정시 기본값은 1. 뒤에 행이 없으면 default값을 반환
    
    LEAD(column, n, default값) OVER (ORDER BY column_name [ASC|DESC])
    

### LAG

- LAG(column, n, defalut) : 현재 행을 기준으로 n행 앞의 값을 가져옴. n 미지정시 기본값은 1. 앞에 행이 없으면 default 값을 반환

### FIRST_VALUE

- 파티션된 윈도우에서 첫 번째 값을 가져옴
    
    FIRST_VALUE(column) OVER (PARTITION BY column_name ORDER BY column_name [ASC|DESC])
    

### LAST_VALUE

- 파티션된 윈도우에서 마지막 값을 가져옴
    
    LAST_VALUE(column) OVER (PARTITION BY column_name ORDER BY column_name [ASC|DESC]
    
    RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    

## PERCENT_RANK(), CUME_DIST(), NTILE()

### PERCENT_RANK

- 행의 백분위 순위를 계산
- 결과는 0부터 1사이 값
    
    PERCENT_RANK() OVER (ORDER BY column_name [ASC|DESC])
    

### CUME_DIST

- 행의 누적 분포를 계산
- 결과는 0부터 1사이 값
    
    CUME_DIST() OVER (ORDER BY column_name [ASC|DESC])
    

### NTILE(n)

- 행을 n개의 그룹으로 분할
- 각 그룹에는 같은 수의 행의 포함

NTILE(n) OVER (ORDER BY column_name [ASC|DESC])

# **5. SQL DCL(Data Control Language) 이해 및 참고**

SQL DCL 명령은 MySQL 관리자(데이터베이스 관리자)를 위한 명령이므로, 가볍게 참고하시면 됩니다.

### **MySQL 접속하기**

> MySQL에 접속하려면 터미널(명령 프롬프트)에서 다음 명령을 사용합니다.
> 

```bash
mysql -u 사용자이름 -p
```

- `u`: 접속할 사용자 이름을 지정합니다.
- `p`: 비밀번호 입력을 요구합니다.

**예시:**

- `root` 사용자로 접속하기:
    
    ```bash
    mysql -u root -p
    ```
    
    비밀번호 입력 프롬프트가 나타나면 `root` 사용자의 비밀번호를 입력합니다.
    

### **5.1 MySQL 사용자 관리**

MySQL에서 사용자 계정을 관리하는 기본 명령어입니다.

### **1. 사용자 확인**

> 현재 MySQL에 등록된 사용자 목록을 확인합니다.
> 

```sql
# mysql-u root-p
mysql> USE mysql;
mysql>SELECT host,userFROMuser
```

### **2. 사용자 추가**

> 새로운 사용자를 생성합니다.
> 
- **로컬에서만 접속 가능한 사용자 생성**
    
    ```sql
    mysql>CREATEUSER 'userid'@'localhost' IDENTIFIEDBY '비밀번호'
    ```
    
- **모든 호스트에서 접속 가능한 사용자 생성**
    
    ```sql
    mysql>CREATEUSER 'userid'@'%' IDENTIFIEDBY '비밀번호'
    ```
    

### **3. 사용자 비밀번호 변경**

> 기존 사용자의 비밀번호를 변경합니다.
> 

```sql
mysql>SET PASSWORDFOR 'userid'@'호스트'= '신규비밀번호'
```

### **4. 사용자 삭제**

> 사용자를 삭제합니다.
> 

```sql
mysql>DROPUSER 'userid'@'호스트'
```

### **5.2 MySQL 접속 권한 설정**

사용자의 접속 권한을 설정하는 방법입니다.

### **1. 권한 확인**

> 특정 사용자가 가진 권한을 확인합니다.
> 

```sql
mysql> SHOW GRANTSFOR 'userid'@'호스트'
```

### **2. 접속 허용 설정**

- **로컬에서만 접속 허용**
    
    ```sql
    mysql>GRANTALLON 데이터베이스.테이블TO 'userid'@'localhost'
    ```
    
- **특정 권한만 허용**
    
    ```sql
    mysql>GRANTSELECT,UPDATEON 데이터베이스.테이블TO 'userid'@'호스트'
    ```
    

---

### **옵션 설명**

1. **권한 종류**
    - `ALL`: 모든 권한을 부여합니다.
    - `SELECT`, `UPDATE`: 특정 권한만 부여하여 권한을 제한할 수 있습니다.
    
    **예시:**
    
    ```sql
    mysql>GRANTINSERT,UPDATE,SELECTON*.*TO 'username'@'localhost'
    ```
    
2. **데이터베이스 및 테이블 지정**
    - `데이터베이스.테이블`: 특정 데이터베이스의 특정 테이블에만 권한을 부여합니다.
    - `.*`: 모든 데이터베이스의 모든 테이블에 대한 권한을 의미합니다.
3. **사용자명과 호스트**
    - `'userid'@'호스트'` 형식으로 지정합니다.
    - `'localhost'`: 로컬 호스트에서의 접속을 의미합니다.
    - `'%'`: 모든 호스트에서의 접속을 허용합니다.

**참고:** 권한 변경 후 변경 사항을 적용하려면 `FLUSH PRIVILEGES;` 명령을 실행할 수 있습니다.

```sql
mysql> FLUSHPRIVILEGES
```

# 7.인덱스 (MySQL INDEX)

- 데이터베이스 분야에 있어서 테이블에 대한 동작의 속도를 높여주는 자료 구조
- 어떤 데이터를 인덱스로 만드느냐에 따라 방대한 데이터의 경우 성능에 큰 영향을 미칠수 있음

## 7.1 인덱스 종류

- 클러스터형 인덱스 : 영어 사전과 같은 형태로 데이터를 재정렬하여 저장한다고 생각하면 됨
    - 테이블의 데이터를 실제로 재정렬하여 디스크에 저장
    - 테이블에 PRIMARY KEY로 정의한 컬럼이 있을 경우, 자동 생성
    - 한 테이블당 하나의 클러스터형 인덱스만 가질 수 있음
- 보조 인덱스 : 데이터는 그대로 두고, 일반 책 뒤에 있는 <찾아보기> 와 같은 형태가 만들어진다고 생각하면 됨
    - 클러스터형 인덱스와는 달리 데이터를 디스크에 재정렬하지 않고, 각 데이터의 위치만 빠르게 찾을 수 있는 구조로 구성
    - 보조 인덱스를 저장하는 데 필요한 디스크 공간은 보통 테이블을 저장하는 데 필요한 디스크 공간보다 작음
        - 인덱스는 키-필드만 갖고 있고, 나머지 세부 테이블 컬럼 정보는 가지고 있지 않기 때문

### userTBI 테이블 생성

CREATE TABLE userTbl (

userID CHAR(8) NOT NULL PRIMARY KEY,

name VARCHAR(10) NOT NULL,

birthYear INT NOT NULL,

addr CHAR(2) NOT NULL,

mobile1 CHAR(3),

mobile2 CHAR(8),

height SMALLINT,

mDate DATE

);

### 인덱스 확인

SHOW INDEX FROM userTbl

- Key_name 이 PRIMARY로 된 것은 클러스터형 인덱스를 의미
- Column_name이 userID 임을 확인할 수 있음
- (참고) 주요 인덱스 컬럼
    - Table : The name of the table
    - Non_unique : 0 if the index cannot contain duplicates, 1 if it can
    - Key_name : The name of the index. if the index is the primary key, the nma is always PRIMARY
    - Seq_in_index : The column sequence number in the index, starting with 1
    - Column_name : The column name
    - Collation : How the column is sorted in the index. This can hava values A (ascending) or Null (not sorted)
    - Caridinality : An estimate of the number of unique values in the index
    - index_type : the index method used (BTREE, FULLTEXT, HASH, RTREE)

### buyTbl 테이블 구조

CREATE TABLE buyTbl (

num INT AUTO_INCREMENT NOT NULL PRIMARY KEY,

userID CHAR(8) NOT NULL,

prodName CHAR(4),

groupName CHAR(4),

price INT NOT NULL,

amount SMALLINT NOT NULL,

FOREIGN KEY (userID) REFERENCES userTbl (userID)

);

- Key_name 이 PRIMARY가 아닌것은 보조 인덱스를 의미
- foreign key로 설정된 컬럼이 인덱스가 없다면, 보조 인덱스를 자동 생성

### 참고 : 테이블 변경

ALTER TABLE userTbl ADD [CONSTRAINT TESTDate] UNIQUE(mDate);

- ALTER TABLE 테이블 이름 ADD [CONSTRAINT 제약조건명] UNIQUE(컬럼명)
    - 테이블에 특정 컬럼에 duplicate 값이 나오지 않도록 제약조건 추가하기

### 참고 : UNIQUE 제약을 넣으면, 보조 테이블이 만들어짐

SHOW INDEX FROM userTbl

## 7.2 인덱스 생성 및 삭제

- 인덱스를 필요에 따라 생성/삭제 가능

### 생성된 테이블에 인덱스 추가하기

- 기본 문법
    - CREATE INDEX 인덱스명 ON 테이블명 (column1, column2, …);
    - ALTER TABLE 테이블명 ADD INDEX 인덱스명 (column1, column2, …);
- 생성된 테이블에 인덱스 추가 예제 (CREATE INDEX 사용)

CREATE INDEX idx_name ON userTbl (name);

- 인덱스 확인

SHOW INDEX FROM userTbl;

- 생성된 테이블에 인덱스 추가 예제(ALTER TABLE 사용)

ALTER TABLE userTbl ADD INDEX idx_addr (addr);

- 인덱스 확인

SHOW INDEX FROM userTbl;

## 7.3 테이블 생성하며 인덱스도 함께 만들기

- 기본 문법
    - INDEX<인덱스명> (컬럼명1, 컬럼명2)
    - UNIQUE INDEX<인덱스명>(컬럼명) —> 항상 유일해야함
        - UNIQUE INDEX의 경우 컬럼명은 유일한 값을 가지고 있어야 함

## 7.4 인덱스 삭제

- 기본 문법
    - ALTER TABLE 테이블명 DROP INDEX 인덱스명

ALTER TABLE userTbl DROP INDEX idx_userTbl_name

ALTER TABLE userTbl DROP INDEX idx_userTbl_addr

## 8. python 연동
# pymysql 모듈 이해

## pymysql라이브러리 소개 및 설치

- mysql을 python에서 사용할 수 있는 라이브러리
    - pymysql 라이브러리 이외에도 MySQLdb(MySQL-pytion), MYSQL connector등 다양한 라이브러리 존재

## pymysql 설치

!pip install pymysql

## pymysql 기본 코드 패턴

### 1. PyMySql모듈 import

import pymysql

### 2. pymysql.connec() 메소드를 사용하여 MySQL에 연결

- 호스트명, 포트, 로그인, 암호, 접속할 DB등을 파라미터로 지정
    - host : 접속할 mysql server 주소
    - port : 접속할 mysql server 의 포트 번호
    - user : mysql ID
    - passwd : mysql ID의 암호
    - db : 접속할 데이터베이스
    - charset=’utf8’ : mysql에서 select하여 데이터를 가져올 때 한글이 깨지지 않도록 연결 설정에 넣어줌
    
    db = pymysql.connect(
    
    host=’localhost’,
    
    port=3306,
    
    user=’root’,
    
    passwd=’0000’,
    
    db=’ecomeenrce’,
    
    charset=’utf8’)
    
    ### 3. MySQL 접속이 성공하면, Connection 객체로부터 cursor() 메서드를 가져옴
    
    ### 4. Cursot 객체의 execute()메서드를 사용하여 SQL문장을 DB 서버에 전송
    
    ### 5. 테이블 생성
    
    - Cursor Object 가져오기 : cursor = db.cursor()
    - SQL실행하기 : cursor.execute(SQL)
    - 실행 mysql 서버에 확정 반영하기 : db.commit()
    
    ecommerce = db.cursor()
    
    ### ex) 테이블 생성
    
    sql = “””
    
    CREATE TANLE product (
    
    PRODUCT_CODE VARCHAR(20) NOT NULL,
    
    TITLE VARCHAR(200) NOT NULL,
    
    ORI_PRICE INT
    
    DISCOUNT_PRICE INT,
    
    DISCOUNT_PERCENT INT,
    
    DELIVEREY VARCHAR(2),
    
    PRIMARY KEY(PRODUCT_CODE)
    
    );
    
    “””
    
    ### 6. SQL 실행(cursor 객체의 excute()메서드를 사용하여 INSERT, UPDATE 혹은 DELETE문장을 DB서버에 보냄)
    
    ecommerce.execute(sql)
    
    ### 7. 삽입,갱신,삭제 등이 모두 끝났으면 Connection 객체의 commit() 메서드를 사용하여 데이터를 Commit
    
    db.commit()
    
    ### 8. Connection 객체의 close() 메서드를 사용하여 DB연결을 닫음
    
    db.close()
    

## 데이터 삽입 패턴

import pymysql

db = pymysql.connect(

host=’localhost’,

port=3306,

user=’root’,

passwd=’0000’,

db=’ecomeenrce’,

charset=’utf8’)

ecommerce = db.cursor()

for index in range(10):

product_code = 215673140 + index + 1

SQL = “””INSERT INTO product VALUES(”””+str(product_code)+”””,

‘스위트바니 여름신상5900원~롱원피스티셔츠/긴팔/반팔’, 23000, 6900, 80, ‘F’

);

“””

cursor.execute(SQL)

db.commit()

db.close()

## 데이터 조회 패턴

- 조회할뿐이니까 commit은 없어도됨
- fetchall() : 쿼리 결과의 모든 행을 가져옴. 결과는 튜플의 튜플로 반환. 각 내부 튜플은 하나의 레코드를 나타냄
    
    cursor.execute(”SELECT * FROM MyTable”)
    
    row = cursor.fetchall()
    
    for row in rows:
    
    print(row)
    
    → fetchall()은 테이블의 모든 행을 반환함
    
- fetchone() : 쿼리 결과의 다음 행을 가져옴. 결과는 하나의 튜플로 반환. 튜플의 각 요소는 각 필드를 나타냄. 더이상 가져올 행이 없으면 None을 분환
    
    cursor.execute(”SELECT * FROM MyTable”)
    
    row = cursor.fetchone()
    
    while row is not None:
    
    print(row)
    
    row = cursor.fetchone()
    
     → fetchone()은 테이블의 한 행씩 순차적으로 반환
    
- fetchmany(size) : 쿼리 결과의 다음 행들을 가져옴. size는 가져올 행의 수를 지정, 결과는 튜플의 튜플로 반환
    
    cursor.execute(”SELECT * FROM MyTable”)
    
    rows = cursor.fetchmany(5)
    
    while rows:
    
    pritn(rows)
    
    rows = cursor.fetchmany(5)
    
     → ftchmany(5)는 테이블의 다섯행씩 순차적으로 반환
    

import pymysql

db = pymysql.connect(

host=’localhost’,

port=3306,

user=’root’,

passwd=

cursor = db.cursor()

SQL = “SELECT*FROM product”

cursor.execute(SQL)

row = cursor.fetchmany(2)

for row in rows:

print(row)

row = cursor.fetchone()

print(row)

### 데이터 수정

import pymysql

db = pymysql.connect(

host=’localhost’,

port=3306,

user=’root’,

passwd=’0000’,

db=’ecomeenrce’,

charset=’utf8’)

ecommerce = db.cursor()

SQL = “””

UPDATE product SET

TITLE = ‘달리샵린넨원피스 비스튀에 썸머 가디건 코디전’,

ORI_PRICE = 33000,

DISCOUNT_PRICE = 9900,

DISCOUNT_PERCENT = 80

WHILE PRODUCT_CODE = ‘214675423’

“””

cursor.execute(SQL)

db.commit()

db.close()

### 데이터 삭제

import pymysql

db = pymysql.connect(

host=’localhost’,

port=3306,

user=’root’,

passwd=’0000’,

db=’ecomeenrce’,

charset=’utf8’)

ecommerce = db.cursor()

SQL = “”” DELETE FROM product WHERE PRODUCT_CODE=’2139513’; “””

db.commit()

db.close()

# 데이터 수집부터 저장까지 파이썬과 MYSQL

### 파이썬 크롤링

!pip install bs4

import requests

from bs4 import BeautifulSoup

for page_num in range(10):

if page_num == 0:

res = requests.get(’https://davelee-fun.github.io/’)

else:

res = requests.get(’https://davelee-fun.gitbub.io/page’ + str(page_num +1))

soup = BeaurifulSoup(res.content, ‘html.parser’)

data = soup.select(’div.card-body’)

for item in data:

category = item.select_one(’h2.card-title’).get_text().replace(’관련 상품 추천’, ‘’).strip()

product = item.select_one(’h4.card-text’).get_text().replace(’상품명:’, ‘’).strip()

print(category, product)

### table schema 는 SQL로 작성하는게 일반적 → WORKBENCH에서 작성

DROP DATABASE IF EXISTS ecommerce;

CREATE DATABASE ecommerce;

USE ecommerce;

CREATE TABLE teddyproducts (

ID INT UNSIGNED NOT NULL AUTO_INCREMENT,

TITLE VARCHAR(200) NOT NULL,

CATEGORY VARCHAR(20) NOT NULL,

PRIMARY KEY(OD)

);

### 크롤링 + mysql 저장

import pymysql

db = pymysql.connect(

host=’localhost’,

port=3306,

user=’root’,

passwd=’0000’,

db=’ecomeenrce’,

charset=’utf8’)

ecommerce = db.cursor()

for page_num in range(10):

if page_num == 0:

res = requests.get(’https://davelee-fun.github.io/’)

else:

res = requests.get(’https://davelee-fun.gitbub.io/page’ + str(page_num +1))

soup = BeaurifulSoup(res.content, ‘html.parser’)

data = soup.select(’div.card-body’)

for item in data:

category = item.select_one(’h2.card-title’).get_text().replace(’관련 상품 추천’, ‘’).strip()

product = item.select_one(’h4.card-text’).get_text().replace(’상품명:’, ‘’).strip()

print(category, product)

for index in range(10):

product_code = 215673140 + index + 1

SQL = “””INSERT INTO teddyproducts (TITLE, CATEGORY) VALUES(’”””+product+”””’,’”””+category+”””’);”””

cursor.execute(SQL)

db.commit()

db.close()

### mysql 읽기

import pymysql

db = pymysql.connect(

host=’localhost’,

port=3306,

user=’root’,

passwd=

cursor = db.cursor()

SQL = “SELECT*FROM teddyproducts WHERE CATEGORY = ‘행거도어’;”

cursor.execute(SQL)

rows = cursor.fetchall()

for row in rows:

print(row)

db.close()
