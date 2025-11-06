# 머신러닝 기초

## 머신러닝 프로세스

- 지도학습
    - 분류(이진분류, 다중분류)← 클래스 분류
    - 회귀(ex. 매출액, 방문객 숫자 예측)← 연속형 범주 분류
- 비지도학습 - 군집화, 차원축소
- 강화학습
- 전통적 접근방식 : 데이터 + 규칙 → 결과를 도출 (기존 규칙 base)
- 머신러닝 접근방식 : 데이터 + 해답 → 규칙을 도출 (by 훈련,학습)
- **프로세스**
    - **문제정의 → 라이브러리/데이터 불러오기 → 탐색적 데이터분석 → 데이터 전처리 →  검증 데이터 나누기 → 모델 학습 및 평가 → 예측 및 제출**
- 예측
    - predict 분류/회귀 → 평가지표 : 모든 평가지표
    - predict_proba 분류 → 평가지표 : ROC-AUC : 예측이 **확률**로 나옴

## 탐색적분석 (EDA)

- 데이터 샘플 : train.head()
- 데이터 크기 : train.shape, test.shape
- 자료형 확인 : train.info()
- 타겟 시각화 : train[’Item_Outlet_Sales’].hist()
- 수치형 컬럼 통계값 확인 : train.describe(), test.describe()
- 범주형 컬럼 통계값 확인 : train.describe(include = ‘0’), test.describe(include = ‘object’)
- train 데이터와 test 데이터의 동질성 확인 :
    
    set(train[’열’]) == set(test[’열])
    
- 결측치 확인 : train.isnull().sum(), test.isnull().sum()
- label(target) 별 개수 확인 : train[’income’].value_counts()

## 데이터 전처리

### 결측치 처리(범주형)

**★★★train 데이터에서 행을 삭제하는건 상관없는데**

**★★★test 데이터에서는 행을 마음대로 삭제하면 안됨**

- 결측치가 있는 행 삭제

df = train.dropna()

- 결측치가 있는 특정컬럼 행 삭제 dropna(subset= “행”)

df = train.dropna(subset = [’native.country’,’workclass’])

- 결측치가 있는 특정컬럼 열 삭제 dropna(axis = 1)

df = train.dropna(axis=1)

- 결측치가 많은(채우기불가) 특정컬럼 삭제

df = train.drop([’native.country’, ‘workclass’], axis=1)

- 결측치 채우기
    
    최빈값
    
    train[’workclass’] = train[’workclass’].fillna(train[workclass’].mode()[0])
    
    평균값
    
    fillna.mean()
    
    중앙값으로 채우기
    
    fillna.median()
    
    결측값을 새로운 카테고리로 생성
    
    train[’occupation’] = train(’occupation’].fillna(’X’)
    

**★TEST데이터에서도 똑같이 결측치 전처리 해줘야함**

**★TEST데이터는 미래시점이기 때문에 최빈값 없음. TRAIN 데이터의 최빈값 이용**

### 이상치 처리(age <=0 을 이상치로 판단한 경우)

cond = train[’age’] > 0

train = train[cond]

**★TEST 데이터 행은 삭제하면 안됨**

### 인코딩(문자 데이터를 숫자 데이터로 변형)

y_train = train.pop(”income”)  ← 타겟 분리

cf) pop → 기존 dataframe에서 열을 분리해서 없애고, 새로운 열만 가져오는기능

- 원핫(one-hot) 인코딩 : 문자 데이터에 숫자부여
    
    train_oh = pd.get_dummies(train)
    
    test_oh = pd.get_dummies(test)
    

심화) train 과 test의 열 갯수가 다를 때 합쳐서 원핫인코딩 후 분리

data=pd.concat([’train’, ‘test’], axis=0) ← concat : 합치기

data_oh = pd.get_dummies(data)

train_oh = data_oh.iloc[:len(train)].copy() ← len(train) 기준으로 train과 test 분리

test_oh = data_oh.iloc[len(train):].copy()

- 레이블(label)인코딩 : 사전을 미리 만들어서 하나씩 적용해주는 방법
    
     from sklearn.preprocessing import LabelEncoder
    
    cols = train.select_dtypes(include=’object’).columns ← object 컬럼목록(.columns)만 가져오기
    
    for col in cols: ←필수
    
    le = LabelEncoder()
    
    train[col] = le.fit_transform(train[col]) ← fit : 사전만들기 , transform : 하나씩 적용
    
    test[col] = le.transform(test[col])
    

### 스케일링

- 스케일링 : 기존 범위를 특정 숫자 범위(ex : 0~1)로 조정
- 민-맥스 스케일링 : 모든 값이 0과 1사이로 변경 → for i in range 안해도됨
    
    cols = [’age’, ‘fnlwgt’, ‘education.num’, ‘capital.gain’]
    
    from sklearn.preprocessing import MinMaxScaler
    
    sclaer = MinMaxScaler()
    
    train_copy[cols] = scaler.fit_transform(train_copy[cols])
    
    test_copy[cols] = scaler.tranform(test_copy[cols])
    
- StandardScaler : Z-score 정규화, 평균이 0 표준편차가 1인 표준정규분포로 변경
    
    from sklearn.preprocessing import StandardScaler
    
    sclaer = StandardScaler()
    
    train_copy[cols] = scaler.fit_transform(train_copy[cols])
    
    test_copy[cols] = scaler.tranform(test_copy[cols])
    
- 로버스트 스케일링 : 중앙값과 사분위 값 활용, 이상치 영향 최소화 방법
    
    from sklearn.preprocessing import RobustScaler
    
    sclaer = RobustScaler()
    
    train_copy[cols] = scaler.fit_transform(train_copy[cols])
    
    test_copy[cols] = scaler.tranform(test_copy[cols])
    

★ target 은 문자열이여도 머신러닝 가능

심화) target 도 인코딩 하고싶은 경우

target = y_train.map({’<=50K’ : 0, ‘>50k’ : 1}) or

target = y_train.replace(’<=50K’ , 0).replace(’>50K’, 1)

## 검증데이터 분할

- train 데이터(y값이 이미존재)를 train 데이터와 검증 데이터로 분할 (test 데이터는 y값이 없음)

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(train, y_train, test_size=0.2, random_state=0)

## 머신러닝 학습 및 평가(분류)

- 머신러닝 모델 : 랜덤포레스트, lightGBM
- 머신러닝 학습 및 예측 방법 : 모델불러오기 → 학습 fit(X,y) → 예측 predict(test) , predict_proba() ← 평가지표가 roc-auc일때 주로 사용(확률값으로 나옴)
- 랜덤포레스트
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(random_state=0)
    
    rf.fit(X_tr, y_tr)
    
    pred=rf.predict_proba(X_val)
    
    pred[:10] ← 예측 결과 확인
    
    print(rf.calsses_) ← 클래스 확인
    

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_val, pred[:,1]) ← pred[:,1] : 50K>0 인 확률 구하기위해서

print(’roc_auc:’, roc_auc)

- 정확도(accuracy) 확인
    
    pred = rf.predict(X_val)
    
    pred[:10]
    
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(y_val, pred)
    
    print(’accuracy_score:’, accuracy)
    
- f1 score 확인
    
    from sklearn.metrics import f1_score
    
    f1 = f1_score(y_val, pred, pos_label=’>50K’)
    
    pirnt(’f1_score:’, f1)
    
- LightGBM
    
    import lightbgm as lgb
    
    lgbmc = lgb.LGBMClassfier(random_state=0, verbose=-1)  →  verbose=-1 : 로그내용 숨기기
    
    lgbmc.fit(X_tr, y_tr)
    
    pred = lgbmc.predict_proba(X_val)
    
    roc_auc = roc_auc_score(y_val, pred[:,1])
    
    print(’roc_auc:’, roc_auc)
    
    accuracty = accuracy_score(y_val, pred)
    
    print(’accuracy_score:’, accuracy)
    
    f1 = f1_score(y_val, pred, pos_label=’>50K’)
    
    print(’f1_score:’, f1)
    

### 예측 및 결과 파일 생성

submit = pd.DataFrame({’pred’:pred[:,1]}) ← dataframe : 딕셔너리형태로 넣어야함

submit.to_csv(”result.csv”, index=False) ← csv파일로 생성, index False 필수

pd.read_csv(”result.csv”) ← 생성된 파일 확인

## 머신러닝 학습과 평가(회귀)

- 평가지표
    
    from sklearn.metrics import root_mean_squared_error
    
    from sklearn.metrics import mean_squared_error
    
    from sklearn.metrics import mean_absolute_error
    
    from sklearn.metrics import r2_score 
    
    cf ) R2 score 만 커야 좋음, 나머진 작아야 좋음
    
- 선형회귀모델
    
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    
    lr.fir(X_train, y_train) ← 선형회귀는 random_state = 0 없음 . 항상 결과 같아서 필요없음
    
    y_pred = lr.predict(X_val)
    
    result = root_mean_squared_error(y_val, y_pred)
    
    print(’RMSE:’, result)
    
    result = mean_squared_error(y_val, y_pred)
    
    print(’MSE:’, result)
    
    result = mean_absolute_error(y_val, y_pred)
    
    print(’MAE:’, result)
    
    result = r2_score(y_val, y_pred)
    
    print(’R2:’, result)
    
- 랜덤포레스트
    
    from sklearn.esnemble import RandomForestRegressor
    
    rf = RandomForestRegressor(random_state=0)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_val)
    
    result = root_mean_squared_error(y_val, y_pred)
    
    print(’RMSE:’, result)
    
- LightGBM
    
    import lightgbm as lgb
    
    model = lgb.LGBMRegressor(random_state=0, verbose=-1)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    result = root_mean_squared_error(y_val, y_pred)
    
    print(’RMSE:’, result)