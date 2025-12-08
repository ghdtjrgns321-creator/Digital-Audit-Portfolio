# 머신러닝 기초

★★★ test.shape , result.shape 행의 수 반드시 같아야함

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

### 탐색적분석 (EDA)

- 데이터 샘플 : train.head()
- 데이터 크기 : train.shape, test.shape
- 자료형 확인 : train.info()
- 타겟 시각화 : train[’Item_Outlet_Sales’].hist()
    - 타겟 데이터가 1:1로 치우쳐져있지않은 경우, 정규분포를 띄는 경우 성능이 제일 확실해짐
- 수치형 컬럼 통계값 확인 : train.describe(), test.describe()
- 범주형 컬럼 통계값 확인 : train.describe(include = ‘0’), test.describe(include = ‘object’)
- train 데이터와 test 데이터의 동질성 확인 :
    
    print(set(train.columns) == set(test.columns))
    
- 결측치 확인 : train.isnull().sum(), test.isnull().sum()
- label(target) 별 개수 확인 : train[’income’].value_counts()
- ★★value_counts(normalize=True) → 백분율 값으로 반환해줌
- 상관관계 확인
    
    df.corr()
    
    import seaborn as sns
    
    sns.heatmap(df.corr(), annot=True) → 히트맵으로 확인
    
- 히스토그램으로 데이터 확인

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 5)

for i in range(2):

for j in range(5):

attr = i * 5 + j

sns.histplot(x = df.columns[attr], data = df, kde = True, ax=axs[i][j])

![image.png](attachment:69f44f09-3e55-4418-a6fa-15b9540ef92a:image.png)

### 데이터 전처리

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

### 로그 변환

import numpy as np

train[’SalePrice’] = np.log1p(train[’SalePrcie’])

- 로그(`log`)보다 힘이 약한 **루트(`sqrt`)**를 씌우면 딱 적당하게 퍼질 가능성이 높습니다.

import numpy as np

df['TotalCharges_sqrt'] = np.sqrt(df['TotalCharges'])

- Yeo-Johnson / Box-Cox 변환 (PowerTransformer) - ★ 추천 ← 데이터 왜곡 제일 줄여줌

Scikit-Learn의 `PowerTransformer`를 쓰면, 컴퓨터가 **"가장 정규분포에 가까워지는 최적의 공식"**을 자동으로 계산해서 적용해 줍니다. 고민할 필요 없이 가장 확실한 방법입니다.

- **특징:** 데이터의 상태를 보고 로그를 쓸지, 루트를 쓸지, 제곱을 할지 알아서 `lambda` 값을 찾아 변환합니다.

from sklearn.preprocessing import PowerTransformer

 #method='yeo-johnson' (음수, 0, 양수 모두 가능)

 #method='box-cox' (양수만 가능)

pt = PowerTransformer(method='yeo-johnson')

 #2차원 배열로 넣어야 함

df['TotalCharges_opt'] = pt.fit_transform(df[['TotalCharges']])

### 인코딩(문자 데이터를 숫자 데이터로 변형)

y_train = train.pop(”income”)  ← 타겟 분리

cf) pop → 기존 dataframe에서 열을 분리해서 없애고, 새로운 열만 가져오는기능

- 원핫(one-hot) 인코딩 : 문자 데이터에 숫자부여
    - ★원핫 → object 분리 안하고 해도 알아서 해줌
    - 순서의 개념 완전히 제거
    - 선형 회귀 같은 선형모델에서도 잘어울림
    - 데이터 과도하게 사용될 수 있음
    
    train_oh = pd.get_dummies(train)
    
    test_oh = pd.get_dummies(test)
    

심화) train 과 test의 열 갯수가 다를 때 합쳐서 원핫인코딩 후 분리

★data=pd.concat([train, test], axis=0) ← concat : 합치기

data_oh = pd.get_dummies(data)

train_oh = data_oh.iloc[:len(train)].copy() ← len(train) 기준으로 train과 test 분리

test_oh = data_oh.iloc[len(train):].copy()

- 레이블(label)인코딩 : 사전을 미리 만들어서 하나씩 적용해주는 방법
    - RandomForest, XGBoost, LightGBM 같은 트리 기반과 잘맞음
    - 숫자 할당이 순서가 있는것처럼 보일 수 있어서 선형회귀와 안어울림
    - 데이터 아낄 수 있음

 from sklearn.preprocessing import LabelEncoder

cols = train.select_dtypes(include=’object’).columns ← object 컬럼목록(.columns)만 가져오기

for col in cols: ←필수

le = LabelEncoder()

train[col] = le.fit_transform(train[col]) ← fit : 사전만들기 , transform : 하나씩 적용

test[col] = le.transform(test[col])

- 합쳐서 라벨인코딩

for col in cols:
le = LabelEncoder()
combine[col] = le.fit_transform(combine[col])

- 순서형 인코딩
    - 순서가 있는 범주를 숫자로 매핑
    - 나쁨, 보통, 좋음, 우수 등
    - 순서정보가 명시적으로 모델에 전달됨

### 스케일링

`★★★fit_transform` 같은 함수는 특성(Feature)이 여러 개일 때를 기본으로 설계되어 있어서, 특성이 하나뿐이라도 **반드시 2차원(DataFrame) 형태**로 넣어줘야 에러가 나지 않습니다.
df[['score']] = scaler.fit_transform(df[['score']])

★★★"규칙(fit)은 훈련 데이터에서만 만들고, 테스트 데이터는 그 규칙을 따르기만(transform) 해야 한다.”

- 스케일링 : 기존 범위를 특정 숫자 범위(ex : 0~1)로 조정
- 독립변수 간 범위가 다를때 사용
    - 선형회귀 → 스케일링 사용하는게 좋음
    - 트리모델(rf, lgb, xgb) → 스케일링 안해도 됨
    - 신경망 → 스케일링 필수임
- 민-맥스 스케일링 : 모든 값이 0과 1사이로 변경 → for i in range 안해도됨
    
    cols = [’age’, ‘fnlwgt’, ‘education.num’, ‘capital.gain’]
    
    from sklearn.preprocessing import MinMaxScaler
    
    sclaer = MinMaxScaler()
    
    train_copy[cols] = scaler.fit_transform(train_copy[cols]) ← 대괄호 두개 쓴거처럼 됨
    
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

- 차원축소
    - 차원의 저주 : 차원의 증가함에 따라 vector 공간 내 space증가 → 빈공간 많아짐, 예측의 정확도 떨어짐
    - 유사한 성격의 feature는 하나의 새로운 feature로 성분을 합칠 수 있음
    - 정보소실을 최소화 하면서 차원을 축소해야함
    - 선형대수학의 SVD(특이값 분해)를 이용하여 분산이 최대인 축을 찾음
    - 데이터의 분산을 최대한 보존하면서 서로 직교하는 축을 찾아 고차원 공간의 표본들을 선형 연관성이 없는 저차원으로 변환

from sklearn.decomposition import PCA

pca = PCA(n_components=4)

pca.fit(X)

print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_, ‘o—’) *#엘보우 포인트 확인*

![image.png](attachment:c1ae0f4e-4831-414b-83da-498206ce5d88:image.png)

- 차원축소 (4차원 → 2차원)

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

- 차원 축소된 data 시각화
    - 27개의 feature가 2개의 PCA로 차원축소 → 평면으로 시각화 가능해짐
    
    X1, X2 = X_train_pca[y_train==0, 0], X_train_pca[y_train == 0, 1]
    
    plt.scatter(X1, X2, color=’r’, label=’churn -0’)
    
    X1, X2 = X_train_pca[y_train==1, 0], X_train_pca[y_train == 1, 1]
    
    plt.scatter(X1, X2, color=’b’, label=’churn -1’)
    
    plt.title(’Dimension Reduction 27 → 2’)
    
    plt.xlbel(’PCA1’)
    
    plt.ylabel(’PCA2’)
    
    plt.legend
    
    ![image.png](attachment:c4b1d12f-eda6-4e49-8ea6-a6faebc9c906:image.png)
    

### 검증데이터 분할

- train 데이터(y값이 이미존재)를 train 데이터와 검증 데이터로 분할 (test 데이터는 y값이 없음)

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(train, y_train, test_size=0.2, random_state=0)

★stratify=target ← 하나의 DataFrame을 train / test 로 나눌 때 클래스들의 비율을 일정하게 유지해주기 위한 옵션

★ 회귀일 경우 stratify 불필요

쏠렸는지 확인 → y_tr.value_counts(), y_vr.value_counts()

- Training set, Testing set 섞이면 안됨, 동일한 분포 유지해야함
- Cross Validation (교차검증)
    - 훈련세트를 여러 개의 sub-set으로 나누고 각 모델을 이 sub-set 의 조합으로 훈련시키고 나머 부분으로 검증
    - Data의 수가 적은 경우 사용
    
    ![image.png](attachment:22838ddd-1ef6-407d-b024-80afc07a172d:image.png)
    

### 학습 : 분류

- 머신러닝 학습 및 예측 방법 : 모델불러오기 → 학습 fit(X,y) → 예측 predict(test) , predict_proba() ← 평가지표가 roc-auc일때 주로 사용(확률값으로 나옴)
- 의사결정나무
    - 모든 가능한 결정경로를 tree 형태로 구성
    - node → test
    - branch → test의 결과
    - leaf node → classification
    - 장점 : data preprocessing 불필요
    - 단점 : overfitting 되기 쉬움, 훈련 데이터의 작은 변화에도 매우 민감함
        - ID3 : 기본적 알고리즘. 정보이득을 이용한 트리구성 (criterion=‘entropy’)
        - CART : Gini 불순도에 기반한 트리 구성
        - C4.5, C5 : ID3 개선
    - 엔트로피 : 주어진 데이터 집합의 혼잡도 (0~1)
    - 정보이득 : 시스템에 대해 알게될수록 시스템의 엔트로피 감소
    - 의사결정나무 → 엔트로피가 낮은 상태가 되도록 나무모양으로 구분해나감

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=0, max_depth=2)

df.fit(X_tr, y_tr)

- 의사결정나무 시각화

plt.figure(figsize=(25,20))

_=tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)

![image.png](attachment:45e8d58d-6c03-45f4-bf72-699cb770f756:image.png)

- 앙상블 : 다수의 약한 학습기를 조합하여 더 높은 성능 추출
    - 배깅(부트스트래핑) → Variance 감소
        - 부트스트랩 → 중복을 허용하는 랜덤 샘플링(복원추출) → variance 감소, mean 동일
        - 배깅 트리 → RandomForest가 대표적 모델임
    - 부스팅 → Bias 감소
        - 잘못 분류된 데이터에 더 높은 가중치 부여
        - AdaBoost, Gradient Boost(XGBoost, lightgbm)
            - Gradient Boost
                - random choce보다 약간 더 나은 성능의 weak 모델을 계속 생성하여 loss function을 optimize(경사하강법)
                - 이전 tree에서 발생한 잔차를 next tree에서 보정
- RandomForest (Bagging)
    - 트리 베이스 모델 → white box 특징 유지(설명가능함)
    - 훈련데이터에 스케일링 필요 없음
    - 정확도 높음, 속도 빠름, 과적합 방지
    - low bias, low variance
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(random_state=0, max_depth=3, n_estimators=200)
    
    #max_depth (나무의 깊이)=3~12
    
    #n_estimators=100(나무가 몇개?) 이 기본, 200 400으로 가보면됨 → 과적합 방지하기위한 제한
    
    rf.fit(X_tr, y_tr)
    
    pred=rf.predict_proba(X_val)
    
    pred[:10] ← 예측 결과 확인
    
    print(rf.calsses_) ← 클래스 확인
    
- XGBoost (Gradient Boost)

from xgboost import XGBClassifier, plot_importance

rf = XGBClassifier(random_state=0, max_depth=5, n_estimators=200, learning_rate=0.01)
*#n_estimators 가 올라가면 learning_rate는 낮아져야함*

*# learning_rate → 각 tree의 기여도 조정*

rf.fit(X_tr, y_tr)

fig, ax = plt.subplots()

plot_importance(xgb, ax=ax)

![image.png](attachment:e81819d8-7c72-4af8-8b25-71afb4168c99:image.png)

- LightGBM (Gradient Boost)

from lightgbm import LGBMClassifer, plot_importance

lgbmc = lgb.LGBMClassfier(random_state=0, n_estimators= 300, verbose=-1)  →  verbose=-1 : 로그내용 숨기기

lgbmc.fit(X_tr, y_tr)

fig, ax = plt.subplots()

plot_importance(lgb, ax=ax)

![image.png](attachment:0a31f258-925f-4888-855b-414fcf893d46:image.png)

### Visualization of the Ensemble model

from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap([’#FF0000’, ‘#00FF00’])

cmap_light = ListedColormap([’#FFAAAA’, ‘#AAFFAA’])

x1_min, x1_max = X_test[:,0].min() - 1, X_test[:,0].max() + 1

x2_min, x2_max = X_test[:,1].min() - 1, X_test[:,1].max() + 1

X1, X2 = np.meshgrid(np.arrange(x1_min, x1_max, 0.1),

                                  np.arragne(x2_min, x2_max, 0.1))

XX = np.cilumn_stack([X1.ravel(), X2.ravel()])

Y_rf = rf.predict(XX)

Y_gb = gb.predict(XX)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey = True)

ax1.pcolormesh(X1, X2, Y_rf.reshape(X1.shape)), cmap=cmap_light, shading = ‘auto’)

for i in range(2):

ax1.scatter(X_test[y_test ==i, 0], X_test[y_test == i, 1], color=cmap_bold(i), label=i, s=30, edgecolor=’k’)

ax2.pcolormesh(X1, X2, Y_gb.reshape(X1.shape)), cmap=cmap_light, shading = ‘auto’)

for i in range(2):

ax2.scatter(X_test[y_test ==i, 0], X_test[y_test == i, 1], color=cmap_bold(i), label=i, s=30, edgecolor=’k’)

ax1.set_title(’Random Fores’)

ax2.set_title(’Gradient Boost’)

ax1.legend()

ax2.legend()

![image.png](attachment:7ed3e245-a811-4ff1-acd8-5673e5216f4d:image.png)

- KNN (회귀, 분류 다 가능)
    - simple and easy
    - datasets 많아지면 느려짐
    - 결측값, outlier에 영향을 많이 받음
    - K값 선택해야함
    - weights 인자
        - uniform : 모든 neighbor의 가중치를 동일하게 취급(거리 가중치 주지 않겠다)
        - distance : neighbor의 거리에 반비례하여 가중치조정

from sklearn. neighbors import KNneighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights=unifrom) ←random_state 없음

for i in range(3):

plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], label=i)

plt.plot(X_test[20,0], X_test[20, 1], c=’r’, marker=’x’, markersize=20)

plt.legend()

clf.predict(X_test[20:21])

![image.png](attachment:7a4fa3ea-0f3f-4c1d-b556-72e4a273c272:image.png)

knn.fit(X_tr,y_tr)

- SVC ← 정규화 필수

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

#정규화 작업

scaler = Standardcaler()

scaler.fit(X_tr)

X_tr = scaler.transform(X_tr)

#svc훈련

svc = SVC(kernel=’poly’, C=3, degree=3)

moderl.fit(X_tr,y_tr)

- LogisticRegressor ← 분류모델에 적용(회귀선을 바탕으로 분류)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_tr,y_tr)

- 기본값 → 0.5 보다 높으면 1, 낮으면 0
- threshold 조정
    
    y_pred_proba = lr.predict_proba(X_test)
    
    y_pred_proba[:,1]> 0.5
    
     # 조정
    
    threshold = 0.8 *#0.8 이상인경우에만 1로 판단*
    
    y_pred_proba = lr.predict_proba(X_test)
    
    y_pred_proba1 = y_pred_ptroba[:, 1] > threshold
    
    y_pred_ptroba1
    
    sum(y_pred_proba1 == y_test) / len(y_test) *#accuray 계산(떨어짐)*
    
    precision_score(y_test, y_pred_proba1) ← *떨어짐*
    
    recall_score(y_test, y_pred_proba1) ← *높아짐*
    
- GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)

gbc.fit(X_tr, y_tr)

### 학습 : 회귀

- LinearRegression
    - 결과값을 얻기 위한 수식 생성
    - 추세를 외삽하는데 탁월(어떤 값이 오더라도 결과생성가능)
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_tr,y_tr)
    
    print('weight:', lr.coef_) ← x
    print('bias:', lr.intercept_) ← 절편 
    
    coef = pd.Series(data=np.round(lr.coef_, 1), index=df.columns)
    coef *#보기좋게 나열*
    
    ![image.png](attachment:63c5a173-76eb-4e7c-9716-18c09db531c0:image.png)
    
    coef_sort = coef.sort_values(ascending=False)
    sns.barplot(x=coef_sort.values, y=coef_sort.index)
    
    ![image.png](attachment:9835a763-e215-48dd-98c5-64f92790d065:image.png)
    

plt.scatter(X_test, y_test, label=’True value’)

plt.plot(X_test, y_pred, color=”r”, label=’Predicted’)

plt.xlabel(’bmi’)

plt.ylabel(’Progress’)

plt.lengend()

![image.png](attachment:5c783c2e-a436-4cb1-9f80-3c7a187a16f1:image.png)

- DicisionTreeRegressior
    - 결과값을 그룹화하여 예측
    - 속성 간 상호작용을 학습 할 수 있음
- RandomForestRegressor
- LGBMRegressor
- XGBRegressor

### 학습 : 군집

- 비지도학습
    - 비슷한 object 들끼리 모으는 것
    - label data가 없음
- K Means
    - 거리계산
    - 랜덤하게 k개의 중심점을 정함
        - k 정하기 → 엘보우 포인트
    - 중심점이 변하지 않을 때 까지 반복

from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, random_state=0)

y_preds = km.fit_predict(X)

df[’clusters’] = y_preds

 # k-means 센터찾기

centers = k_means.cluster_cneters_

 # k-means 시각화

from matplotlib.colors import ListedColormap

colors_bold = ListedColormap([’#FF0000’, ‘#00FF00’, ‘#0000FF’])

colors_light = ListedColormap([’#FFAAAA’, ‘#AAFFAA’, ‘#AAAAFF’])

plt.figure(figsize=(8,6))

for i in range(3):

plt.scatter(X[members == i, 0], X[members == i,1], marker=’.’, color=colors_light(i), label=i)

plt.plot(centers[i,0], centers[i,1], ‘o’, markersize=20, color=color_bold(i), markeredgecolor=’k’)

plt.legned()

![image.png](attachment:7bb133d2-2bf0-448a-8093-1c5462d561e1:image.png)

- DBSCAN
    
    ![image.png](attachment:c71ced75-6350-494f-9a44-530b40f2f530:image.png)
    
    - 밀도가 높은 지역과 낮은 지역을 분리
    - cluster 숫자 미리 지정 필요x
    - outlier의 영향을 적게받음

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=7)

db.fit(X)

 *#  시각화*

labels = list(set(db.labels_))

color = plt.cm.Spectral(np.linspace(0, 1, len(labels)))

list(zip(labels, colors))

plt.figure(figsize=(8,6))

for k, col in zip(labels, colors):

members = (dv.labels == k)

plt.scatter(X[members,0], X[members, 1], color=col, marker=’o’, s=10)

![image.png](attachment:aa5c38fb-523e-4fe1-994e-b74f49e0b65b:image.png)

### 피쳐 엔지니어링

- 피쳐 엔지니어링 → 학습 후 성능 개선을 위해 다시 조정(스케일링 등)
- 스케일링
- 교차검증
    
    from sklearn.model_selection import cross_val_score
    import numpy as np
    scores = cross_val_score(rf, train, train_target, cv=5, scoring='neg_mean_absolute_error')
    

![image.png](attachment:e6ef7363-3ead-4140-b84a-c7f35d750b58:image.png)

- K Fold
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=3) ← n_splits : 3등분하겠다
    
    from sklearn.moder_selection import StratifiedKFold ←범주유형 고려해서 K-Fold 해줌
    
    skf = StratifiedKFold(n_splits=3)
    
    - cross_validate ← 분류해서 학습까지 해주는 기능
        
        from sklearn.model_selection import corss_validate
        
        scores = cross_validate(tree_model, X_train, y_train, cv=3, return_estimator=True)
        
        scores
        
- 차원축소
- 하이퍼 파라미터 튜닝
    - GridSearchCV - RandomForestClassifier
    # - n_estimators : 100, 300, 500
    # - max_depth : 6, 8, 10, 12
    # - min_samples_leaf : 8, 12, 18
    # - min_samples_split : 8, 16, 20
    
    from sklearn.model_selection import GridSearchCV
    
    params = {
        'n_estimators':[100, 300, 500],
        'max_depth' : [6, 8, 10, 12],
        'min_samples_leaf' : [8, 12, 18],
        'min_samples_split' : [8, 16, 20]
    }
    
    #RandomForestClassifier 객체 생성 후 GridSearchCV 수행
    rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
    grid_cv.fit(X_tr , y_tr)
    
    print('최적 하이퍼 파라미터:\n:', grid_cv.best_params_)
    print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
    - GridSearchCV - GBM
    # -n_estimators : 100, 500
    # learning_rate : 0.05, 0.1
    
    from sklearn.model_selection import GridSearchCV
    
    params = {
        'n_estimators':[100, 500],
        'learning_rate': [0.05, 0.1]
    }
    
    gbm_model = GradientBoostingClassifier(random_state=10)
    
    grid_cv = GridSearchCV(gbm_model, param_grid=params, cv=2, verbose=1)
    grid_cv.fit(X_tr, y_tr)
    print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
    print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
- Early Stopping
    - XGBoost
    
    xgb2 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0, 
                         early_stopping_rounds=100)
    xgb2.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)])
    
    - LGBM
    
    lgb2 = LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0,
                          early_stopping_round=100, verbose=1)
    lgb2.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)])
    lgb_pred = lgb2.predict(X_test)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    print(lgb_accuracy)
    
    import lightgbm as lgb ← 진행하면서 어떻게 변화했는지 확인
    lgb.plot_metric(lgb_model2)
    
    ![image.png](attachment:2b149dc9-487e-42b7-888a-7bb71a26c958:image.png)
    
    lgb.plot_importance(lgb_model2) ← 어떤 변수에 영향 제일 많이 받는지 확인
    
    ![image.png](attachment:22157f98-5bc1-487d-a6c2-5ee74b413859:image.png)
    
- VotingClassifier
    - 앙상블 모델들을 생성/학습/평가
    - LogisticRegression, KNeighborsClassifier 조합
    - voing=’soft’
    
    from sklearn.ensemble import VotingClassifier
    voting_model = VotingClassifier(estimators=[('LR', lr),\
                                                ('KNN', kn)],
                                     voting='soft')
    voting_model.fit(X_tr, y_tr)
    pred = voting_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print('VotingClassifier 정확도: {0:.4f}'.format(accuracy))
    
- 독립변수 일부만 선택
from sklearn.feature_selection import SelectKBest, f_regression
# k=3 선택
X_selected = SelectKBest(score_func = f_regression, k=3)
X_selected.fit_transform(df,target)
features = df.columns[X_selected.get_support()]
print('features = {}'.format(features))
    
    # best3 만 활용하여 회귀
    X_selected = df[features].copy()
    X_tr, X_test, y_tr, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=0)
    lr2 = LinearRegression()
    lr2.fit(X_tr,y_tr)
    y_pred = lr2.predict(X_test)
    printRegressorResult(y_test, y_pred)
    
- 다항회귀모델로 변경 (2차식만들기 degree=2)
    
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X_selected)
    
    X_tr, X_test, y_tr, y_test = train_test_split(X_poly, target, test_size=0.2, random_state=0)
    lr3 = LinearRegression()
    lr3.fit(X_tr,y_tr)
    y_pred = lr3.predict(X_test)
    printRegressorResult(y_test, y_pred)
    
- 로그변환 (좌측으로 치우처진 데이터에 대해서)
    
    # 데이터 분포 확인 - 히스토그램
    nrows = 1
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(20,4)
    
    sns.histplot(x= X_selected.columns[0], data = X_selected, kde=True, bins=30, ax=axs[0])
    sns.histplot(x= X_selected.columns[1], data = X_selected, kde=True, bins=30, ax=axs[1])
    sns.histplot(x= X_selected.columns[2], data = X_selected, kde=True, bins=30, ax=axs[2])
    sns.histplot(x= target, data = df, kde=True, bins=30, ax=axs[3])
    
    ![image.png](attachment:4f526682-b692-439a-b0a2-d150a76e19ea:image.png)
    
    # 왼쪽으로 치우쳐진 인구밀집도 <- 로그 변환
    X_selected['인구 밀집도'] = np.log1p(X_selected['인구 밀집도'])
    target = np.log1p(target)
    
    # 데이터 분포 확인 - 히스토그램
    nrows = 1
    ncols = 4
    
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(20, 4)
    
    sns.histplot(x= X_selected.columns[0], data = X_selected, kde=True, bins=30, ax=axs[0])
    sns.histplot(x= X_selected.columns[1], data = X_selected, kde=True, bins=30, ax=axs[1])
    sns.histplot(x= X_selected.columns[2], data = X_selected, kde=True, bins=30, ax=axs[2])
    sns.histplot(x= target, data = df, kde=True, bins=30, ax=axs[3])
    
    print(X_selected.skew())
    print('\n평균 주택 가격: {0:.2f}'.format(y.skew()))
    
    ![image.png](attachment:51edf063-d701-4ff5-9d5c-6618956551b7:image.png)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    lr_model3 = LinearRegression()
    lr_model3.fit(X_train, y_train)
    
    y_pred = lr_model3.predict(X_test) 
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    rmse = np.sqrt(mse)
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))
    print('MSE : {0:.3f} , RMSE : {1:.3f}, r2 : {2:.3f}'.format(mse , rmse, r2))
    

### 평가(분류)

- 혼동행렬, 정확도, 정밀도, 재현율, F1_Score, AUC

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pred = ~.predict(X_val)

pred.proba = ~.predict_proba(X_val[:,1])

- 혼동 행렬
    - TP - 1 을 1로 제대로 분류
    - TN - 0 을 0 으로 제대로 분류
    - FP - 0 을 1 로 잘못 분류
    - FN - 1 을 0 으로 잘못 분류
        - Accuracy = TP+TN / TP+TN+FP+FN (단순 정확성, 전체 데이터 중 제대로 분류된 데이터 비율)
        - Precision = TP / TP + FP → 전체 Positive 예측 중 실제 Postive(높을수록 1종오류 없음)
        - Recall = TP / TP + FN → 실제 Positive 데이터 중 Positive로 예측한 비율(높을수록 2종오류 없음)
            - Precision, Recall → 상충관계에 있음
            - F1 Score → Precison, Recall의 조화평균
                - Precision과 Recall의 균형을 통해서 종합적인 성능 평가
                    
                    → ★데이터가 불균형할때 F1 스코어 유용
                    
                    ![image.png](attachment:7dba8b7b-332d-4da6-8c63-0385b82d09f7:image.png)
                    
    
    ![image.png](attachment:6629b4b7-1fec-439d-95d7-9196c2572247:image.png)
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    print(cm)
    
    ![image.png](attachment:0b816b4b-bb64-4026-b41b-6a79778ab3dd:image.png)
    
    import seaborn as sns
    
    plt.figure(figsize=(5,4))
    
    ax = sns.heatmap(cm, annot=True, fmt=’d’)
    
    ax.set_ylabel(’True’)
    
    ax.set_title(’Confusion Matrix\nPredicted’)
    
    ![image.png](attachment:9df3564a-6e95-4b4e-9888-cbc612058f2a:image.png)
    
- 정확도(accuracy) 확인
    
    from sklearn.metrics import accuracy_score
    
    result = accuracy_score(y_val, pred)
    
- roc-auc
    - ★분류기 간 성능 비교★가능
    
    ![image.png](attachment:6b1dbf4e-5047-4fb1-87de-53846f129deb:image.png)
    
    from sklearn.metrics import roc_auc_score
    
    result = roc_auc_metrics(y_val, pred_proba([:,1]) ← 1일 확률
    
    - roc_auc 시각화
        
        y_porba = lr.predict_proba(X_test)
        
        y_scores = y_proba[:,1]
        
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        
        auc = roc_auc_score(y_test, y_scores)
        
        plt.plot(fpq, tqr, label=”auc={:.2f}”.format(auc))
        
        plt.xlabel(’False Positive Rate’)
        
        plt.ylabel(’True Positive Rate’)
        
        plt.title(’ROC-AUC curve’)
        
        plt.legend()
        
- RMSE
    
    from sklearn.metrics import root_mean_squared_error
    
    result = root_mean_squared_error(y_val, pred)
    
- f1_score
    
    from sklearn.metrics import f1_score
    
    result = f1(y_val, pred, average=’macro’)
    
- 이진 분류 시 threshold 조정
    - threshold 조정
    
    y_pred_proba = lr.predict_proba(X_test)
    
    y_pred_proba[:,1]> 0.5
    
    - 조정
    
    threshold = 0.8 #0.8 이상인경우에만 1로 판단
    
    y_pred_proba = lr.predict_proba(X_test)
    
    y_pred_proba1 = y_pred_ptroba[:, 1] > threshold
    
    y_pred_ptroba1
    
    sum(y_pred_proba1 == y_test) / len(y_test) #accuray 계산(떨어짐)
    
    precision_score(y_test, y_pred_proba1) ← 떨어짐
    
    recall_score(y_test, y_pred_proba1) ← 높아짐
    

### 평가(군집)

- 실루엣

from sklearn.metrics import silhouette_samples, silhouette_score

silhouette_score(X, y_preds)

- 최적의 k 찾기

# 최적의 k 찾기
silhouette_avg = []
for k in range(10):
  model = KMeans(n_clusters=k+2, random_state=0)
  y_preds = model.fit_predict(X)
  score = silhouette_score(X, y_preds)
  silhouette_avg.append(score)
  print("군집개수: {0}개, 평균 실루엣 점수 : {1:.4f}".format(k+2, score))

![image.png](attachment:7b986c49-5c77-49fb-8a76-dee8c4aa033c:image.png)

plt.plot(range(2,12), silhouette_avg, 'bo--')
plt.xlabel('# of clusters')
plt.ylabel('silhouette_avg')
plt.show()

![image.png](attachment:9de78f20-8ded-4ccc-93c7-33a6a8f483ad:image.png)

### 평가표 확인

from sklearn.metrics import confusion_matrix, accuract_score

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score, roc_auc_score

def get_clt_eval(y_test, pred=None): *# None은 "값이 없음" 또는 "인자가 제공되지 않음"을 명시적으로 나타내는 데 널리 사용됩니다. 만약 기본값을 지정하지 않으면 pred 인자는 항상 필수로 전달되어야 합니다. 따라서 pred=None은 pred가 선택적임을 의미하고, 함수 내부에서 pred가 None인지 체크하여 다른 동작을 할 수 있게 유연하게 설계하는 데 쓰입니다.*

 confusion = confusion_matrix(y_test, pred)
  accuracy = accuracy_score(y_test, pred)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)
  f1 = f1_score(y_test, pred)
  print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}'.format(accuracy, precision, recall, f1)) *#4f : 소숫점 4자리까지 출력, 0,1,2,3에 대응하는 변수들은 format() 함수 내부에 순서대로 들어간 accuracy, precision, recall, f1입니다.*
  print(confusion)

model_list = [dt_model, neighbor_model, svm_model, forest_model, logistic_model, gbm_model, xgb_model, lgb_model]

for model in model_list:
  pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, pred)
  model_name = model.__class__.__name__
  print('\n{} 성능지표:'.format(model_name))
  get_clf_eval(y_test, pred)

![image.png](attachment:52c713ef-f048-453a-aadd-7492e3a900d9:image.png)

### 평가 (회귀)

- RMSE
    
    from sklearn.metrics import root_mean_squared_error
    
    result = root_mean_squared_error(y_val, y_pred)
    
    print(’RMSE:’, result)
    
- RSE
    
    from sklearn.metrics import mean_squared_error
    
    result = mean_squared_error(y_val, y_pred)
    
    print(’MSE:’, result)
    
- MAE
from sklearn.metrics import mean_absolute_error
    
    result = mean_absolute_error(y_val, y_pred)
    
    print(’MAE:’, result)
    
- R2
    
    from sklearn.metrics import r2_score 
    
    result = r2_score(y_val, y_pred)
    
    print(’R2:’, result)
    

from sklearn.metrics import mean_squared_error, r2_score

y_pred = lr.predict(X_test)
def printRegressorResult(y_test, y_pred):
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_test, y_pred)
  print('MSE : {0:.3f} , RMSE : {1:3f}, r2 : {2:.3f}'.format(mse, rmse, r2))
printRegressorResult(y_test, y_pred)

![image.png](attachment:f0f0c300-2071-4b35-b300-43abc76ad76e:image.png)

# 실제값과 예측값 얼마나 차이나는지 확인
result = pd.DataFrame({'y' : y_test.values,
                       'y_pred' : y_pred,
                       'diff' : np.abs(y_test.values - y_pred)})
result.sort_values('diff', ascending=False).head()

![image.png](attachment:90cf2b52-dbcb-4c91-8688-45dc9e48b6a4:image.png)

### 예측 및 결과 파일 생성

submit = pd.DataFrame({’pred’:pred[:,1]}) ← dataframe : 딕셔너리형태로 넣어야함

submit.to_csv(”result.csv”, index=False) ← csv파일로 생성, index False 필수

pd.read_csv(”result.csv”) ← 생성된 파일 확인

★★★★★★★★★★★★정리★★★★★★★★★★★★

- 원핫인코딩 → ★‘ ‘안씀★

train = pd.get_dummies(train)

test = pd.get_dummies(test)

- 합쳐서 원핫인코딩

combine = pd.concat([train,test])

combine_dummies = pd.get_dummies(combine)

n_train = len(train)

train = combine_dummies[:n_train]

test = combine_dummiesd[n_train:]

- 라벨인코딩

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cols = train.select_dtypes(include = ‘object’).columns

for col in cols:

train[col] = le.fit_transform(train[col])

test[col] = le.transform(test[col])

- 분할

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(train, y_train, test_size=0.2, random_state=0)

- RandomForest → 분류 : RandomForestClassifier, 회귀 : RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifer(random_state=0)

pred = rf.predict(X_val)

- lightgbm

import lightgbm as lgb

lgb = lgb.LGBMClassifier(random_state=0, verbose=-1)

- XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=0)

- roc-auc

from sklearn.metrics import roc_auc_score

result = roc_auc_metrics(y_var, pred([:,1])

- RMSE

from sklearn.metrics import root_mean_squared_error

result = root_mean_squared_error(y_val, pred)

- f1_score

from sklearn.metrics import f1_score

result = f1(y_val, pred, average=’macro’)

**Random Forest를 선택해야 할 때:**

- 데이터가 충분하고 피처 개수가 적당할 때
- 모델의 해석 가능성이 중요할 때
- 빠른 개발 속도가 필요할 때 (파라미터 조정이 간단)
- 이상치가 많은 데이터를 다룰 때

**XGBoost를 선택해야 할 때:**

- 높은 정확도가 최우선일 때
- 표 형식의 정제된 데이터를 다룰 때
- 계산 자원이 충분할 때
- 범주형 데이터가 적당한 수준일 때

**LightGBM을 선택해야 할 때:**

- 대규모 데이터셋을 빠르게 처리해야 할 때
- 메모리 효율성이 중요할 때
- 범주형 특성이 많을 때
- 학습 속도가 프로젝트의 핵심 요구사항일 때 (LightGBM이 다른 모델보다 **10배 이상 빠를 수 있습니다**)
