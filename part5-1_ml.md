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
- 수치형 컬럼 통계값 확인 : train.describe(), test.describe()
- 범주형 컬럼 통계값 확인 : train.describe(include = ‘0’), test.describe(include = ‘object’)
- train 데이터와 test 데이터의 동질성 확인 :
    
    print(set(train.columns) == set(test.columns))
    
- 결측치 확인 : train.isnull().sum(), test.isnull().sum()
- label(target) 별 개수 확인 : train[’income’].value_counts()
- 히스토그램으로 데이터 확인

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 5)

for i in range(2):

for j in range(5):

attr = i * 5 + j

sns.histplot(x = df.columns[attr], data = df, kde = True, ax=axs[i][j])

![image.png](image.png)

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

### 인코딩(문자 데이터를 숫자 데이터로 변형)

y_train = train.pop(”income”)  ← 타겟 분리

cf) pop → 기존 dataframe에서 열을 분리해서 없애고, 새로운 열만 가져오는기능

- 원핫(one-hot) 인코딩 : 문자 데이터에 숫자부여
    
    train_oh = pd.get_dummies(train)
    
    test_oh = pd.get_dummies(test)
    

심화) train 과 test의 열 갯수가 다를 때 합쳐서 원핫인코딩 후 분리

★data=pd.concat([’train’, ‘test’], axis=0) ← concat : 합치기

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
    
    - 합쳐서 라벨인코딩
    
    for col in cols:
    le = LabelEncoder()
    combine[col] = le.fit_transform(combine[col])
    

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

- 차원축소

from sklearn.decomposition import PCA

pca = PCA(n_components=4)

pca.fit(X)

print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_, ‘o—’)

![image.png](image%201.png)

- 차원축소 (4차원 → 2차원)

pca = PCA(n_components=2)

pca_transformed = cpa.fit_transform(iris.data)

df[’pca_1’] = pca_transformed[:,0]

df[’pca_2’] = pca_transformed[:,1]

df.head()

![image.png](image%202.png)

### 검증데이터 분할

- train 데이터(y값이 이미존재)를 train 데이터와 검증 데이터로 분할 (test 데이터는 y값이 없음)

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(train, y_train, test_size=0.2, random_state=0)

★stratify=target ← 하나의 DataFrame을 train / test 로 나눌 때 클래스들의 비율을 일정하게 유지해주기 위한 옵션

★ 회귀일 경우 stratify 불필요

쏠렸는지 확인 → y_tr.value_counts(), y_vr.value_counts()

### 학습 : 분류

- 머신러닝 모델 : 랜덤포레스트, lightGBM
- 머신러닝 학습 및 예측 방법 : 모델불러오기 → 학습 fit(X,y) → 예측 predict(test) , predict_proba() ← 평가지표가 roc-auc일때 주로 사용(확률값으로 나옴)
- 의사결정나무

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=0, max_depth=2)

df.fit(X_tr, y_tr)

- 의사결정나무 시각화

from sklearn.tree import export_graphviz

from subprocess import call

from IPython.display import image

export_graphbiz(dt, feature_names = X_train.columnsm out_filt=’tree.dot’) ← 의사결정나무 모델을 dot 파일로 추출

call([’dot’, ‘-Tpng’, ‘tree.dot’, ‘-o’, ‘tree.png’, ‘=Gdpi=600’]) ← dot 파일을 .-png파일로 변환

Image(filename = ‘tree.png’) ← png 출력

![image.png](image%203.png)

- 랜덤포레스트
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(random_state=0, max_depth=3, n_estimators=200) #max_depth=3~12
    
    #n_estimators=100 이 기본, 200 400으로 가보면됨 → 과적합 방지하기위한 제한
    
    rf.fit(X_tr, y_tr)
    
    pred=rf.predict_proba(X_val)
    
    pred[:10] ← 예측 결과 확인
    
    print(rf.calsses_) ← 클래스 확인
    
- XGBoost

from xgboost import XGBClassifier, plot_importance

rf = XGBClassifier(random_state=0, max_depth=5, n_estimators=200, learning_rate=0.01)
#n_estimators 가 올라가면 learning_rate는 낮아져야함

rf.fit(X_tr, y_tr)

fig, ax = plt.subplots()

plot_importance(xgb, ax=ax)

![image.png](image%204.png)

- LightGBM

from lighrgbm import LGBMClassifer, plot_importance

lgbmc = lgb.LGBMClassfier(random_state=0, n_estimators= 300, verbose=-1)  →  verbose=-1 : 로그내용 숨기기

lgbmc.fit(X_tr, y_tr)

fig, ax = plt.subplots()

plot_importance(lgb, ax=ax)

![image.png](image%205.png)

- KNN

from sklearn. neighbors import KNneighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5) ←random_state 없음

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
    
    ![image.png](image%206.png)
    
    coef_sort = coef.sort_values(ascending=False)
    sns.barplot(x=coef_sort.values, y=coef_sort.index)
    
    ![image.png](image%207.png)
    
- DicisionTreeRegressior
    - 결과값을 그룹화하여 예측
    - 속성 간 상호작용을 학습 할 수 있음
- RandomForestRegressor
- LGBMRegressor
- XGBRegressor

### 학습 : 군집

- K Means

from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, random_state=0)

y_preds = km.fit_predict(X)

df[’clusters’] = y_preds

### 피쳐 엔지니어링

- 피쳐 엔지니어링 → 학습 후 성능 개선을 위해 다시 조정(스케일링 등)
- 스케일링
- 교차검증
    
    from sklearn.model_selection import cross_validate
    
    # cv: 3개의 train, test set fold 로 나누어 학습 
    scores = cross_validate(lr_model, X, y, scoring="neg_mean_squared_error", cv=3, return_train_score=True, return_estimator=True)
    print('Scores', scores)
    
    mse = (-1 * scores['train_score'])
    print('MSE:', mse)
    
    rmse  = np.sqrt(-1 * scores['train_score'])
    print('RMSE:', rmse)
    
    print('RMSE 평균: {0:.3f} '.format(np.mean(rmse)))
    

![image.png](image%208.png)

- K Fold
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=3) ← n_splits : 3등분하겠다
    
    from sklearn.moder_selection import StratifiedKFold ←범주유형 고려해서 K-Fold 해줌
    
    skf = StratifiedKFold(n_splits=3)
    
    - cross_validate ← 분류해서 학습까지 해주는 기능
        
        from sklearn.model_selection import corss_validate
        
        scores = cross_validate(tree_model, X_train, y_train, cv=3, return_estimator=True)
        
        scores
        
        ![image.png](image%209.png)
        
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
    
    ![image.png](image%2010.png)
    
    lgb.plot_importance(lgb_model2) ← 어떤 변수에 영향 제일 많이 받는지 확인
    
    ![image.png](image%2011.png)
    
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
    
    ![image.png](image%2012.png)
    
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
    
    ![image.png](image%2013.png)
    
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

- 정확도(accuracy) 확인
    
    from sklearn.metrics import accuracy_score
    
    result = accuracy_score(y_val, pred)
    
- roc-auc
    
    from sklearn.metrics import roc_auc_score
    
    result = roc_auc_metrics(y_val, pred_proba([:,1])
    
- RMSE
    
    from sklearn.metrics import root_mean_squared_error
    
    result = root_mean_squared_error(y_val, pred)
    
- f1_score
    
    from sklearn.metrics import f1_score
    
    result = f1(y_val, pred, average=’macro’)
    

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

![image.png](image%2014.png)

plt.plot(range(2,12), silhouette_avg, 'bo--')
plt.xlabel('# of clusters')
plt.ylabel('silhouette_avg')
plt.show()

![image.png](image%2015.png)

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

![image.png](image%2016.png)

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

![image.png](image%2017.png)

# 실제값과 예측값 얼마나 차이나는지 확인
result = pd.DataFrame({'y' : y_test.values,
                       'y_pred' : y_pred,
                       'diff' : np.abs(y_test.values - y_pred)})
result.sort_values('diff', ascending=False).head()

![image.png](image%2018.png)

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

[머신러닝 실습1](https://www.notion.so/1-2a449b8bb7f88035b3aed43e728951b0?pvs=21)

[머신러닝 실습2](https://www.notion.so/2-2a449b8bb7f880ce8b5ae37dbbcbe8dd?pvs=21)

[머신러닝 실습3](https://www.notion.so/3-2a449b8bb7f880fd9f13f5e2a3a13da1?pvs=21)

[](https://www.notion.so/2ad49b8bb7f8802fad06ff9033a1fbd6?pvs=21)