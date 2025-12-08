# 딥러닝

### 딥러닝

- 전통적인 ML : Domain 지식, 통계학적 지식을 바탕으로 좋은 Feature를 만들어 Data를 구성 (알고리즘 학습 중요성이 상대적으로 떨어짐)
- 딥러닝 : 중요한 Feature를 스스로 구분하여 weight를 부여
- 사람이 manual하게 정한 feature는 위험이 있고 작성에 시간이 많이 소요됨
- 딥러닝은 parameter 를 스스로 학습 → 적용하기 쉽고 빠름
- Raw data를 거의 그대로 사용 → computer vision, 언어 처리 등 ex) image, sound, characters, words
- 이미지 인식, 대화/언어 문제에 탁월한 성능

### 퍼셉트론

- Pre-Activation + Activation
    - Pre-Activation : linear Regression
    - Activation : sigmoid 등 활성화함수
    
    ![image.png](image.png)
    

### 활성화 함수

- Sigmoid(아웃풋에서 많이 쓰임) : 기울기 소실 문제
- 하이퍼탄젠트 : 기울기 소실 문제
- ReLU(현대 딥러닝) : 미분 안함, 기울기 소실 안됨 (직선임)
- Leaky ReLU
- ELU(언어모델에 가끔쓰임)

![image.png](image%201.png)

### Softmax 활성화 함수

- 출력값의 다중 class 분류를 위하여 출력값에 대해 정규화 → 확률 분포 출력

![image.png](image%202.png)

### Deep Neural Network 훈련의 핵심

- 경사하강법
    - 실제값과 예측값의 차이를 최소화하는 parameter 발견
    - 손실함수를 정의하여 손실함수의 값이 0으로 수렴하도록 parameter 조절
    - 방향 : Gradient
    - 이동속도 : learning rate
- 오차역전파
    - 손실함수를 최소화 하는 방향으로 신경망 전체의 parameter update
- 손실함수
    - =비용함수, =목적함수
    - 경사하강법이 가능하도록 미분 가능한 함수를 정의
        - 선형회귀(Linear Regression) → MSE(Mean Squared Error) → MSE를 최소화하는 W와 b를 optimize(3차원 포물선)
            
            ![image.png](image%203.png)
            
        - 이진분류(Logistic Regression) → Binary-Cross-Entropy (로그함수 2개 겹쳐서 사용)
            
            ![image.png](image%204.png)
            
        - 다중분류 → Categorical-Cross-Entropy(Softmax loss) : 원핫인코딩 필수
            
            ![image.png](image%205.png)
            
- Global Minimum/ Local minimum

![image.png](image%206.png)

- Learning Rate(step size)
    - 높음 : 빠른러닝, 최솟값 지나칠 가능성 높음
    - 낮음 : 느린러닝, 최솟값 지나칠 가능성 낮음
    - Adaptive Learning Rates : 초기값을 크게 주고 학습, 진행에 따라 slow down
- Momentum : 관성. 방향성을 유지하면서 가속 → Local Minima, Saddle points 탈출
    - 하이퍼 파라미터임

![image.png](image%207.png)

- 옵티마이저 : 최솟값 찾기 알고리즘 → 보통 adam 사용
- epoch
    - 전체 dataset이 neural network를 통해 한 번 처리된것
    - 하이퍼파라미터임
    - 여러개의 batch로 나누어 처리
    - Parameter Training을 위해서는 여러 번 epoch를 반복해야함
    - One epoch내에서의 iteration 횟수는 total sample size / batch size
        
        ex) 1 epoch = 4 iterations = 2000 training example / 500 batches
        
- 하이퍼파라미터
    - 러닝레이트
    - 모멘텀 텀
    - 은닉층 갯수
    - dropout rate(과적합 방지)
    - 에폭 수
    - 배치 사이즈
- 하이퍼파라미터 결정
    - 정해진 RULE 없음
    - 유사한 model 참조
    - 경험에 의한 guessing
    - Grid search → 다 해보고 결정

# 인공 신경망

## 인공 신경망

- 신호를 전달받아 가중치를 곱하여 임계값이 넘을경우(by 활성화함수) 0또는 1 출력
- 계단함수(활성화 함수) → 임계값이 넘으면 1, 못넘으면 0 출력
- 입력층과 출력층 사이에 반복층 → 은닉층

![image.png](image%208.png)

- 인공신경망 : 여러개의 선형회귀가 겹친 형태로 생각할 수 있음
- 인공신경망 학습 → 가중치를 찾아내는것

## 인공 신경망 학습

### 환경변수 지정 → 신경망 실행 → 예측값과 실제값 비교 → 가중치 수정

- 환경변수 지정 : 몇 개의 층? 몇 개의 노드? 가중치 업데이트는 어떻게?
- 신경망 실행 : 연산 수행
    - 은닉층 활성화 함수
        - 계단함수(Logistic Regression) : 임계값보다 적으면 0 , 높으면 1 을 반환, 미분불가능(깊어질수록 정확도 낮아짐)
        - sigmoid, tanh(하이퍼블릭탄젠트) : 0에서 1까지 연속적인 함수, 미분가능 but 미분 할수록 결과가 작아지는 문제가 생김 → 0으로 수렴하는 문제
            
            ![image.png](image%209.png)
            
        - relu : 임계값 넘을시 넘은 값 그대로 출력 → 미분 x → 기울기 소실 문제 해결
        
        ![image.png](image%2010.png)
        
    - 출력층의 활성화함수
        - sigmoid → 분류(0또는 1로출력)
        
        ![image.png](image%2011.png)
        
        - softmax → 분류(확률의 형태로 예측값이 출력)
        
        ![image.png](image%2012.png)
        
        - relu → 회귀문제에 사용 (임계값 넘을 시 넘은 값 그대로 반환)
- 예측값과 실제값 비교
    - 손실함수 사용
        - 평균제곱오차(MSE) → 손실의 최솟값 찾기
            
            최솟값 찾기 위해 건너뛰는 범위 : 학습률
            
        
        ![image.png](image%2013.png)
        
        - 교차 엔트로피 오차(Cross Entropy Error) → 예측값의 범위와 실제값의 범위를 비교 (분류 문제에서 자주 활용)
        
        ![image.png](image%2014.png)
        
- 가중치 수정 : 출력층 → 은닉층 → 입력층 순으로 가면서 가중치 수정(에러 역전파)

## 인공신경망 설계

### 신경망 만들기

- 신경망 설계항목 (은닉층 → 하이퍼파라미터)
    1. 층(Layer)
    2. 각 Layer의 노드 수
    3. 활성화함수
    4. 손실함수
    5. 옵티마이저(ADAM)

![image.png](image%2015.png)

## 인공신경망 실습(TensorFlow)

### 모델 = 레이어 + 레이어 + 레이어 + 레이어 .. + 레이어

- 모델
    - 구성 : add(), summary(), getlayer()
    - 학습 및 평가 : compile(), fit(), evaluate(), predict()
    - 저장 : save_weights(), load_weights()
- 레이어
    - Dense : units로 구성, activation(활성화함수)

### 모델 설계 및 학습(회귀)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

 *# 모델의 설정*

 *# input_dim → 입력층, X_train.shape[1] ← X_train의 열 받아옴*

 *# 16, 8 ← 은닉층 노드의 갯수*

 *# 출력층 노드 1개 ← 회귀문제라서 1개출력*

model = Sequential()

model.add(Dense(16, input_dim=X_train.shape[1], activation=’relu’))

model.add(Dense(8, activation=’relu’))

model.add(Dense(1, activation=’relu’))

 *#model.summary() 로 확인 가능*

 *# 모델 컴파일 metrics → 중간중간 출력해보기*

model.compile(loss=’mse’, optimizer=’adam’, metrics=[’mse’])

 *# 모델 훈련*

 *# validation_split = test_size*

 *# epochs=50 → 전체데이서 셋 50번 학습*

 *# batch_size → 한번 볼때마다 64개씩 쪼개서 봐라*

 *# verbose = 0 조용히학습, 1 , 2 커질수록 학습내용 다 보여줌*

 *# validation 미리 쪼개놨으면 validation_data=(X_test, y_test) 로 줘도 됨*

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, verbose=1)

 # history 시각화

plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)

plt.plot(history.history[’loss’], label=’Train’)

plt.plot(history.history[’val_loss’], label=’Validation’)

plt.title(’Loss’)

plt.ylabel(’Loss’)

plt.xlabel(’epochs’)

plt.legend

![image.png](image%2016.png)

 *#만약 val_loss가 증가하는 추세가 나오면 과적합이 되고 있다는 뜻임*

 *# 모델 평가*

model.evaluate(X_test, y_test)

 *# 예측값 시각화*

y_pred = model.predict(X_test)

plt.sactter(y_test, y_test, label=’True value’)

plt.scatter(y_test, y_pred, label=’Predicted Value’)

plt.legend()

### 모델 설계 및 학습(이진분류)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

 *# 모델의 설정 분류 → 출력함수 : sigmoid*

model = Sequential()

model.add(Dense(16, input_dim=X_train.shape[1], activation=’relu’))

model.add(Dense(8, activation=’relu’))

model.add(Dense(1, activation=’sigmoid’))

 *#model.summary() 로 확인 가능*

 *# 모델 컴파일 metrics → 중간중간 출력해보기*

model.compile(loss=’binary_crossentropy’, optimizer=’adam’, metrics=[’accuracy’])

 *# 모델 훈련*

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, verbose=1)

 # history 시각화

plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)

plt.plot(history.history[’loss’], label=’Train’)

plt.plot(history.history[’val_loss’], label=’Validation’)

plt.title(’Loss’)

plt.ylabel(’Loss’)

plt.xlabel(’epochs’)

plt.legend

plt.figure(figsize=(12,4))

plt.subplot(1, 2, 2)

plt.plot(history.history[’accuracy’], label=’Train’)

plt.plot(history.history[’val_accuracy’], label=’Validation’)

plt.title(’Accuracy’)

plt.ylabel(’accuracy’)

plt.xlabel(’epochs’)

plt.legend

*# 모델 평가*

model.evaluate(X_test, y_test)

 *# 예측값 시각화*

y_pred = model.predict(X_test) > 0.5

accuracy_score(y_test, y_pred)

plt.sactter(y_test, y_test, label=’True value’)

plt.scatter(y_test, y_pred, label=’Predicted Value’)

plt.legend()

# CNN

## CNN(Convolutional Neural Network)

- 이미지 → 숫자(픽셀)로 이루어진 데이터(가로x세로)
- 컬러 이미지 → 가로x세로x3장 (R,G,B) 겹쳐져있는 3차원형태
- 이미지 전처리 for 딥러닝 : 스케일링, 이미지 증강(Augmentation)
    - 이미지 증강
        
        ![image.png](image%2017.png)
        
- 기존 이미지 처리 → 2차원 데이터를 분리하고 옆으로 붙여서 1차원으로 변환 → 지역적 특성 상실, 계산량 급증
- CNN
    - Convolutional Layer (합성곱층) : 지역적(공간적) 특성 보존, filter(kernel)을 이용한 이미지 추출
    - Pooling Layer (풀링층) : 이미지 데이터 압축 → 정보 손실 없음, 계산량 감소, 파라미터 감소(과적합 방지)

### Convolution layer

- 컨볼루션 레이어(합성곱) : 이미지 * 필터(커널) = 피처맵 (이미지의 각 구간→patch)
    
    ![image.png](image%2018.png)
    
    - Couvolution Layer의 2가지 특징
        - Locality
            - 커털 사이즈 만큼의 작은 구역(patch)의 인접한 pixel들에 대한 correlation 관계를 비선형 필터를 적용하여 추출
            - 이러한 필터를 여러 개 적용하면 다양한 local 특징을 추출 할 수 있음
            - 이미지 전체가 아니라 구간에 대한 특징 추출 가능하다는 의미 → 지역성 살실 해결
        - Parameter Sharing
            - input 상의 모든 patch들을 동일한 kernel을 적용하여 next layer의 output을 출력
            - parameter의 수를 획기적으로 줄임
            - 커널을 자동적으로 생성하기때문에 manual하게 커널을 만들 필요성이 없어졌음
    - 컨볼루션 연산 과정 예시
        
        ![image.png](image%2019.png)
        
    - 컨볼루션 레이어의 피처맵 → 원본 이미지의 특징들을 추출해냄, 반복수행할수록 복잡한 패턴을 추출해냄
    
    ![image.png](image%2020.png)
    
    - 커널의 특성 추출
        - 각 다른 커널을 사용해서 다른 특성을 추출할 수 있음 (blur, shapen 등등)
        
        ![image.png](image%2021.png)
        
    - 컬러 이미지 컨볼루션 → 기존 가로X세로X3장 에서 가로X세로X2장 으로 줄어듬
    
    ![image.png](image%2022.png)
    
    - 패딩 : 이미지 주위를 0으로 감쌈 for 원본 이미지의 크기가 줄어드는것을 방지
    
    ![image.png](image%2023.png)
    
    - 스트라이드 : 필터를 건너뛰는 단위 → 클수록 듬성듬성 필터적용 (메모리 아낄 수 있어짐)
    
    ![image.png](image%2024.png)
    

### Pooling layer

- 풀링 레이어 (메모리 아낄 수 있음)
    - pooling 뉴런은 가중치 없음 (계산, 학습x)
    - 최대, 평균을 이용하여 이미지를 subsampling(부표본 작성)
    - 맥스풀링 : 구간을 나누어 최대 값을 추출 (에버리지 풀링보다 성능이 좋음)
        
        ![image.png](image%2025.png)
        
    - 에버리지풀링 : 구간을 나누어 평균 값을 추출
    
    ![image.png](image%2026.png)
    
    - 풀링 레이어 2가지 특성
        - Positional Invariance
            - 특정 pixel의 정확한 position에 less sensitive
            - 여러 번의 pooling을 거치면 넓은 영역에 걸쳐 같은 효과 발생 → 더 넓게 보면서 학습한다
        - Size 축소
            - 계산량을 크게 줄임
            - 과적합 방지
- flatten → 데이터를 1차원으로 변형 (분류하기위해)

![image.png](image%2027.png)

## CNN기반 인공신경망 학습

![image.png](image%2028.png)

### LeNet-5

![image.png](image%2029.png)

- 데이터 탐색

X_train[-1] *# X_train의 마지막 값불러오기 ← 눈으로 확인 불가능*

![image.png](image%2030.png)

- 데이터의 이미지 보기

plt.figure(figsize=(5,5))

plt.imshow(X_train[0], cmap=’gray’)

![image.png](image%2031.png)

for i in range(9):

plt.subplot(3, 3, i+1)

plt.imshow(X_train[i], cmap=’gray’)

plt.title(”Class: {}”.format(y_train[i])

plt.xticks([])

plt.yticks([])

plt.tight_layout() *#바짝 붙은 그림들 띄어줌*

![image.png](image%2032.png)

- 이미지 데이터 scailing (그대로하면됨)

 *# simple scailing*

X_train_scaled = X_train / 255.0 

X_test_scaled = X_test / 255.0 *#float만들기 위해 .0 붙임*

- 차원증가

 *#손글씨 CNN 처리에서 차원변경이 필요한 이유*

 *#CNN(합성곱 신경망)은 이미지의 공간 구조를 인식해야 하기 때문에 특정 차원 형태를 요구합니다.*

 *#CNN 입력 데이터 형식 요구사항*

 *#CNN은 4차원 데이터를 입력받습니다:*

 *#`text(샘플 수, 높이, 너비, 채널)`*

 *#MNIST 손글씨의 경우:*

- 샘플 수: 훈련 데이터 개수 (예: 60,000)
- 높이: 28 (픽셀)
- 너비: 28 (픽셀)
- 채널: 1 (흑백 이미지)

 #따라서 최종 형태는 **(60000, 28, 28, 1)**이 됩니다.

X_train_scaled = np.expand_dims(X_train_scaled, axis=3)

X_test_scaled = np.expand_dims(X_test_scaled, axis=3)

X_train.shape, X_test.shape 

![image.png](image%2033.png)

 

- **원핫 인코딩**

***#**원핫벡터란?*

*#원핫벡터는 한 개의 값만 1이고 나머지는 모두 0인 벡터입니다*

*#CNN 분류모델의 마지막 레이어에서 보통 소프트맥스 활성화 함수를 사용합니다. 소프트맥스는 각 클래스별 확률을 0~1 범위로 출력합니다. 따라서 라벨도 같은 형식(확률 분포)으로 변환해야 모델이 학습할 수 있습니다. → target 값을 0~1 범위 사이로 변경*

from tensorflow.keras.utils import to_categorical

y_train_onehot = tf.keras.utils.to_categorical(y_train)

y_test_onehot = tf.keras.utils.to_categorical(y_test)

- (참고) tf.data를 이용한 shuffling and batch 구성 (속도 빨라짐) → numpy 데이터를 tensor 형태로 변환

train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_onehot)).shuffle(10000).batch(128)

test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test_onehot)).batch(128) *#test데이터는 셔플 안해도 됨(비경제적)*

### 모델설계(Lenet 구성)

### 일반적 설계는 Conv2D(16, (3,3) → 32, (3,3) → 64, (3,3) 을 많이 사용

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

model = Sequential()

model.add(Conv2D(6, kernel_size=5, padding=’same’, input_shape(28, 28, 1)))

model.add(Activation(’relu’))

model.add(MaxPooling2D(pool_size(2,2), strides(2,2)))

model.add(Conv2D(16, kernel_size=5, padding=’valid’))

model.add(Activation(’relu’))

model.add(MaxPooling2D(pool_size=(2,2), strides(2,2)))

model.add(Flatten())

mdeo.add(Dense(120))

model.add(Activation(’relu’))

model.add(Dense(84))

model.add(Activation(’relu’))

model.add(Dense(10))

model.add(Activation(’softmax’))

model.summary() ← 로 확인 가능

 # complile

model.compile(loss=’categorical_crossentropy’, optimizer=’adam’, metrics=[’accuracy’])

 # 학습

history = model.fit(train_ds, epochs=5, validation_data=test_ds) *# tf.data를 이용한 shuffling and batch 구성에서 사이즈랑 배치는 이미 정해놓음*

 # 평가

score = model.evaluate(test_ds, verbose=0)

print(”마지막 loss =”, score[0])

print(”마지막 accuracy =”, score[1])

 # 평가 시각화

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

plt.plot(history.history[’accuracy’])

plt.plot(history.history[’val_accuracy’])

plt.title(’Accuracy’)

plt.xlabel(’epochs’)

plt.ylabel(’accuracy’)

plt.subplot(1,2,2)

plt.plot(history.history[’loss’])

plt.plot(history.history[’val_loss’])

plt.title(’LOSS’)

plt.xlabel(’epochs’)

plt.ylabel(’accuracy’)

 # 이미지 시각화

fig, axes = plt.subplots(2, 8, figsize=15,4))

axes = axes.ravel()

for i in range (16): ←무작위 이미지 16개 선택하여 표기

idx = np.random.randint(0, len(y_test))

axes[i].imshow(X_test[idx,:])

axes[i].set_xticks([])

axes[i].set_yticks([])

axes[i].set_title(”true={} \npredicted={}”.

format(cifa10_clasees[y_true[idx]], cifa10_classes[y_pred[idx]])))

plt.tight_layout ← 서브플롯들이 겹치지 않도록 레이아웃 조정

plt.show()

 # 히트맵으로 시각화

plt.figure(figsize=(10,8))

sns.heatmap(cm, annot=True, fmd=’d’)

plt.xticks(np.arrange(10), cifa10_classes, rotation=45, fontsize=12)

plt.yticks(np.arragne(10), cifa10_classes, rotation=45, fontsize=12)

plt.xlabel(”true class”)

plt.ylabel(predicted class”)

plt.title(’Confusion Matrix’)

### (참고)

import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

 *# 학습 결과를 저장하기 위한 환경 설정 ← 최저점을 지나가서 멈췄을경우 최저점값 다시 불러오기 위해서 저장 경로 설정*

MODEL_DIR = ‘./mode/’

if not os.path.exists(MODEL_DIR):

os.mkdir(MODEL_DIR)

modelpath=”./model/{epoch:02d}-{val_lossL.2f}.hdf5”

 *#학습 중 모니러팅 할 조건 설정*

checkpointer = ModelCheckpoint(filepath=modelpath, monitor=’val_loss’, verbose=1, save_best_only=True)

 *#학습의 이른 종료를 위한 조건 설정*

early_stopping_callback = EarlyStopping(monitor=’val_loss’, patience=10)

 #학습

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=1, callback=[early_stopping_callback, checkpointer])

 #테스트 정확도 출력

print(”\n Test Accuracy: %.4f” % (model.evaluate(X_test, Y_test)[1]))

![image.png](image%2034.png)

history.history

import pandas as pd

import seaborn as sns

accuracy = history.history[’accuracy’]

loss = history.history[’loss’]

val_accuracy = history.history[’val_accuracy’]

val_loss = history.history[’val_loss’]

 *#데이터 프레임 생성*

df = pd.DataFrame({’epochs’ : range(len(accuracy)), ‘accuracy’: accuracy, ‘loss’ : loss, ‘val_accuracy’ : val_accuracy, ‘val_loss’ : val_loss})

 *# epochs에 따른 loss값의 변화 시각화*

sns.lineplot(x=df[’epochs’], y=df[’val_loss’])

sns.lineploy(x=df[’epochs’], y=df[’loss’])

![image.png](image%2035.png)

### 전이 학습 Transfer Learning

- Transfer Learning
    - 대량의 데이터로 학습한 모델을 전이시켜서 새로운 분야의 적은양의 학습데이터로도 좋은 결과를 낼 수 있도록

![image.png](image%2036.png)

![image.png](image%2037.png)

- Transfer Learning Strategy
    - Strategy 1
        - CNN layer는 freeze하고, 추가한 완전연결층만 새로이 train (내 데이터 적을때)
    - Strategy 2
        - 전체 Layer를 매우 작은 learning rate로 re-train (내 데이터 많을때)
- Transfer Learning 고려사항
    - 목적에 맞는 dataset 선택
    - 보유 데이터의 volume 고려
        - large dataset : 모든 weight 새로이 training
        - medium dataset : Weight의 일부만 training
        - small dataset : 마지막 layer만 fine-tuning
- ImageDataGenrator methods
    - .flow_from_directory
        - 대용량 data를 directory 에서 직접 load (with data augmentation - 이미지 조금씩 조정 가능)
        - directory 구조에 의해 자동으로 label 인식
    - flow_from_directory 사용법
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
    - Instacne 생성
        
        train_data_gen = ImageDataGenerator(resclae = 1/255.)
        
    - flow_from_directory method 호출 :
        
        train_generator = train_data_gen.flow_from_directory(
        
        train_dir,
        
        target_size(150,150), *# 이미지 사이즈 통일*
        
        batch_size=20,
        
        class_mode=’binary’ or ‘categorical’)
        

### 전이학습 실습(tensorflow_hub에 저장되어있는 mobilenet 사용)

- Import Libraries

import tensorflow as tf

import tensorflow_hub as hub

import tensorflow.keras.applications.mobilenet import decode_predictions

- Tensorflow Hub에서 Pre-Trained Mobilenet의 weight를 가져옴

Trained_MobileNet_url = “https://tfhub.dev/google/tf2-preview/mobilenet_V2/classification/2”

Trained_Mobilenet = tf.keras.Sequential([

hub.KerasyLayer(Trained_MobileNet_url, input_shape=(244, 244, 3)) *#처음 학습된 형태 224,224,3 맞춰줘야함*

])

Trained_Mobilenet.input, Trained_Mobilenet.output

- Pre-Trained Mobilnet 평가
    - 임의의 사진을 internet에서 가져옴
        
        from PIL import Image
        
        from urlib import request
        
        from io import BytesIO
        
        url = “https://github.com/ironmanciti/MachineLearningBasic/blob/master/watch.jpg?raw=true”
        
        res = request.urlopen(url).read()
        
        Sample_Image = Image.open(BytesIO(res)).resize((224, 224))
        
        Sample_Image
        
    - 이미지 분류
        
        x = tf.keras.applications.mobilenet.preprocess_input(np.array(Sample_Image))
        
        x.shape
        
        predicted_class = Trained_Mobilenet.predict(np.expand_dims(x, axis=0))
        
        predicted_class
        
        predicted_class.shape
        
        predicted_class.argmax(axis=-1) → 카테고리를 숫자로 알려줘서 읽을 수 없음
        
        decode_prediction(predicted_class[:, 1:]) → 카테고리(문자이름)로 분류될 확률 프린트해줌
        
- 특정 domain 의 Batch Image에 대한 MobileNet 평가 - No Fine Tuning
    - MobileNet은 Flower에 특화된 model이 아니므로 정확도는 낮을것으로 예상
    
    flowers_data_path = tf.keras.utils.get_file(
    
    ‘flower_photos’, ‘https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz’, untar=True)
    
    flowers_data_path ← flower data는 5개의 class로 구성
    
    image_generator = tf.keras.preprocessing.Image.ImageDataGenerator(
    
    preprocessing_function = tr.keras.applications.mobilenet.preprocess_input
    
    )
    
    flowers_data = Image_generator.flow_from_directory(flowers_data_path, target_size(224,224), batch_size=64, shuffle=True)
    
     # next 함수 : yield로 끝나는 def. 끝날때까지 계속 실행됨
    
    input_batch, label_batch = next(flower_data) 
    
    input_batch.shape, label_batch.shape
    
    label_batch[-1]
    
    flower_data.num_classes ← flower class 숫자 확인
    
    flower_data.class_indices ← flower class label 확인용
    
    class_names = {v : k for k , v in flower_data_indices.items()} ← 밸류와 키 값을 바꿈
    
- 10개 이미지 시각화
    
    input_batch[0].min()
    
    plt.figure(figsize=(16,8))
    
    for i in range(10):
    
    plt.subplot(1, 10, i+1)
    
    img = ((imput_batch[i] + 1)*127.5).astype(np,unit8) ← 0~255 사이 숫자로 다시 변환
    
    idx = np.argmax(label_batch[i])
    
    plt.title(class_names[idx])
    
    plt.imshow(input_batch[i])
    
    plt.axis(’off’) ← 눈금 off
    
- 임의의 꽃 image 1개를 선택하여 prediction 비교
    
    prediction = Trained_Mobilenet.predict(input_batch[2:3])
    
    decode_predictions(prediction[:, 1:])
    
- 전이학습 MODEL을 Flower분류에 적합한 model로 Retrain
    - Fine Tuning을 위해 head가 제거된 model 을 download
    
    extractor_url = “https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2”
    
    extractor_layer = hub.KerasLayer(extractor_url, input_shape=(224, 224, 3))
    
    extractor_layer(input_batch).shape
    
- Top layer에 Dense layer 추가
    
    extractor_layer.trainable = False ← 기존 트레인된거 잠금
    
    model = tf.keras.Sequential([
    
    extractor_layer,
    
    tf.keras.layers.Dense(5, activation=’softmax’)
    
    ])
    
    model.summary()
    
    model.input, model.output
    
    - 훈련 안된 모델에 넣어봄
    
    prediction = model(input_batch)
    
    prediction.shape
    
    - output shape이 정확한지 training 전에 사전 check
    
    model.compile(losdd=’categorical_crossentropy’, optimizer=’adam’, metrics=[’accuracy’])
    
    hitsory = model.fit(flower_data, epochs = 30)
    
- flower 분류 전문으로 fine tuning 된 모델 평가

y_pred = model.predict(input_batch)

y_pred = np.argmax(y_pred, axis=-1)

y_pred

y_true = np.argmax(label_batch, axis=-1)

y_true

sum(y_pred == y_true) / len(y_true) ← accuracy 점수 확인

## 똑똑한 인공신경망 활용(성능 좋은 CNN) - 참고용

- VGGNet
    - 레이어를 단단하게 쌓음
    - 3x3 kernel_size를 연속적으로 사용하면 필드를 더 늘리지 않아도 충분하다는 연구결과 발표 (메모리 사용량 아낄 수 있어짐)

![image.png](image%2038.png)

![image.png](image%2039.png)

- GoogLeNet
    - 반복되는 모듈사용
    - 보조분류기를 통해서 역전파에 도움
    - 1x1 커널연산을 통해서 채널의 사이즈를 줄임

![image.png](image%2040.png)

![image.png](image%2041.png)

![image.png](image%2042.png)

![image.png](image%2043.png)

- ResNet
    - 건너뛰는 레이어 활용 → 중간 중간 학습결과를 확인할 수 있음 (정확도높아짐)

![image.png](image%2044.png)

# RNN

### Sequence data

- speech recognition : 파동의 연속 → 단어의 연속으로 변환
- Music genaration : 연속된 음표 출력
- sentiment classification : 평점으로 부정/긍정 판단
- DNA분석 : 염기서열 → 질병유무, 단백질 종류 등
- 자동번역
- Video activity recognition : 연속된 장면 → 행동판단
- Fianacial Dta : 시계열 자료 → 주가 ,환율 예측 등

## RNN(Recurrent Nueral Network)

- Sequence data에 특화되어있음
- 기억 능력을 갖고 있음 : 네트워크의 기억 → 지금까지 입력 데이터를 요약한 정보
- 새로운 입력이 들어올 때 마다 네트워크 자신의 기억을 조금씩 수정
- 입력을 모두 처리하고 난 후 네트워크에게 남겨진 기억은 시퀀스 전체를 요약하는 정보
- 새로운 단어마다 계속해서 반복 → Recurrent(순환적)
- 과거 은닉층에서 처리했던 데이터를 다시한번 처리
    - BPTT(Backpropagation Through Time)으로 parameter 학습
    - RNN을 순서대로 펼쳐 놓으면 weight를 공유하는 deep한 neural network이 됨

![image.png](image%2045.png)

![image.png](image%2046.png)

![image.png](image%2047.png)

- 1:N, N:1, N:M, N:N 다양한 구성 가질 수 있음

![image.png](image%2048.png)

- 시간을 거슬러서 역전파 수행 가능(계산량 많음)

![image.png](image%2049.png)

### Simple RNN

![image.png](image%2050.png)

### LSTM ← 거의 다 이거 씀

- forget 추가 ← 기억해야할 데이터 조절

![image.png](image%2051.png)

### GRU

- update ← 기억해야 할 데이터 선택

![image.png](image%2052.png)

### RNN Overview

- Input → 3차원으로 입력
    - Batch size
    - Time step
    - Input features
        - Univeriate - one
        - Multiveriate - many

### LSTM → 수열 패턴 인식

문제 ) data는 0~99 까지의 연속된 숫자이고, target은 (1~101)*2로 구성.

입력 data에 대응하는 출력 data를 예측하는 model을 LSTM으로 작성

연속된 5개의 숫자를 보고 다음 숫자를 알아맞추도록 LSTM을 이용한 model 작성

- import

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM

- Training Data 작성 요령

numbers = [[i] for i in range(105)]

data = []

target = []

for i in range(5, len(numbers)): ← 5부터 105까지

data.append[numbers[i-5:i]) ← data : 0에서부터 5 사이

target_append(numbers[i][0]*2)

- List → array로 변경

data = np.array(data, dtype=”float32”)

target = np.array(target, dtype=”float32”)

- 모델 생성

model = Sequential()

model.add(LSTM(16, input_shape=(5,1)))

model.add(Dense(1))

model.compile(optimizer=’adam’, loss=’mae’, metircs=’mae’)

- 훈련

model.fit(data, target, epoch=500, validation_split=0.2)

### LSTM → 주식 가격 예측(정확한 가격이 아닌 추세를 파악하기 위함)

- 불러오기

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

!pip install yfinance

import yfinance as yf ← 야후 주식 불러오기 위함

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dropout, Dense

- APPLE 주식 가격 불러오기

appl = yf.download(’AAPL’, start=’2015-01-01’, end=’2022-04-21’, progress=False)

import seaborn as sns

sns.lineplot(x=aapl.index, y=’Close’, data=aapl) ← 주식가격 시각화

plt. title(’APPLE stock price’)

- training dataset 생성

hist = []

target = []

window = 3

close = appl[’Close’].values

for i in range(len(close) - window):

x = close[i : i+window]

y = close[i+window]

hist.append(x)

target.append(y)

 #hist의 각 요소는 window개 timestpe의 list임. 1씩 증가하기 때문에 hist의 두번째 요소의 마지막 항목은 target의 첫 번째 요소와 같아야함. 또한 마지막 숫자가 같아야 함.

- 리스트를 array로 변환

hist = np.array(hist)

target = np.array(target).reshape(-1,1)

- 분할

 # train_test_split 으로 섞지 않음 ← 순서가 랜덤이 되면 안되는 시계열 데이터 이기 때문. 날짜를 기준으로 분할함

split = len(hist) - 100

X_train = hist[:split]

X_test = hist[split:]

y_train = target[:split]

y_test = target[split:]

- 스케일링

sc1 = MinMaxScaler()

X_train_scaled = sc1.fit_transform(X_train)

X_test_scaled = sc1.transform(X_test)

sc2 = MinMaxScaler()

y_train_scaled = sc2.fit_transform(y_train)

y_test_scaled = sc2.transform(y_test)

X_train_scaled = X_train_scaled.reshape(-1, window, 1)

X_test_scaled = X_test_scaled.reshape(-1, window, 1)

- 모델 생성

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(window,1), dropout=0.2))

model.add(LSTM(32, return_sequences=True, dropout=0.2)) ← return_sequences : 다음 LSTM에 연결

model.add(LSTM(16, dropout=0.2))

model.add(Dense(1))

model.compile(optimizer=’adam’, loss=’mse’)

- 훈련

history = model.fit(X_Train_scaled, y_train_scaled, epochs=30, batch_size=16)

- 시각화

plt.plot(hitstory.history[’loss’])

plt.xlabel(’epoch’)

plt.ylabel(’mse’)

- 예측

pred = model.predict(X_test_scaled)

plt.figure(figsize=(12,6))

plt.plot(np.concatenate((y_train_scaled.faltten(), y_test_sacled.faltten())), label=’original data’)

plt.plot(np.concatenate((y_train_sacled.flatten(), pred.flatten())), label = ‘prediction’)

plt.legend()

### LSTM → 영화평 감성분석 실습(긍정/부정 이진분류)

![image.png](image%2053.png)

- 데이터 불러오기

from [tensorflow.keras.datasets.imdb](http://tensorflow.keras.datasets.imdb) import load_data

num_words = 20000 *# 2만개의 단어만 학습에 활용*

(x_train, y_train), (x_test, y_test) = load_data(num_words = num_wrods)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

![image.png](image%2054.png)

- 데이터 탐색하기

x_train[0] *#이미 단어가 숫자로 매칭되어있는 데이터셋임(인코딩완료)*

![image.png](image%2055.png)

*len(x_train[0]) #토큰이 218개*

![image.png](image%2056.png)

y_train[0]

→ 1

import numpy as np

np.unique(y_train)

→ array([0, 1])

import pandas as pd

df = pd.DataFrame(x_train, columns=[’review’])

df[’length’] = df[’review’].apply(lamvda x: len(df[’review’][x]))

df[’label’] = y_train

df.head()

![image.png](image%2057.png)

import seaborn as sns

sns.histplot(df[’length’], kde=True)

![image.png](image%2058.png)

df[’length’].describe() *#패딩의 기준치 설정하기 위해 확인* 

 *#입력 데이터의 길이가 다르면 배치처리를 할 수 없어서 패딩을 통해 입력 데이터 길이를 통일시켜야함*

- 데이터 전처리

from tensorflow.keras.preprocessing.sequence import pad_sequences

 *#리뷰 문장 중 300 단어만 활용하는 패딩*

maxlen = 300

x_train = pad.sequences(x_train, maxlen = maxlen)

x_test = pad_sequences(x_test, maxlen = maxlen)

x_train.shape, x_test.shape

![image.png](image%2059.png)

- 신경망 설계하기
    
    from tensorflow.keras.models import Sequential
    
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
    
    model = Sequential()
    
     *# 입력 데이터를 128차원 벡터로 임베딩*
    
    *# `python"좋다"  → [0.8, 0.2, -0.1, ...]   ← 긍정적 의미 반영
    # "나쁘다" → [-0.7, -0.3, 0.2, ...]  ← 부정적 의미 반영
    # "훌륭하다" → [0.9, 0.1, -0.05, ...] ← "좋다"와 비슷한 벡터`*
    
    *# 벡터 공간에서는 **의미가 비슷한 단어들이 서로 가까이 위치**합니다. RNN이 이를 통해 단어의 의미를 이해할 수 있습니다.*
    
    model.add(Embedding(num_words, 128))
    
     *# 양방향 LSTM*
    
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    
    model.add(Bidirectional(LSTM(128)))
    
     *# 분류기*
    
    model.add(Dense(1, activation = ‘sigmoid’))
    
    model.summary()
    

![image.png](image%2060.png)

 *# 모델 구조 시각화*

from tensorflow.keras.utils import plot_model

plot_model(model, to_file=’graph.png’)

![image.png](image%2061.png)

model.compile(loss=’binary_crossentropy’, optimizer=’adam’, metrics=[’accuracy’])

import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

 *# 학습 결과를 저장하기 위한 환경 설정 ← 최저점을 지나가서 멈췄을경우 최저점값 다시 불러오기 위해서 저장 경로 설정*

MODEL_DIR = ‘./mode/’

if not os.path.exists(MODEL_DIR):

os.mkdir(MODEL_DIR)

modelpath=”./model/{epoch:02d}-{val_lossL.2f}.hdf5”

 *#학습 중 모니러팅 할 조건 설정*

checkpointer = ModelCheckpoint(filepath=modelpath, monitor=’val_loss’, verbose=1, save_best_only=True)

 *#학습의 이른 종료를 위한 조건 설정*

early_stopping_callback = EarlyStopping(monitor=’val_loss’, patience=5)

history = model.fit(x_train, y_train, batch_size=256, epochs=50, validation_slpit=0.2, callbacks=[early_stopping_callback, checkpointer])

![image.png](image%2062.png)

history_df = pd.DataFrame(history.history)

history_df.loc[0:, [’accuracy’, ‘val_accuracy’]].plot()

![image.png](image%2063.png)

model.evaluate(x_test, y_test)

![image.png](image%2064.png)

## Autoencoder → 비지도 학습(비선형 변환)

- PCA : linear transformation. 선형대수학의 특이값분해를 이용, 분산을 최대한 보존하면서 서로 직교하는 새 축을 찾아 차원축소
- Autoencoder : non-linear transformation. most relevant feature을 추출하여 차원을 감소
- 정보의 압축, Noise제거, 유사한 image 검색, image변형에 의한 새로운 image 생성, pre-training 등에 사용

![image.png](image%2065.png)

![image.png](image%2066.png)

- latent representation : 잠재적 표현 → 머신러닝은 사람보다 잘 찾아냄. 이미지나 시계열 등 포함
- input, output → 3차원
- encoder 부분 : latent representation 학습 (2차원)
- code 부분 : Bottleneck, Latent Space(잠재공간), feature, code 등으로 불림
- decoder 부분 : 원래의 이미지를 복원(Reconstruction)

### Autoencoder 작성 및 시각화

- 불러오기

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

- 3차원 데이터 생성

np.random.seed(42)

m = 100

angles = np.random.rand(m)*3*np.pi/2-0.5

data = np.empty((m,3))

data[:,0] = np.cos(angles) + np.sin(angles/2 + 0.1*np.random.randn(m)/2

data[:,1] = np.sin(angles)*0.7+0.1*np.random.randn(m)/2

data[:,2] = data[:,0]*0.1 + data[:,1]*0.3+0.1*np.random.randn(m)

data.shape

- 3차원 데이터 시각화

X_train = data

ax = plt.axes(projection=’3d’)

ax.scatter(X_train[:,0], X_train[:, 1], X_train[:, 2], c=X_train[:, 0], cmap=’Reds’)

- Autoencdoer model 작성, fit
- 3차원 data를 2차원으로 축소

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

encoder = Dense(2, input_shape=(3,))

decoder = Dense(3, input_shape=(2,))

autoencoder = Sequential([encoder, decoder])

autoencoder.summary()

autoencoder.compile(loss=’mse’, optimizer=keras.optimizers.SGD(learning_rate=0.1))

history = autoencoder.fit(X_train, X_train, epochs=200) ← autoencoder 자기 자신을 학습시킴

encodings = encoder.predict(X_train)

- encoder output 시각화

fig = plt.figure(figsize=(4,3))

plt.scatter(encodings[:, 0], encodings[:, 1])

- Decoder를 이용한 data 복원

decodings = decoder.predict(encodings)

ax = plt.axes(projection=’3d’)

ax.scatter(decodings[:, 0], decodings[:, 1], decodings[:, 2], c=encodings[:, 0], cmap=”Reds”)

### Deep Auto-Encoders 연습문제(이미지 압축 후 복구)

- 데이터 불러오기

import numpy as np

import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras import regularizers

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

- Sample Image 시각화

(X_train, _), (X_test,_) = fashion_mnist.load_data()

fig, ax=plt.subplots(1, 10, figsize=(20,4))

for i in range(10):

ax[i].imshow(X_test[i], cmap=’grey’)

ax[i].set_xticks([])

ax[i].set_yticks([])

- data normalization, 2차원 모델을 1차원으로 변경

X_train = X_train / 255.

X_test = X_test / 255.

X_train = X_train.reshape(-1, 784) < 784 : 28x28

X_test = X_test.reshape(-1, 784)

- Stacked autoencdoder 작성

input = input(shape=(784,))

 #stacked autoencoder

x = Dense(units=128, activation=’relu’)(input)

x = Dense(units=64, activation=’relu’)(x)

encoder = Dense(unit32, activation=’relu’)(x)

x = Dense(units=64, activation=’relu’)(encoder)

x = Dense(units=128, activation=’relu’)(x)

decoder = Dense(units=784, activation=’sigmoid’)(x)

 # autoencoder model

encoder_model = Model(inputs=input, outputs=encoder)

autoencoder = Model(inputs=input, outputs=decoder)

autoencoder.compile(loss=’binary_crossentropy’, optimizer=’amam’)

autoencoder.summary()

plot_model(autoencoder, show_shapes=True) ← 설계도 시각화

history = autoencoder.fit(X_train, X_train, epochs=50, shuffle=True, batch_size=256, validation_data=(X_test, X_test)

plt.plot(history.history[’loss’]), label=’train loss’)

plt.plot(history.history[’val_loss’], label=’test loss’)

plt.lengend()

- 결과 시각화

fig, ax = plt.subplots(3, 10, figsize=(20,8))

for i in range(10):

ax[0, i].imshow(X_test[i].reshape(28, 28), cmap=’gray’)

img = np.expand_dims(X_test[i], axis=0)

ax[1, i].imshow(encoder_model.predict(img, verbose=0).reshape(8,4), cmap=’gray’)

ax[2,i].imshow(autoencoder.predict(img, verbose=0).reshpae(28,28), cmap=’gray’)

ax[0,i].axis(’off’)

ax[1,i].axis(’off’)

ax[2,i].axis(’off)

## GAN : 생성적 적대 모델 (Genrative Adversarial Network)

- GAN
    - Computer가 이미지, 목소리, 악기소리 등을 실제와 같이 생성
    - 위조를 담당하는 Genrator(생성자)와 위조를 판별하는 Discriminator(감별자)의 두개의 Deep Neural Network로 구성
- Probability Basics
    - Pixel : 64x64x3 인 Image의 확률분포임
    - X : 0~255의 값을 갖는 64x64x3 인 고차원 vector
    - Image 특성에 따른 확률분포
        - 안경을 쓴 사람? - 안경의 특성과 관련된 pixel → X1
        - 금발 ? - 금발과 관련된 pixel → X2
- 결합확률을 학습 → 그럴듯 한 이미지 중 가장 높은확률
    
    ![image.png](image%2067.png)
    
- 생성모델 : 실제 존재하지 않지만 있을 법한 이미지를 생성 할 수 있는 모델
    - 분류모델 : 결정 경계를 학습
    - 생성모델 : 각 클라스의 분포를 학습
    
    ![image.png](image%2068.png)
    
- GAN 의 목표 : 두개 확률 분포의 차이를 줄여줌

![image.png](image%2069.png)

![image.png](image%2070.png)

- 검은 점선 : 원본 데이터 이미지의 분포
- 파란 점선 : discriminator distribution
- 녹샌 선 : generator distribution
- Genrative vs Discriminator
    - Discriminator(분류자)
        - input data의 feature를 기준으로 label 예측
        - p(yㅣX) → the probability of y given X
    - Genrative(생성자)
        - 주어진 label을 기준으로 feature 예측
        - p(Xㅣy) → the probability of feature given y
- GAN process - Training of Discriminator
    - Generator는 random number를 취하여 random image(fake image) 생성
    - Generator가 생성한 가짜 image와 actual dataset의 진짜 image를 discriminator에게 공급
    - Discriminator는 진짜 image는 1, 가짜 image는 0 을 출력하도록 이진분류 훈련
- GAN process - Training of Generator
    - Discriminator 출력의 corssentropy 값을 1과 비교하여 차이분을 손실로 인식하여 backpropagation으로 보정
    - 즉 Discriminator을 속이는게 주 목적임
- GAN flow

![image.png](image%2071.png)

- Discriminator 의 목표
    - dataset을 진짜로 인식하고, Generator에서 공급되는 image를 fake로 구분
- Generator의 목표
    - Discriminator가 진짜로 인식할 fake image 생성(Gaussian random noise로부터 image생성)
- Discriminator가 너무 강하면 항상 0과 1에 근사한 값이 나오므로 Generator가 gredient를 얻을 수 없고, Generator가 너무 smart하면 discriminator의 weakness를 계속 이용 (exploitation)하여 discriminative가 false negative를 predict 하도록 함 → Learning rate 조절

### GAN 모델 작성 실습

- 데이터 불러오기

import tensorflow as tf

import tensorflow.keras as keras

import numpy as np

import matploylib.pyplot as plt

from IPython import display ← 주피터 노트북 사이사이 gan 샘플 출력

- 훈련되는 동안 GAN의 샘플 출력

def plot_multiple_images(images, n_cols=None):

“visualizes fake images”

display.clear_output(wait=False)

n_cols = n_cols or len(images)

n_rows = (len(images) - 1) // n_cols + 1

if images.shape[-1] == 1:

images = np.squeeze(images, axis=-1)

plt.figure(figsize=(n_cols, n_rows))

for index, image in enumerate(images):

plt.subplot(n_rows, n_cols, index + 1)

plt.imshow(image, cmap=’binary’)

plt.axis(”off”)

- Download and Prepare the Dataset

(X_train, _), _ = keras.datasets.mnist.load_data() ← X_train 데이터만 받겠다

X_train = X_train.astype(”float32”)/255

- 훈련 이미지의 배치를 생성 (훈련하는 동안 모델에 공급할 수 있도록)

dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000)

dataset = dataset.batch(128, drop_remainder=True).prefetch(1)

- Build the model
- Generator
    - SELU : GAN에 적합한 활성화 함수→ 최종 Dense 네트워크는 0과 1사이의 픽셀 값을 생성하기 원하므로 sigmoid로 활성화. 그런 다음 MNIST 데이터 차원에 맞게 reshape
    
     # declare shape of the noise input
    
    random_normal_dimensions = 32
    
     # build the generator model
    
    generator = keras.models.Sequential([
    
    keras.layers.Dense(64, activation=’selu’, input_shape=[random_noraml_dimensions]),
    
    keras.layers.Dense(128, activation=’selu’),
    
    keras.layers.Dense(784, activation=’sigmoid”),
    
    keras.layers.Reshape([28,28])
    
    ])
    
- 훈련되지 않은 generator의 샘플 출력

 # generate a batch of noise input (batch size=16)

test_noise = tf.random.normal([16, random_normal_dimensions])

 # feed the batch to the untrained generator

test_image = generator(test_noise)

 # visualize sample output

plot_multiple_images(test_image, n_cols=4)

- Discriminator

 # build the discriminator model

discriminator = keras.models.Sequential([

keras.layers.Flatten(input_shape=[28,28]),

keras.layers.Dense(128, activation=’selu’),

keras.layers.Dense(64, activation=’selu’),

keras.layers.Dense(1, activation=’sigmoid”),

])

- GAN 모델 구축

gan = keras.models.Sequential([generator, discriminator])

gan.summary()

discriminator.compile(loss=’binary_crossentropy’, optimizer=’rmsprop’)

discriminator.trainable = False

gan.compile(loss=’binary_crossentropy’, optimizer=’rmsprop’)

- GAN train
    - 1단계 : 판별자 훈련
    - 2단계 : 생성자 훈련

def train_gan(gan, dataset, random_normal_dimensions, n_epochs=50):

generator, discriminator = gan.layers

for epoch in range(n_epochs):

print(”Epoch {}/{}”.format(epoch + 1, n_epochs))

for real_images in dataset:

batch_size = real_images.shape[0]

noise = tf.random.normal(shape=[batch_size, random_noraml_dimensions])

fake_images = generator(noise)

mixed_images = tf.concat([fake_images, real_images], axis=0)

discriminator_labels = tf.constant([[0,]] * batch_size + [[1.]] * batch_size)

discriminator.trainable = True

discriminator.train_on_batch(mixed_images, discriminator_labels)

noise = tf.random.normal(shape=[batch_size, random_normal_dimensions])

generator_labels = tf.constant([[1.]] * batch_size)

discriminator.trainable = False

gan.train_on_batch(noise, generator_labels)

plot.multiple_images(fake_images, 8)

plt.show()

train_gan(gan, dataset, random_normal_dimensions, n_epochs=20)

## 인코더, 디코더, 어텐션(참고용)

- 인코더 : 압축, 디코더 : 해석 → 압축해제
- Sequence to Sequence
    - 마지막 input에 결국 모든 input값이 들어있음
    - 모든 input 값을 마지막 input으로 사용
    - 번역 프로그램 구성 시 Encoder, Decoder 신경망 따로따로 구성 → sequence to sequence

![image.png](image%2072.png)

![image.png](image%2073.png)

![image.png](image%2074.png)

![image.png](image%2075.png)

- Attention score
    - 기존의 S2S에서 동적으로 컨텍스트 벡터 할당

![image.png](image%2076.png)

- 트랜스포머
    - RNN을 제거
    - Attention 에만 집중
    - 인코더에서 셀프 어텐션 사용

![image.png](image%2077.png)

![image.png](image%2078.png)

- PLM 모델
    - step1. 최대한 많은 학습
    - step2. 목적에 맞게 파인튜닝
    - 사전 학습 내용을 목적에 맞게 전이시킴

![image.png](image%2079.png)

![image.png](image%2080.png)