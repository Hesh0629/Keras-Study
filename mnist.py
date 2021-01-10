#2020.12.20 D+104
#숫자 이미지를 읽고 학습하여 숫자를 맞히도록 훈련.
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist

(train_images,train_labels), (test_images, test_labels)=mnist.load_data() #데이타 셋 로드

train_images=train_images.reshape(60000, 28*28) #(sample수, 28*28 사이즈)
train_images=train_images.astype('float32')/255 #이미지 분석에서는 float32를 255로 나눈것을 이용

test_images=test_images.reshape(10000, 28*28)
test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels) #to_categorical ->라벨을 원 핫 벡터로 빠르게 바꿔줌 (해당 인덱스만 1, 나머진 0)
test_labels=to_categorical(test_labels)

network = models.Sequential() #이용할 모델은 순차모델 (레이어를 순차적으로 쌓음)
network.add(layers.Dense(512,activation ='relu',input_shape=(28*28,))) #비선형층으로 만들기 위해 활성함수를 relu(음수를 0으로 만듬)이용 (outuput = relu(w,input) + b)
network.add(layers.Dense(10,activation='softmax')) #각 10개의 숫자(결과)에 대한 확률이 들어있는 배열 반환 (sum=1)
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics='acc') #옵티마이저는 rmsprop 손실함수는 보통 crossentropy 이용, 측정지표는 정확도(acc)
network.fit(train_images,train_labels,epochs=5,batch_size=128) #모든 자료를 5번 반복하고 1회 반복하는 동안 한번에 들어가는 자료는 128개 (SGD업데이트시)

test_loss, test_acc=network.evaluate(test_images,test_labels) #network.evaluate의 결과는 (loss_first_output, acc_first_output, loss_second_output, acc_second_output, ...)
print('test_acc:', test_acc)
