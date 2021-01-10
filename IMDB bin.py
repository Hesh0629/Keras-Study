#2020.12.20 D+104
#영화 평가를 보고 긍정리뷰인지 부정적인 리뷰인지 판단
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
(train_data, train_labels), (test_data,test_labels)=imdb.load_data(num_words=10000) #자주 이용하는 단어 10,000개만 이용
#imdb 데이터셋은 numpy 배열임
def one_hot(seq,dimension=10000): #to_categorical의 입력값으로 numpy 배열을 못쓰기 때문에 직접 선언해서 사용
  result = np.zeros((len(seq),dimension))
  for i,seq in enumerate(seq):
    result[i,seq]=1. #result[i][seq]의 위치만 1, 나머지는 0
  return result
#x는 샘플 y는 라벨
x_train =one_hot(train_data)
x_test=one_hot(test_data)
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

model=models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #확률을 해석하기 위해 0부터 1사이로 값을 압축하는 sigmoid함수 이용
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics='acc')
x_val=x_train[:10000] #훈련 데이터와 평가 데이터를 나눔
x_train=x_train[10000:]
y_val=y_train[:10000]
y_train=y_train[10000:]
history = model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'go',label='Training acc')
plt.plot(epochs, val_acc,'g',label='validation acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
