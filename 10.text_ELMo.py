# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:11:25 2021

@author: xck
"""

from elmoformanylangs import Embedder
import numpy as np
from tensorflow.keras.preprocessing import sequence
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from utils import *

e = Embedder('./zhs.model/')


#padding
def pad_sent(x, max_len):
    if len(x)>max_len:
        return x[:max_len]
    else:
        return x+['']*(max_len-len(x))
    
    
#手写使用ELMo变换后的生成器generator
def batch_generator(x, y, batch_size=64):
    n_batches_per_epoch = len(x)//batch_size
    for i in range(n_batches_per_epoch):
        x_batch = e.sents2elmo([pad_sent(sent,30) for sent in x[batch_size*i:batch_size*(i+1)]])
        y_batch = y[batch_size*i:batch_size*(i+1),:]
        yield np.array(x_batch), y_batch
        
        
#定义网络结构
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
import numpy as np

class ELMoTextClassifier(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=5,
                 last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        embedding = Input((self.maxlen, self.embedding_dims,))
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=embedding, outputs=output)
        return model
    
    


# 数据文件夹
data_dir = "./processed_data"

# 神经网络配置
maxlen = 30
batch_size = 64
max_features = 40001
embedding_dims = 1024
epochs = 8

print('数据预处理与加载数据...')
# 获得 词汇/类别 与id映射字典
categories, cat_to_id = read_category()
# 全部数据
x, y = read_files(data_dir)
data = list(zip(x,y))
del x,y
# 乱序
random.shuffle(data)

# 切分训练集和测试集
train_data, test_data = train_test_split(data)
# 对文本的词id和类别id进行编码
x_train = [content[0] for content in train_data]
y_train = to_categorical(encode_cate([content[1] for content in train_data], cat_to_id))
x_test = [content[0] for content in test_data]
y_test = to_categorical(encode_cate([content[1] for content in test_data], cat_to_id))

print('构建模型...')
model = ELMoTextClassifier(maxlen, max_features, embedding_dims).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('训练...')
# 设定callbacks回调函数
my_callbacks = [
    ModelCheckpoint('./cnn_model.h5', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
]

# fit拟合数据
history = model.fit_generator(generator=batch_generator(x_train, y_train),
          epochs=epochs,
          callbacks=my_callbacks,
          validation_data=batch_generator(x_test, y_test),
          steps_per_epoch=len(y_train)//batch_size,
          validation_steps=len(y_test)//batch_size)


import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves :CNN',fontsize=16)
fig1.savefig('loss_cnn.png')
plt.show()


fig2=plt.figure()
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves : CNN',fontsize=16)
fig2.savefig('accuracy_cnn.png')
plt.show()

