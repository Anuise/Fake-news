
import pandas as pd

train = pd.read_csv('./train.csv', index_col=0, encoding='utf-8').astype(str)

cols = ['title1_zh','title2_zh', 'label']
train = train.loc[:, cols]

import jieba.posseg as pseg
import time

def jieba_tokenizer(text):       
    words = pseg.cut(text)    
    return ' '.join([word for word, flag in words if flag != 'x'])

s1_time = time.time()
train['title1_tokenized'] = train.loc[:, 'title1_zh'].apply(jieba_tokenizer)
e1_time = time.time()

s2_time = time.time()
train['title2_tokenized'] = train.loc[:, 'title2_zh'].apply(jieba_tokenizer)
e2_time = time.time()

print('花費時間:%s'%(e1_time-s1_time))
print('花費時間:%s'%(e2_time-s2_time))

import keras
MAX_NUM_WORDS = 10000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)

corpus_x1 = train.title1_tokenized
corpus_x2 = train.title2_tokenized
corpus = pd.concat([corpus_x1, corpus_x2])

tokenizer.fit_on_texts(corpus)
x1_train = tokenizer.texts_to_sequences(corpus_x1)
x2_train = tokenizer.texts_to_sequences(corpus_x2)

MAX_SEQUENCE_LENGTH = 20
x1_train = keras.preprocessing.sequence.pad_sequences(x1_train,maxlen=MAX_SEQUENCE_LENGTH)
x2_train = keras.preprocessing.sequence.pad_sequences(x2_train,maxlen=MAX_SEQUENCE_LENGTH)

import numpy as np

# 定義每一個分類對應到的索引數字
label_to_index = {
    'unrelated': 0, 
    'agreed': 1, 
    'disagreed': 2
}

# 將分類標籤對應到剛定義的數字
y_train = train.label.apply(lambda x: label_to_index[x])
y_train = np.asarray(y_train).astype('float32')

y_train = keras.utils.to_categorical(y_train)

from sklearn.model_selection import train_test_split

VALIDATION_RATIO = 0.1
RANDOM_STATE = 9527
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, test_size = VALIDATION_RATIO, random_state = RANDOM_STATE)

# 基本參數設置，有幾個分類
NUM_CLASSES = 3
# 在語料庫裡有多少詞彙
MAX_NUM_WORDS = 10000
# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 20
# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256
# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128

# 建立孿生 LSTM 架構（Siamese LSTM）
from keras import backend as K
from keras import Input
import keras.layers as layers
from keras.engine import Layer
from keras.layers import Embedding,LSTM, concatenate, Dense
from keras.models import Model, load_model
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np

# Initialize session
sess = tf.Session()
K.set_session(sess)

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=0), as_dict=True, signature='default', )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)



# 分別定義 2 個新聞標題 A & B 為模型輸入
# 兩個標題都是一個長度為 20 的數字序列
top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='string')
bm_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='string')

# 詞嵌入層
# 經過詞嵌入層的轉換，兩個新聞標題都變成
# 一個詞向量的序列，而每個詞向量的維度
embedding_layer = ElmoEmbeddingLayer()
top_embedded = embedding_layer(top_input)
bm_embedded = embedding_layer(bm_input)

# 串接層將兩個新聞標題的結果串接單一向量
# 方便跟全連結層相連
merged = concatenate([top_embedded, bm_embedded], axis=-1)

# 全連接層搭配 Softmax Activation
# 可以回傳 3 個成對標題
# 屬於各類別的可能機率
dense =  Dense(units=NUM_CLASSES, activation='softmax')
predictions = dense(merged)

# 我們的模型就是將數字序列的輸入，轉換
# 成 3 個分類的機率的所有步驟 / 層的總和
model = Model(inputs=[top_input, bm_input], outputs=predictions)

from keras.utils import plot_model
plot_model(
    model, 
    to_file='model.png', 
    show_shapes=True, 
    show_layer_names=False, 
    rankdir='LR')

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# 決定一次要放多少成對標題給模型訓練
BATCH_SIZE = 1

# 決定模型要看整個訓練資料集幾遍
NUM_EPOCHS = 1

x1_train = x1_train.astype(str)
x2_train = x2_train.astype(str)
# y_train = y_train.astype(str)
# 實際訓練模型
history = model.fit(
    # 輸入是兩個長度為 20 的數字序列
    x=[x1_train, x2_train], 
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    # 每個 epoch 完後計算驗證資料集
    # 上的 Loss 以及準確度
    validation_data=([x1_val, x2_val],y_val),
    # 每個 epoch 隨機調整訓練資料集
    # 裡頭的數據以讓訓練過程更穩定
    shuffle=True
)

import pandas as pd
test = pd.read_csv('./train.csv', index_col=0, encoding='utf-8').astype(str)

# 以下步驟分別對新聞標題 A、B　進行
# 文本斷詞 / Word Segmentation
test['title1_tokenized'] = test.loc[:, 'title1_zh'].apply(jieba_tokenizer)
test['title2_tokenized'] = test.loc[:, 'title2_zh'].apply(jieba_tokenizer)

# 將詞彙序列轉為索引數字的序列
x1_test = tokenizer.texts_to_sequences(test.title1_tokenized)
x2_test = tokenizer.texts_to_sequences(test.title2_tokenized)

# 為數字序列加入 zero padding
x1_test = keras.preprocessing.sequence.pad_sequences(x1_test, maxlen=MAX_SEQUENCE_LENGTH)
x2_test = keras.preprocessing.sequence.pad_sequences(x2_test, maxlen=MAX_SEQUENCE_LENGTH)    

# 利用已訓練的模型做預測
predictions = model.predict([x1_test, x2_test])

index_to_label = {v: k for k, v in label_to_index.items()}

test['Category'] = [index_to_label[idx] for idx in np.argmax(predictions, axis=1)]

submission = test.loc[:, ['Category']].reset_index()

submission.columns = ['Id', 'Category']
print(submission.head())