# coding: UTF-8
"""
    @author: samuel ko
    @date: 2018/12/12
    @link: https://blog.csdn.net/zwqjoy/article/details/80493341
"""
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN
from keras.utils import np_utils

# fix random seed for reproducibility
numpy.random.seed(5)

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print(len(alphabet))
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)
# 我们运行上面的代码，来观察现在我们的input和output数据集是这样一种情况
# A -> B
# B -> C
# ...
# Y -> Z

# 喂入网络的特征为 [batch_size, time_step, input_dim] 3D的Tensor
# 用易懂的语言就是: time_step为时间步的个数, input_dim为每个时间步喂入的数据
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# print(X)
# [[[ 0]]
#  [[ 1]]
#  [[ 2]]
#  [[ 3]]
#  ...
#  [[24]]]
# normalize 最后接一个分类的任务
X = X / float(len(alphabet))
print(X.shape)
# (25, 3, 1)
# one hot编码输出label
y = np_utils.to_categorical(dataY)
print(y.shape)

# 创建&训练&保存模型
model = Sequential()
# input_shape = (time_step, 每个时间步的input_dim)
model.add(LSTM(5, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, nb_epoch=100, batch_size=1, verbose=2)
model.save("simplelstm.h5")
