import pandas as pd
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, Masking
import ezPickle as p
from keras.optimizers import RMSprop
from keras.activations import softmax
def padSequence(sequence, max_len,num_inputs):
        while len(sequence) < max_len:
                sequence.append([0]*num_inputs)
        return sequence
max_len = p.load('max_len')
batch_size = 100
input_shape = (max_len, 10)


model = Sequential()
model.add(Masking(mask_value=0, input_shape=input_shape))
model.add(LSTM(64,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(p.load('count'),  activation='softmax'))


rms = RMSprop()
model.compile(loss='categorical_crossentropy',optimizer=rms, metrics=['categorical_accuracy'])
series_list = p.load('series_list')
output_list = p.load('output_list')
input_data = [padSequence(item,max_len,10) for item in series_list]
#print(input_data[0:5])
input_data = np.array(input_data).reshape(3810,128,10)
model.fit(np.array(input_data), np.array(output_list), epochs=10, batch_size=batch_size, verbose = 1)
