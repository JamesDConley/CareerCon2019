import pandas as pd
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, Masking
import ezPickle as p
from keras.optimizers import RMSprop
from keras.activations import softmax
from keras.utils import multi_gpu_model
import keras
def padSequence(sequence, max_len,num_inputs):
        while len(sequence) < max_len:
                sequence.append([0]*num_inputs)
        return sequence
max_len = p.load('max_len')
batch_size = 1000
input_shape = (max_len, 3)


model = Sequential()
model.add(Masking(mask_value=0, input_shape=input_shape))
model.add(LSTM(128,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(p.load('count'),  activation='softmax'))


cbs = [keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=120, restore_best_weights=True),
             keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)]
#EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.001, patience=5, verbose=1, mode='max', baseline=0.7, restore_best_weights=True)


rms = RMSprop()
model = multi_gpu_model(model, gpus = 3)
model.compile(loss='categorical_crossentropy',optimizer=rms, metrics=['categorical_accuracy'])
series_list = p.load('series_list')
output_list = p.load('output_list')
input_data = [padSequence(item,max_len,10) for item in series_list]
#print(input_data[0:5])
input_data = np.array(input_data).reshape(3810,128,3)
model.fit(np.array(input_data), np.array(output_list), epochs=200, batch_size=batch_size, verbose = 1, validation_split=0.1, callbacks=cbs)
import ezPickle as p
model = p.load('trained_model')

model.load_weights('best_model.h5')

p.save(model,'trained_model')
