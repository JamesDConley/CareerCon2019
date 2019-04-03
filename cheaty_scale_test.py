#Preprocess data for training
import ezPickle as p
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def padSequence(sequence, max_len,num_inputs):
        while len(sequence) < max_len:
                sequence.append([0]*num_inputs)
        return sequence

test_x = pd.read_csv('X_test.csv')
transformer = p.load('transformer')
current_len = 0
tempSeriesId = 0
series_list = []
temp_list = []

all_seq = []

for id, data in test_x.iterrows():
        if data[1] != tempSeriesId:
                if current_len > 128:
                        print("Uh Oh")
                tempSeriesId = data[1]
                series_list.append(transformer.transform(padSequence(temp_list.copy(),128,10)))
                temp_list = []
                current_len = 0
        current_len+=1
        temp_list.append(data[3:6].tolist())
series_list.append(transformer.transform(padSequence(temp_list.copy(),128,10)))
p.save(series_list,'test_list')
