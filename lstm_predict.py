#Preprocess data for training
import ezPickle as p
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
def getReverseOneHotDict(dict):
    inverted_dict = {str(value): key for key, value in dict.items()}
    return inverted_dict
encoder_dict = p.load('encoder_dict')
decoder_dict = getReverseOneHotDict(encoder_dict)
series_list = p.load('test_list')
num_sequences = len(series_list)

model = p.load('trained_model')
output = model.predict(np.array(series_list))
output = output.tolist()
for i in range(len(output)):
    temp = [0] * len(output[i])
    temp[output[i].index(max(output[i]))] = 1  
    output[i] = temp
output_data = pd.DataFrame({'series_id': range(num_sequences),'surface':[decoder_dict[str(item)] for item in output]})
output_data.to_csv('LSTM_Output.csv',index = False)




