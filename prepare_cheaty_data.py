#Preprocess data for training
import ezPickle as p
import pandas as pd
from sklearn.preprocessing import RobustScaler
#batchSize, sequenceLength, numInputs
train_x = pd.read_csv('X_train.csv')
train_y = pd.read_csv('y_train.csv')
max_len = 0
current_len = 0
tempSeriesId = 0
series_list = []
temp_list = []
num_sequences = 0
all_seq = []
for id, data in train_x.iterrows():
        if data[1] != tempSeriesId:
                if max_len < current_len:
                        max_len = current_len
                tempSeriesId = data[1]
                series_list.append(temp_list.copy())
                temp_list = []
                current_len = 0
                num_sequences+=1
                
        current_len+=1
        temp_list.append(data[3:6].tolist())
        all_seq.append(data[3:6].tolist())
series_list.append(temp_list)


transformer = RobustScaler().fit(all_seq)
transformed_list = [transformer.transform(item) for item in series_list]
encoder_dict = {}
count = 0
output_list = []
for id, data in train_y.iterrows():
        if not data[2] in encoder_dict.keys():
                encoder_dict[data[2]] = count
                count+=1
        output_list.append(data[2])
for key in encoder_dict.keys():
        temp = [0]*count
        temp[encoder_dict[key]] = 1
        encoder_dict[key] = temp
output_list = [encoder_dict[item] for item in output_list]
p.save(count,'count')
p.save(encoder_dict,'encoder_dict')
p.save(max_len,'max_len')
p.save(num_sequences,'num_sequences')
p.save(transformed_list,'series_list')
p.save(output_list,'output_list')
p.save(transformer,'transformer')
print(max_len)
print(num_sequences)
print(count)
#print(train_x)
#print(train_y)
