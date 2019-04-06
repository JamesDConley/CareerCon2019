from sklearn import svm
import ezPickle as p
import pandas as pd


series_stats = pd.read_csv('summary_stats.csv')
outputs = p.load('output_list')
for i in range(len(outputs)):
	outputs[i] = outputs[i].index(1)
clf = svm.LinearSVC(max_iter=100000)
clf.fit(series_stats.values[0:3000], outputs[0:3000])
p.save(clf,'clf')
predictions = clf.predict(series_stats.values[3000:])
print(clf.score(series_stats.values[3000:], outputs[3000:]))
new = []
for item in predictions:
	temp = [0]*9
	temp[item] = 1
	new.append(temp.copy())
predictions = new
correct = 0
val_data = outputs[3000:]
for i in range(len(val_data)): 
	#print(predictions[i])
	if str(predictions[i]) == str(val_data[i]):
		correct+=1
print(correct/len(val_data))
		


#print(clf.score(series_stats.values[3000:], outputs[3000:]))
