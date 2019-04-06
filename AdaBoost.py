import ezPickle as p
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
series_stats = pd.read_csv('summary_stats.csv')
outputs = p.load('output_list')
for i in range(len(outputs)):
	outputs[i] = outputs[i].index(1)
clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=500, max_depth=7,random_state=0), random_state=0)
clf.fit(series_stats.values[0:3000], outputs[0:3000])
p.save(clf,'clf')
print(clf.score(series_stats.values[3000:], outputs[3000:]))
