#Generating Summary Statistics For Random Forests
import ezPickle as p
import pandas as pd

train_x = pd.read_csv('X_train.csv')
test_x = pd.read_csv('X_test.csv')

train_x = train_x.drop(columns=['measurement_number'])
test_x = test_x.drop(columns=['measurement_number'])

for index in ['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']:
	train_x[index+'_jerk'] = train_x.groupby(['series_id'])[index].transform(pd.Series.diff)
	test_x[index+'_jerk'] = test_x.groupby(['series_id'])[index].transform(pd.Series.diff)
grouped_train_x = train_x.groupby(['series_id'])
grouped_test_x = test_x.groupby(['series_id'])
out = grouped_train_x.describe()
t_out = grouped_test_x.describe()

out.columns = ['{}_{}'.format(i, j) for i, j in out.columns]
t_out.columns = ['{}_{}'.format(i, j) for i, j in t_out.columns]

out.to_csv('summary_stats.csv')
t_out.to_csv('test_summary_stats.csv')


