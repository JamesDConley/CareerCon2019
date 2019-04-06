import ezPickle as p
import pandas as pd
out = p.load('out')
out.columns = ['{}_{}'.format(i, j) for i, j in out.columns]
out.to_csv('summary_stats.csv')
