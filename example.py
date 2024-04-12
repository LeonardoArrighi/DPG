import cProfile
import pstats
from pstats import SortKey
import fhg.test_sklearn as test


datasets = 'iris'
n_learners = 2 # base leaners for the RF
perc_var = 0 # perc_var to 0.1, it means you want to keep only the paths that occur in at least 10% of the trees
decimal_threshold = 2 # decimal precision of each feature, a small number means aggregate more nodes
plot = True # to plot the FHG


df, df_perc = test.test_base_sklearn(datasets, n_learners, perc_var, decimal_threshold, plot)


#df = cProfile.runctx('test.test_base_sklearn(datasets, n_learners, perc_var, decimal_threshold)', {"datasets": datasets, "n_learners": n_learners, "perc_var": perc_var, "decimal_threshold": decimal_threshold}, locals=locals(), filename="train.prof", sort="cumtime")
#df.sort_values(['Degree'])
#print(df)

#p = pstats.Stats('train.prof')
#p.strip_dirs().sort_stats('cumulative').print_stats(50)
