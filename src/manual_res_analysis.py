import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = '../analysis_results/TOF_PID_cut/output_data/'

bckg = pd.read_parquet(path + 'bckg_ls_scores.parquet.gzip')
data = pd.read_parquet(path + 'data_scores.parquet.gzip')

data.hist()
plt.show()