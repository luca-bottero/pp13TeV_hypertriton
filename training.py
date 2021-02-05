#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uproot
import os
from hipe4ml.tree_handler import TreeHandler
#%%
data = TreeHandler(os.path.abspath(os.getcwd()) + '/data/DataTable_pp.root', "DataTable").get_data_frame()
#%%
training_variables = ["pt", "cos_pa" , "tpc_ncls_de" , "tpc_ncls_pr" , "tpc_ncls_pi", "tpc_nsig_de", "tpc_nsig_pr", "tpc_nsig_pi", "dca_de_pr", "dca_de_pi", "dca_pr_pi", "dca_de_sv", "dca_pr_sv", "dca_pi_sv", "chi2"]
print(list(data.columns))
data.head(10)
data.describe()
# %%
data.hist(figsize=(15,15),bins=100)
# %%
