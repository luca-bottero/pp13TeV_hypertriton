#%%
import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import xgboost as xgb
import mass_fit
import ROOT
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
from hipe4ml import plot_utils
#%%

print('load dummy tree')

a = TreeHandler(os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_mtexp.root',"SignalTable")
backgorund_ls = utils.get_large_data(a, os.path.abspath(os.getcwd()) + '/data/OLD_DataTable_pp_LS.root', "DataTable")
del a

a = TreeHandler(os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_mtexp.root',"SignalTable")
data = utils.get_large_data(a, os.path.abspath(os.getcwd()) + '/data/OLD_DataTable_pp.root', "DataTable")
del a

a = TreeHandler(os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_mtexp.root',"SignalTable")
mc_signal = utils.get_large_data(a, os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_mtexp.root',"SignalTable", 'rej_accept > 0 and pt>0')
del a


training_variables = ["pt", "cos_pa" , "tpc_ncls_de" , "tpc_ncls_pr" , "tpc_ncls_pi", "tpc_nsig_de", "tpc_nsig_pr", "tpc_nsig_pi", "dca_de_pr", "dca_de_pi", "dca_pr_pi", "dca_de_sv", "dca_pr_sv", "dca_pi_sv", "chi2"]

#%%
import imp
imp.reload(utils)

# %%

train_test_data, y_pred_test, data = utils.train_xgboost_model(mc_signal, backgorund_ls, data, training_variables)

# %%

utils.mass_spectrum_efficiency(train_test_data, y_pred_test, data)

# %%
