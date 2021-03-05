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
background_ls = TreeHandler()
background_ls.get_handler_from_large_file(file_name=os.path.abspath(os.getcwd()) + '/data/OLD_DataTable_pp_LS.root',tree_name= "DataTable")

data = TreeHandler()
data.get_handler_from_large_file(file_name=os.path.abspath(os.getcwd()) + '/data/OLD_DataTable_pp.root',tree_name= "DataTable")

mc_signal = TreeHandler()
mc_signal.get_handler_from_large_file(file_name=os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_mtexp.root',tree_name= "SignalTable",preselection='rej_accept > 0 and pt>0')

training_variables = ["pt", "cos_pa" , "tpc_ncls_de" , "tpc_ncls_pr" , "tpc_ncls_pi", "tpc_nsig_de", "tpc_nsig_pr", "tpc_nsig_pi", "dca_de_pr", "dca_de_pi", "dca_pr_pi", "dca_de_sv", "dca_pr_sv", "dca_pi_sv", "chi2"]

min_eff = 0.5
max_eff = 0.9
step = 0.01

eff_array = np.arange(min_eff, max_eff, step)

# %%
data.get_var_names()
#data.get_data_frame().describe()
#%%
model_hdl = ModelHandler()
model_hdl.load_model_handler('./model/model_hdl')

#%%

train_test_data, y_pred_test, data, model_hdl = utils.train_xgboost_model(mc_signal, background_ls, data, training_variables)
#%%
model_hdl.dump_model_handler('./model/model_hdl')
# %%

utils.mass_spectrum_efficiency(train_test_data, y_pred_test, data, min_eff=min_eff, max_eff=max_eff,step=step)

# %%

scores = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(min_eff,max_eff,step))

for score, i in zip(scores, eff_array):
    sel = data.get_data_frame().query('model_output > ' + str(score))
    utils.scatter_with_hist(sel['m'],sel['mppi_vert'],[34,2.96,3.04],[34,1.08,1.13],
                            x_label='Hypertriton mass [GeV/$c^2$]',
                            y_label='$p - \pi$ mass [GeV/$c^2$]', eff = i,name = '/m_mppi/dalitz_eff_')
    


#%%


#with sqrt: [34,2.01,2.09]
#without sqrt: [34,4.04,4.37]

sel_m = data.get_data_frame().query('m > 2.989 & m < 2.993')

for score,i in zip(scores,eff_array):
    sel = sel_m.query('model_output > ' + str(score))
    utils.scatter_with_hist(sel['mppi'], sel['mdpi'],
                                [50,1.16,1.26],[50,4.07,4.22], name = '/mppi_mdpi/dalitz_eff_',
                                x_label='$p - \pi$ mass [GeV/$c^2$]',
                                y_label='$d - \pi$ mass [GeV/$c^2$]', eff=i)

del sel_m
#%%

background_ls.apply_model_handler(model_hdl)

#background_ls.get_data_frame().head()

#utils.mass_spectrum_efficiency(train_test_data, y_pred_test, background_ls)

# %%


mass_fit.data_ls_comp_plots(data,background_ls,
                            score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(min_eff,max_eff,step)),
                            np.arange(min_eff,max_eff,step))




# %%
