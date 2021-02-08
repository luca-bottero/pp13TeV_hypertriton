#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uproot
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator, bdt_efficiency_array
from hipe4ml import plot_utils
#%%
print('Loading Monte-Carlo data...')
mc_signal_raw = TreeHandler(os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_tofPID_mtexp.root', "SignalTable")
print('Done \nFiltering MC data...')
mc_signal = mc_signal_raw.get_subset('rej_accept > 0 and pt>0')
del mc_signal_raw       #free the memory. This and the comment are useful because of intense memory usage 
print('Done \nLoading background...')
backgorund_ls = TreeHandler(os.path.abspath(os.getcwd()) + '/data/DataTable_pp_LS.root', "DataTable")
print("Done \nLoading datas...")
data = TreeHandler(os.path.abspath(os.getcwd()) + '/data/DataTable_pp.root', "DataTable")
print('Done')

#mc_signal = mc_signal_raw.get_subset('rej_accept > 0')
#mc_signal_bal = mc_signal_raw.get_subset('rej_accept > 0', size=103200)      #balanced
#%%
# DATA EXPLORATION
training_variables = ["pt", "cos_pa" , "tpc_ncls_de" , "tpc_ncls_pr" , "tpc_ncls_pi", "tpc_nsig_de", "tpc_nsig_pr", "tpc_nsig_pi", "dca_de_pr", "dca_de_pi", "dca_pr_pi", "dca_de_sv", "dca_pr_sv", "dca_pi_sv", "chi2"]
print(list(data.get_data_frame().columns))

print(len(data.get_data_frame()))               #104732
print(len(backgorund_ls.get_data_frame()))      #103200
print(len(mc_signal.get_data_frame()))          #2272026

#The dataset would be highly unbalanced, thus I think it would be better to sample the MC datas to make a 
#balanced dataset. I will try both options

# %%
# TRAINING WITH UNBALANCED DATASET

train_test_data = train_test_generator([mc_signal, backgorund_ls], [1,0], test_size=0.5)
  
#%%

model_clf = xgb.XGBClassifier()
model_hdl = ModelHandler(model_clf, training_variables)
model_hdl.train_test_model(train_test_data)     

#%%

y_pred_train = model_hdl.predict(train_test_data[0], True)
y_pred_test = model_hdl.predict(train_test_data[2], True)

plt.rcParams["figure.figsize"] = (10, 7)
leg_labels = ['background', 'signal']

ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, 
                                               False, leg_labels, True, density=True)

roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,
                                                    train_test_data[1], y_pred_train, None, leg_labels)

#%%
plt.hist(data.get_data_frame()['model_output'],bins=100)


#%%

data.apply_model_handler(model_hdl)
selected_data_hndl = data.get_subset('model_output > 0.95 & m>=2.96 & m<=3.04')

labels_list = ["after selection","before selection"]
colors_list = ['orangered', 'cornflowerblue']
plot_utils.plot_distr([selected_data_hndl, data], column='pt', bins=34, labels=labels_list, colors=colors_list, density=True,fill=True,  alpha=0.5)    #histtype='step',
ax = plt.gca()
ax.set_xlabel(r'$p_T$ (GeV/$c$)')
ax.margins(x=0)
ax.xaxis.set_label_coords(0.9, -0.075)

labels_list = ["after selection"]
colors_list = ['orangered']
plot_utils.plot_distr([selected_data_hndl, data], column='m', bins=34, labels=labels_list, colors=colors_list, density=True,fill=True,  alpha=0.5)
ax = plt.gca()
ax.set_xlabel(r'Mass (GeV/$c^2$)')
ax.margins(x=0)
ax.xaxis.set_label_coords(0.9, -0.075)

labels_list = ["after selection","before selection"]
colors_list = ['orangered', 'cornflowerblue']
plot_utils.plot_distr([selected_data_hndl, data], column='centrality',log=True, bins=34, labels=labels_list, colors=colors_list, density=True,fill=True,  alpha=0.5)
ax = plt.gca()
ax.set_xlabel(r'Mass (GeV/$c^2$)')
ax.margins(x=0)
ax.xaxis.set_label_coords(0.9, -0.075)

print(len(selected_data_hndl))
print(len(selected_data_hndl)/len(data))

# %%
max_score = max(data.get_data_frame()['model_output'])

for score in np.arange(0., max_score, 0.1):
    selected_data_hndl = data.get_subset('m >= 2.96 & m<= 3.04 & model_output > ' + str(score))
    plot_utils.plot_distr(selected_data_hndl,labels='score > ' + str(np.round(score,2)), column='m', bins=34, colors='orange', density=True,fill=True)
    ax = plt.gca()
    ax.set_xlabel(r'Mass (GeV/$c^2$)')
    ax.margins(x=0)
    ax.xaxis.set_label_coords(0.9, -0.075)
    plt.savefig('./images/model_out_GT_' + str(score) + '.png',dpi = 300, facecolor = 'white')



# %%
backgorund_ls.apply_model_handler(model_hdl)
mc_signal.apply_model_handler(model_hdl)
mc_bkg_merge = train_test_generator([mc_signal, backgorund_ls], [1,0], test_size=0.99)
bdt_efficiency = bdt_efficiency_array(mc_bkg_merge[3],mc_bkg_merge[2]['model_output'])

plt.plot(bdt_efficiency[1],bdt_efficiency[0])
plt.title("BDT efficiency as a function of BDT output")
plt.xlabel('BDT output')
plt.ylabel('Efficiency')
plt.savefig('./images/bdt_eff_bdt_out.png',dpi=300,facecolor='white')



# %%

# %%
