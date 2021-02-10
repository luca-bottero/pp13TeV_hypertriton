#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import xgboost as xgb
import mass_fit
import ROOT
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
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

training_variables = ["pt", "cos_pa" , "tpc_ncls_de" , "tpc_ncls_pr" , "tpc_ncls_pi", "tpc_nsig_de", "tpc_nsig_pr", "tpc_nsig_pi", "dca_de_pr", "dca_de_pi", "dca_pr_pi", "dca_de_sv", "dca_pr_sv", "dca_pi_sv", "chi2"]


#mc_signal = mc_signal_raw.get_subset('rej_accept > 0')
#mc_signal_bal = mc_signal_raw.get_subset('rej_accept > 0', size=103200)      #balanced
#%%
# DATA EXPLORATION
print(list(data.get_data_frame().columns))

print(len(data.get_data_frame()))               #104732
print(len(backgorund_ls.get_data_frame()))      #103200
print(len(mc_signal.get_data_frame()))          #2272026

# %%
# TRAINING WITH UNBALANCED DATASET

train_test_data = train_test_generator([mc_signal, backgorund_ls], [1,0], test_size=0.5)

model_clf = xgb.XGBClassifier()
model_hdl = ModelHandler(model_clf, training_variables)
model_hdl.train_test_model(train_test_data)     

#%%

y_pred_train = model_hdl.predict(train_test_data[0], True)
y_pred_test = model_hdl.predict(train_test_data[2], True)

plt.rcParams["figure.figsize"] = (10, 7)
leg_labels = ['background', 'signal']

ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, 
                                               True, leg_labels, True, density=False)
plt.savefig('./images/output_train_test.png',dpi=300,facecolor='white')

roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,
                                                    train_test_data[1], y_pred_train, None, leg_labels)

#%%

data.apply_model_handler(model_hdl)
selected_data_hndl = data.get_subset('model_output > 0.95 & m>=2.96 & m<=3.04')

labels_list = ["after selection","before selection"]
colors_list = ['orangered', 'cornflowerblue']
plot_utils.plot_distr([selected_data_hndl, data], column='pt', bins=34, labels=labels_list, colors=colors_list,log=True, density=False,fill=True,  alpha=0.5)    #histtype='step',
ax = plt.gca()
ax.set_xlabel(r'$p_T$ (GeV/$c$)')
ax.margins(x=0)
ax.xaxis.set_label_coords(0.9, -0.075)

labels_list = ["after selection","before selection"]
colors_list = ['orangered', 'cornflowerblue']
plot_utils.plot_distr([selected_data_hndl, data], column='centrality',log=True, bins=34, labels=labels_list, colors=colors_list, density=False,fill=True,  alpha=0.5)
ax = plt.gca()
ax.set_xlabel(r'Mass (GeV/$c^2$)')
ax.margins(x=0)
ax.xaxis.set_label_coords(0.9, -0.075)

print(len(selected_data_hndl))
print(len(selected_data_hndl)/len(data))

# %%
max_score = max(data.get_data_frame()['model_output'])

for score in np.arange(0., max_score, 0.1):
    selected_data_hndl = data.get_subset('model_output > ' + str(score))
    plot_utils.plot_distr(selected_data_hndl,labels='score > ' + str(np.round(score,2)), column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
    ax = plt.gca()
    ax.set_xlabel(r'Mass (GeV/$c^2$)')
    ax.margins(x=0)
    ax.xaxis.set_label_coords(0.9, -0.075)
    plt.savefig('./images/model_out_GT_' + str(np.round(score,2)) + '.png',dpi = 300, facecolor = 'white')

# %%
# BDT EFFICIENCY AS FUNCTION OF BDT OUTPUT AND INVERSE
bdt_efficiency = bdt_efficiency_array(train_test_data[3],y_pred_test)

plt.plot(bdt_efficiency[1],bdt_efficiency[0])
plt.title("BDT efficiency as a function of BDT output")
plt.xlabel('BDT output')
plt.ylabel('Efficiency')
plt.savefig('./images/bdt_eff_bdt_out.png',dpi=300,facecolor='white')
plt.show()

score_from_eff = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(0,1,0.001))
plt.plot(np.arange(0,1,0.001),score_from_eff)
plt.title('BDT score as a function of BDT efficiency')
plt.xlabel('Efficiency')
plt.ylabel('BDT output')
plt.savefig('./images/bdt_out_dbt_eff.png',dpi=300,facecolor='white')
plt.show()

#%%
#Calculate the previous values on range 65-90 %
import imp
imp.reload(mass_fit)

min_eff = 0.65
max_eff = 0.9

scores = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(min_eff,max_eff,0.01))

mass_fit.systematic_estimate(data,scores,np.arange(.65,.9,.01))


#%%
# BDT FEATURE IMPORTANCE USING HIPE4ML WITH SHAP

feat_imp_1, feat_imp_2 = plot_utils.plot_feature_imp(train_test_data[2],train_test_data[3],model_hdl,approximate=False)
feat_imp_1.savefig('./images/feature_importance_HIPE4ML_violin.png',dpi=300,facecolor='white')
feat_imp_2.savefig('./images/feature_importance_HIPE4ML_bar.png',dpi=300,facecolor='white')

# %%
# BDT FEATURE IMPORTANCE USING XGBOOST METHOD
xgb_model = model_hdl.get_original_model()
sorted_idx = xgb_model.feature_importances_.argsort()
plt.barh([training_variables[i] for i in sorted_idx], xgb_model.feature_importances_[sorted_idx])
plt.xlabel("Xgboost Feature Importance")
plt.savefig('./images/xgb_feature_importance.png', dpi=300, facecolor='white')

# %%
# %%
