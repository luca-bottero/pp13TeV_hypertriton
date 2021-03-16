#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import xgboost as xgb
import mass_fit
import ROOT
import hist
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
from hipe4ml import plot_utils
import matplotlib

matplotlib.use('pdf')

def train_xgboost_model(signal, background, training_variables='', testsize = 0.5, optimize_bayes = False):
    '''
    Trains an XGBOOST model using hipe4ml and plot output distribution and feature importance
    '''
    
    print('Training XGBOOST model')

    params = {'n_jobs' : 8,  
                'seed': 42,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'max_depth': 13,
                'learning_rate': 0.09823,
                'n_estimators': 181,
                'gamma': 0.4467,
                'min_child_weight': 5.751,
                'subsample': 0.7447,
                'colsample_bytree': 0.5727,
                }
    
    params_range = {
    "max_depth": (8, 18),
    "learning_rate": (0.07,0.15),
    "n_estimators": (150, 250),
    "gamma": (0.3,0.5),
    "min_child_weight": (3,8),
    "subsample": (0.5,1),
    "colsample_bytree": (0.3,1),
    }

    train_test_data = train_test_generator([signal, background], [1,0], test_size=testsize)

    if training_variables == '':
        training_variables = train_test_data[0].columns.tolist()

    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, training_variables)
    model_hdl.set_model_params(params)

    if optimize_bayes:
        print('Doing Bayes optimization of hyperparameters\n')
        model_hdl.optimize_params_bayes(train_test_data, params_range,'roc_auc',njobs=-1)
    
    model_hdl.train_test_model(train_test_data, )     

    y_pred_train = model_hdl.predict(train_test_data[0], True)
    y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance

    plt.rcParams["figure.figsize"] = (10, 7)
    leg_labels = ['background', 'signal']

    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, 
                                                True, leg_labels, True, density=False)

    training_fig_path = "../images/training"
    if not os.path.exists(training_fig_path):
        os.makedirs(training_fig_path)

    plt.savefig(training_fig_path + '/output_train_test.png',dpi=300,facecolor='white')
    plt.show()
    plt.close()

    roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,
                                                        train_test_data[1], y_pred_train, None, leg_labels) #ROC AUC plot
    plt.savefig(training_fig_path + '/ROC_AUC_train_test.png',dpi=300,facecolor='white')
    plt.show()
    plt.close()

    efficiency_score_conversion(train_test_data, y_pred_test)

    feat_imp_1, feat_imp_2 = plot_utils.plot_feature_imp(train_test_data[2],train_test_data[3],model_hdl,approximate=False)
    feat_imp_1.savefig(training_fig_path + '/feature_importance_HIPE4ML_violin.png',dpi=300,facecolor='white')
    feat_imp_2.savefig(training_fig_path + '/feature_importance_HIPE4ML_bar.png',dpi=300,facecolor='white')
    plt.show()
    plt.close()

    return train_test_data, y_pred_test, model_hdl  

def data_exploration(data):
    '''
    Plots some interesting feature of the dataset
    '''

    pass

def efficiency_score_conversion(train_test_data, y_pred_test):
    '''
    Plots the efficiency as a function of the score and its inverse
    '''

    training_fig_path = "../images/training"
    if not os.path.exists(training_fig_path):
        os.makedirs(training_fig_path)

    bdt_efficiency = bdt_efficiency_array(train_test_data[3],y_pred_test)

    plt.plot(bdt_efficiency[1],bdt_efficiency[0])
    plt.title("BDT efficiency as a function of BDT output")
    plt.xlabel('BDT output')
    plt.ylabel('Efficiency')
    plt.savefig(training_fig_path + '/bdt_eff_bdt_out.png',dpi=300,facecolor='white')
    plt.show()
    plt.close()    

    score_from_eff = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(1E-6,1-1E-6,0.001))
    plt.plot(np.arange(1E-6,1-1E-6,0.001),score_from_eff)
    #plt.plot(np.arange(1E-6,1-1E-6,0.01),score_from_eff,marker='o',color='red',linestyle='none')
    plt.title('BDT score as a function of BDT efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('BDT output')
    plt.savefig(training_fig_path + '/bdt_out_dbt_eff.png',dpi=300,facecolor='white')
    plt.show()
    plt.close()
    

def mass_spectrum_efficiency(data, scores, eff_array):
    '''
    Plots the mass spectrum histograms at different efficiencies with a fitting curve.
    Plots the count in the relevant mass range at different efficiencies.
    '''

    mass_fit.systematic_estimate(data,scores,eff_array)

def scatter_with_hist(x_data,y_data,x_axis,y_axis,x_label='',y_label='',eff = 0.,path = '',name=''):
    '''
    Plots a scatterplot with histograms of the distributions
    '''
    
    plot = (
        hist.Hist.new
        .Reg(x_axis[0],x_axis[1],x_axis[2], name='x', label=x_label)
        .Reg(y_axis[0],y_axis[1],y_axis[2], name='y', label=y_label)
        .Double()
        )

    plot.fill(x=x_data,y=y_data)

    ax = plot.plot2d_full(
        main_cmap="cividis",
        top_color="steelblue",
        top_lw=2,
        side_lw=2,
        side_color="steelblue"
        )

    if not os.path.exists('../results/images/'):
        os.makedirs('../results/images/')

    if not os.path.exists('../results/images/' + path):
        os.makedirs('../results/images/' + path)

    
    plt.savefig('../results/images/' +path + name + str(np.round(eff,4)) + '.png', dpi=300, facecolor='white')
    plt.show()
    plt.close()
    


