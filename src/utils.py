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

def train_xgboost_model(signal, background, training_variables='', testsize = 0.5):
    '''
    Trains an XGBOOST model using hipe4ml and plot output distribution and feature importance
    '''

    train_test_data = train_test_generator([signal, background], [1,0], test_size=testsize)

    if training_variables == '':
        training_variables = train_test_data[0].columns.tolist()

    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, training_variables)
    model_hdl.train_test_model(train_test_data)     

    y_pred_train = model_hdl.predict(train_test_data[0], True)
    y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance

    plt.rcParams["figure.figsize"] = (10, 7)
    leg_labels = ['background', 'signal']

    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, 
                                                True, leg_labels, True, density=False)  
    plt.savefig('../training/images/output_train_test.png',dpi=300,facecolor='white')
    plt.show()

    roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,
                                                        train_test_data[1], y_pred_train, None, leg_labels) #ROC AUC plot
    plt.savefig('../training/images/ROC_AUC_train_test.png',dpi=300,facecolor='white')
    plt.show()

    efficiency_score_conversion(train_test_data, y_pred_test)

    feat_imp_1, feat_imp_2 = plot_utils.plot_feature_imp(train_test_data[2],train_test_data[3],model_hdl,approximate=False)
    feat_imp_1.savefig('../training/images/feature_importance_HIPE4ML_violin.png',dpi=300,facecolor='white')
    feat_imp_2.savefig('../training/images/feature_importance_HIPE4ML_bar.png',dpi=300,facecolor='white')

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
    bdt_efficiency = bdt_efficiency_array(train_test_data[3],y_pred_test)

    plt.plot(bdt_efficiency[1],bdt_efficiency[0])
    plt.title("BDT efficiency as a function of BDT output")
    plt.xlabel('BDT output')
    plt.ylabel('Efficiency')
    plt.savefig('../training/images/bdt_eff_bdt_out.png',dpi=300,facecolor='white')
    plt.show()

    score_from_eff = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(1E-6,1-1E-6,0.001))
    plt.plot(np.arange(1E-6,1-1E-6,0.001),score_from_eff)
    #plt.plot(np.arange(1E-6,1-1E-6,0.01),score_from_eff,marker='o',color='red',linestyle='none')

    plt.title('BDT score as a function of BDT efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('BDT output')
    plt.savefig('../training/images/bdt_out_dbt_eff.png',dpi=300,facecolor='white')
    plt.show()

def mass_spectrum_efficiency(data, scores, eff_array):
    '''
    Plots the mass spectrum histograms at different efficiencies with a fitting curve.
    Plots the count in the relevant mass range at different efficiencies.
    '''

    mass_fit.systematic_estimate(data,scores,eff_array)

def scatter_with_hist(x_data,y_data,x_axis,y_axis,x_label='',y_label='',eff = 0.,name=''):
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

    plt.savefig('../analysis/images/' + name + str(np.round(eff,4)) + '.png', dpi=300, facecolor='white')
    plt.show()


