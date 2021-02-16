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
#%%

def get_large_data(hndl,data_path, table, query = ''):
    '''
    Loads large ROOT trees into a TreeHandler using low RAM on multiple processes
    '''

    executor = ThreadPoolExecutor()
    iterator = uproot.pandas.iterate(data_path, table, executor=executor, reportfile=True)

    selected_data = pd.DataFrame()

    for current_file, data in iterator:
        #print('current file: {}'.format(current_file))
        #print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))

        if query == '':
            selected_data = selected_data.append(data)
        else:
            selected_data = selected_data.append(data.query(query))
        
        del data
        del current_file
        
    
    hndl._full_data_frame = selected_data

    return hndl

def train_xgboost_model(signal, background, data, training_variables='', testsize = 0.5):
    '''
    Trains an XGBOOST model using hipe4ml and plots some results.
    Applies it to the datas
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
    plt.savefig('./images/model/output_train_test.png',dpi=300,facecolor='white')
    plt.show()

    roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,
                                                        train_test_data[1], y_pred_train, None, leg_labels) #ROC AUC plot
    plt.savefig('./images/model/ROC_AUC_train_test.png',dpi=300,facecolor='white')
    plt.show()

    efficiency_score_conversion(train_test_data, y_pred_test)

    data.apply_model_handler(model_hdl)

    return train_test_data, y_pred_test, data

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
    plt.savefig('./images/model/bdt_eff_bdt_out.png',dpi=300,facecolor='white')
    plt.show()

    score_from_eff = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(0.0000001,0.9999999,0.001))
    plt.plot(np.arange(0.0000001,0.9999999,0.001),score_from_eff)
    plt.plot(np.arange(0.0000001,0.9999999,0.001),score_from_eff,marker='o',color='red',linestyle='none')

    plt.title('BDT score as a function of BDT efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('BDT output')
    plt.savefig('./images/model/bdt_out_dbt_eff.png',dpi=300,facecolor='white')
    plt.show()

def mass_spectrum_efficiency(train_test_data,y_pred_test,data,min_eff=0.65,max_eff=0.9,step=0.01):
    '''
    Plots the mass spectrum histograms at different efficiencies with a fitting curve.
    Plots the count in the relevant mass range at different efficiencies.
    '''

    scores = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(min_eff,max_eff,step))

    mass_fit.systematic_estimate(data,scores,np.arange(min_eff,max_eff,step))

def bdt_feature_importance(train_test_data, model_handler):
    feat_imp_1, feat_imp_2 = plot_utils.plot_feature_imp(train_test_data[2],train_test_data[3],model_hdl,approximate=False)
    feat_imp_1.savefig('./images/model/feature_importance_HIPE4ML_violin.png',dpi=300,facecolor='white')
    feat_imp_2.savefig('./images/model/feature_importance_HIPE4ML_bar.png',dpi=300,facecolor='white')


# %%

def scatter_with_hist(x_data,y_data,x_axis,y_axis,x_label='',y_label='',eff = 0.):
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
    plot.plot2d_full(
        main_cmap="cividis",
        top_color="steelblue",
        top_lw=2,
        side_lw=2,
        side_color="steelblue"
        )
    plt.savefig('./images/m_mppi/scatter_eff_' + str(np.round(eff,4)) + '.png', dpi=300, facecolor='white')
    plt.show()


# %%
