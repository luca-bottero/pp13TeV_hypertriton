#%%
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import xgboost as xgb
import mass_fit
import hist
import optuna
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
from hipe4ml import plot_utils
import matplotlib

matplotlib.use('pdf')

def train_xgboost_model(signal, background, filename_dict, params, params_range, flag_dict, training_variables='', testsize = 0.5):
    '''
    Trains an XGBOOST model using hipe4ml and plot output distribution and feature importance
    '''
    
    print('Training XGBOOST model')

    training_fig_path = filename_dict['analysis_path'] + "/images/training"

    train_test_data = train_test_generator([signal, background], [1,0], test_size=testsize)

    if training_variables == '':
        training_variables = train_test_data[0].columns.tolist()

    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, training_variables)
    if not flag_dict['use_default_param']:
        model_hdl.set_model_params(params)

    if flag_dict['benchmark_opt']:

        print('Benchamarking optimizers\n')
        import time
        from sklearn.metrics import roc_auc_score
        times_sk = []
        roc_sk = []

        for i in range(1):
            start = time.time()

            model_hdl.optimize_params_bayes(train_test_data, params_range,'roc_auc',njobs=-1)
            model_hdl.train_test_model(train_test_data, )

            y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance

            roc_sk.append(roc_auc_score(train_test_data[3], y_pred_test))

            times_sk.append(time.time() - start)

        print('\nBAYES OPTIMIZATION WITH SKLEARN')
        print('Mean time : ' + str(np.mean(times_sk)))
        print('Mean ROC : ' + str(np.mean(roc_sk)))
        print('--------------\n')
        print('OPTUNA')

        time = []
        roc = []

        for i in range(1):
            
            for key in params:
                if isinstance(params[key], str):
                    params_range[key] = params[key]

            model_hdl.optimize_params_optuna(train_test_data, params_range,'roc_auc', timeout = flag_dict['timeout'] ,n_jobs=flag_dict['n_jobs'])
            model_hdl.train_test_model(train_test_data, )

            y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance

            roc.append(roc_auc_score(train_test_data[3], y_pred_test))

        print('\nBAYES OPTIMIZATION WITH SKLEARN')
        print('Mean time : ' + str(np.mean(times_sk)))
        print('Mean ROC : ' + str(np.mean(roc_sk)))
        print('--------------\n')
        print('OPTUNA')
        print('Fixed time : ' + str(np.mean(time)))
        print('Mean ROC : ' + str(np.mean(roc)))
        print('--------------\n')


    if flag_dict['optimize_bayes']:
        print('Doing Bayes optimization of hyperparameters\n')
        model_hdl.optimize_params_bayes(train_test_data, params_range,'roc_auc',n_jobs=-1)
        
    if flag_dict['optimize_optuna']:
        print('Doing Optuna optimization of hyperparameters\n')
        for key in params:
                if isinstance(params[key], str):
                    params_range[key] = params[key]
        study = model_hdl.optimize_params_optuna(train_test_data, params_range, scoring = 'roc_auc', 
                                                    timeout = 3600, n_jobs = 64, n_trials = None)

        print('Parameters optimization done!\n')

        if flag_dict['plot_optim']:
            print('Saving optimization plots')
            fig = optuna.visualization.plot_slice(study)
            fig.write_image(training_fig_path + '/optuna_slice.png')
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(training_fig_path + '/optuna_history.png')
            '''fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(training_fig_path + '/optuna_param_importance.png')
            fig = optuna.visualization.plot_contour(study)
            fig.write_image(training_fig_path + '/optuna_contour.png')'''
            print('Done\n')
        

    model_hdl.train_test_model(train_test_data, )
    print(model_hdl.get_model_params())     

    print('Predicting values on training and test datas')
    y_pred_train = model_hdl.predict(train_test_data[0], True)
    y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance
    print('Prediction done\n')

    plt.rcParams["figure.figsize"] = (10, 7)
    leg_labels = ['background', 'signal']

    print('Saving Output comparison plot')
    plt.figure()
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, 
                                                True, leg_labels, True, density=False)
    plt.savefig(training_fig_path + '/output_train_test.png',dpi=300,facecolor='white')    
    plt.close()
    print('Done\n')

    print('Saving ROC AUC plot')
    plt.figure()
    roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test,
                                                        train_test_data[1], y_pred_train, None, leg_labels) #ROC AUC plot
    plt.savefig(training_fig_path + '/ROC_AUC_train_test.png',dpi=300,facecolor='white')
    plt.close()
    print('Done\n')

    print('Saving feature importance plots')
    plt.figure()
    feat_imp_1, feat_imp_2 = plot_utils.plot_feature_imp(train_test_data[2],train_test_data[3],model_hdl,approximate=True)
    feat_imp_1.savefig(training_fig_path + '/feature_importance_HIPE4ML_violin.png',dpi=300,facecolor='white')
    feat_imp_2.savefig(training_fig_path + '/feature_importance_HIPE4ML_bar.png',dpi=300,facecolor='white')
    plt.close()
    print('Done\n')

    efficiency_score_conversion(train_test_data, y_pred_test, filename_dict)

    return train_test_data, y_pred_test, model_hdl  

def data_exploration(data):
    '''
    Plots some interesting feature of the dataset
    '''

    pass

def save_data_description(filename_dict, dataframe, append = True, name = ''):
    header = ['\n\ncentrality - ' + name]

    if append:
        dataframe['centrality'].describe().to_csv(filename_dict['analysis_path'] + filename_dict['analysis_name'], mode = 'a', header = header)
    else:
        dataframe['centrality'].describe().to_csv(filename_dict['analysis_path'] + filename_dict['analysis_name'], header = header)


def efficiency_score_conversion(train_test_data, y_pred_test, filename_dict):
    '''
    Plots the efficiency as a function of the score and its inverse
    '''

    training_fig_path = filename_dict['analysis_path'] + "/images/training"

    bdt_efficiency = bdt_efficiency_array(train_test_data[3],y_pred_test)

    plt.figure()
    plt.plot(bdt_efficiency[1],bdt_efficiency[0])
    plt.title("BDT efficiency as a function of BDT output")
    plt.xlabel('BDT output')
    plt.ylabel('Efficiency')
    plt.savefig(training_fig_path + '/bdt_eff_bdt_out.png',dpi=300,facecolor='white')
    
    plt.close()    

    plt.figure()
    score_from_eff = score_from_efficiency_array(train_test_data[3],y_pred_test,np.arange(1E-6,1-1E-6,0.001))
    plt.plot(np.arange(1E-6,1-1E-6,0.001),score_from_eff)
    #plt.plot(np.arange(1E-6,1-1E-6,0.01),score_from_eff,marker='o',color='red',linestyle='none')
    plt.title('BDT score as a function of BDT efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('BDT output')
    plt.savefig(training_fig_path + '/bdt_out_dbt_eff.png',dpi=300,facecolor='white')
    
    plt.close()
    

def mass_spectrum_efficiency(data, scores, eff_array):
    '''
    Plots the mass spectrum histograms at different efficiencies with a fitting curve.
    Plots the count in the relevant mass range at different efficiencies.
    '''

    mass_fit.systematic_estimate(data,scores,eff_array)

def scatter_with_hist(x_data,y_data,x_axis,y_axis,filename_dict,x_label='',y_label='',eff = 0.,path = '',name=''):
    """Plots a scatter heatmap with the distribution histogram on the axis. Usually used to make Dalitz Plot or similar

    Args:
        x_data (np.array): data on the x axis
        y_data (np.array): data on the y axis
        x_axis (list): defines the x axis dimensions. a list composed of: number of bin, min value, max value
        y_axis (list): defines the y axis dimensions. A list composed of: number of bin, min value, max value
        filename_dict (dictionary): dictionary of the filenames
        x_label (str, optional): label of the x axis. Defaults to ''.
        y_label (str, optional): label of the y axis. Defaults to ''.
        eff ([type], optional): efficiency. Used for saving the plot. Defaults to 0..
        path (str, optional): the path in which the plot will be stored. Defaults to ''.
        name (str, optional): the name of the saved plot. Defaults to ''.
    """    

    plt.close()
    
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

    if not os.path.exists(filename_dict['analysis_path'] + 'images/' + path):
        os.makedirs(filename_dict['analysis_path'] + 'images/' + path)

    plt.savefig(filename_dict['analysis_path'] + 'images/' + path + name + str(np.round(eff,4)) + '.png', dpi=300, facecolor='white')
    
    plt.close()
    
def plot_efficiency(data_col, data_col_with_cut, x_label, title, name, filename_dict, path = '../images/presel_efficiencies'):
    """Plot the efficiency of the applied selections and save the results

    Args:
        data_col (np.array): the data before the selection
        data_col_with_cut (np.array): the data after the selection
        x_label (string): label of the x axis
        title (string): title of the plot
        name (string): name of the saved plot WITHOUT EXTENSION (eg wihtout '.png')
        filename_dict (dictionary): dictionary of the file names
        path (str, optional): path of the saved plot inside analysis results' folder. Defaults to '../images/presel_efficiencies'.
    """    
    
    print('Plotting efficiency')
    plt.close()
    plt.figure()
    hist, bin_edges = np.histogram(data_col, bins=100, density=False)
    hist_cut, bin_edges = np.histogram(data_col_with_cut, bins=bin_edges, density=False)
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_cut/hist),width=(bin_edges[1] - bin_edges[0]), color="blue")
    plt.title(title, fontsize=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('efficiency', fontsize=12)
    plt.savefig(filename_dict['analysis_path'] + path + name + '.png',dpi = 300, facecolor = 'white')
    
    plt.close()
    print('Done\n')

def plot_scatter(hdl, filename_dict, name, vars = None):
    
    print('Plotting scatter of variables: ' + name)
    plt.close()
    df = hdl.get_data_frame()

    if vars == None:
        vars = list(df.columns)

    if len(df.index) > 1000:    #PARAM
        df = df.sample(1000)

    scatter_plot = sns.pairplot(df[vars], plot_kws={'alpha': 0.1}, corner = True)
    scatter_plot.savefig(filename_dict['analysis_path'] + 'images/scatter/' + name + '.png',dpi = 300, facecolor = 'white')

    print('Done\n')

def plot_distributions(tree_hdl, filename_dict, name, vars = None):
    """Plot the distribution of the variables in the tree handler

    Args:
        tree_hdl (hipe4ml.tree_handler): the tree with the data
        filename_dict (dictionary): dictionary of the filenames
        name (string): name of the plot
        vars (list, optional): the variables to plot. None for all variables. Defaults to None.
    """    
    plt.close()
    plots = plot_utils.plot_distr(tree_hdl, column = vars, figsize = ((20,20)))
    plt.savefig(filename_dict['analysis_path'] + 'images/var_distribution/' + name + '.png', dpi = 500, facecolor = 'white')
    plt.close()

def plot_distr_comparison(hdl1, hdl2, name, filename_dict, label_1 = 'df1', label_2 = 'df2', col_names = None, nbins = 100):
    
    print('Plotting comparison of variable distributions')

    df1 = hdl1.get_data_frame()
    df2 = hdl2.get_data_frame()

    if col_names == None:
        col_names = list(df1.columns)
    
    column = []

    for col in col_names:
        if col in list(df2.columns):
            column.append(col)

    plt.close()

    plot_utils.plot_distr([hdl1, hdl2], alpha = 0.5, bins = 100, labels = [label_1, label_2], figsize = ((20,20)), density = True, column = column)
    plt.savefig(filename_dict['analysis_path'] + 'images/var_distribution/' + name[:-1] + '.png')
    plt.close()

    for col in col_names:
        if col in list(df2.columns):
            plt.figure()
            df1[col].hist(alpha = 0.5, bins = nbins, label = label_1, density = True)
            df2[col].hist(alpha = 0.5, bins = nbins, label = label_2, density = True)
            plt.legend()
            plt.savefig(filename_dict['analysis_path'] + 'images/var_distribution/' + name + str(col) + '.png', facecolor = 'white')
            plt.close()

    print('Done\n')

def folder_setup(analysis_path = 'TEST'):   
    """creates all the needed folders for the analysis

    Args:
        analysis_path (str, optional): the name of the analysis folder. Defaults to 'TEST'.
    """     

    if not os.path.exists('../analysis_results'):
        os.makedirs('../analysis_results')
    
    if analysis_path[-1] != '/':
        analysis_path += '/'

    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
        os.makedirs(analysis_path + '/results')

        os.makedirs(analysis_path + '/images/training')
        os.makedirs(analysis_path + '/images/presel_eff')
        os.makedirs(analysis_path + '/images/scatter')
        os.makedirs(analysis_path + '/images/var_distribution')
        os.makedirs(analysis_path + '/images/var_distribution/signal_bckg')
        os.makedirs(analysis_path + '/images/var_distribution/data_bckg')

        os.makedirs(analysis_path + '/model')

        os.makedirs(analysis_path + '/output_data')
    else:
        print('An analysis with the same name already exists. Previous results will be overwritten')

    
def benchmark_hyperparam_optimizers(filename_dict, params, params_range, flag_dict, presel_dict, training_variables='', testsize = 0.5):

    import time
    from sklearn.metrics import roc_auc_score

    N_run = 1

    data_path = filename_dict['data_path']
    analysis_path = filename_dict['analysis_path']

    print('Loading MC signal')
    mc_signal = TreeHandler()
    mc_signal.get_handler_from_large_file(file_name = data_path + filename_dict['MC_signal_filename'],tree_name= filename_dict['MC_signal_table'])        
    print('MC signal loaded\n')

    print('Loading background data for training')
    background_ls = TreeHandler()
    background_ls.get_handler_from_large_file(file_name = data_path + filename_dict['train_bckg_filename'],tree_name= filename_dict['train_bckg_table'])
    background_ls.apply_preselections(presel_dict['train_bckg_presel'])
    background_ls.shuffle_data_frame(size = min(background_ls.get_n_cand(), mc_signal.get_n_cand() * 4))
    print('Done\n')

    train_test_data = train_test_generator([mc_signal, background_ls], [1,0], test_size=testsize)

    if training_variables == '':
        training_variables = train_test_data[0].columns.tolist()

    model_clf = xgb.XGBClassifier()
    model_hdl = ModelHandler(model_clf, training_variables)
    
    times = []
    roc = []

    for i in range(N_run):
        start = time.time()

        model_hdl.optimize_params_bayes(train_test_data, params_range,'roc_auc',njobs=-1)
        model_hdl.train_test_model(train_test_data, )

        y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance

        roc.append(roc_auc_score(train_test_data[3], y_pred_test))

        times.append(time.time() - start)

    print('BAYES OPTIMIZATION WITH SKLEARN')
    print('Mean time : ' + str(np.mean(time)))
    print('Mean ROC : ' + str(np.mean(roc)))
    print('--------------\n')

    for i in range(N_run):
        model_hdl.optimize_params_optuna(train_test_data, params_range,'roc_auc', timeout = np.mean(times) ,njobs=-1)
        model_hdl.train_test_model(train_test_data, )

        y_pred_test = model_hdl.predict(train_test_data[2], True)       #used to evaluate model performance

        roc.append(roc_auc_score(train_test_data[3], y_pred_test))

    print('OPTUNA')
    print('Fixed time : ' + str(np.mean(time)))
    print('Mean ROC : ' + str(np.mean(roc)))
    print('--------------\n')

    
