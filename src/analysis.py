import ROOT
import utils
import mass_fit
import training_ml as train
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import xgboost as xgb
import yaml
from shutil import copyfile
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
from hipe4ml import plot_utils

#CONFIG

configs = ['ROT_TOF_PID.yaml']

#config_filename = 'OLD_centrality_0dot1_perc.yaml'

for config_filename in configs:

    with open('./config/' + config_filename) as f:
        
        config_params = yaml.load(f, Loader=yaml.FullLoader)
        #print(config_params)

    flag_dict = config_params['flag_dict']

    data_path           = config_params['data_path']   
    analysis_name       = config_params['analysis_name']   
    MC_signal_filename  = config_params['MC_signal_filename']   
    background_filename = config_params['background_filename']   
    data_filename       = config_params['data_filename']   

    presel_dict = config_params['presel_dict']

    eff_array = np.arange(config_params['eff_array'][0],config_params['eff_array'][1],config_params['eff_array'][2])

    #CONFIG SETUP
    analysis_path = '../analysis_results/' + analysis_name

    if analysis_path[-1] != '/':
            analysis_path += '/'

    if data_path[-1] != '/':
            data_path += '/'

    utils.folder_setup(analysis_path = analysis_path)

    copyfile('./config/' + config_filename, analysis_path + config_filename)

    filename_dict  =  {'analysis_path' : analysis_path,
                        'analysis_name' : analysis_name,
                        'MC_signal_filename' : MC_signal_filename,
                        'background_filename' : background_filename,
                        'data_filename' : data_filename,
                        'data_path' : data_path}

    ##########################################################################

    print('\nHypertriton 3-body - pp @ 13 Tev\n')


    
    data_path = filename_dict['data_path']
    analysis_path = filename_dict['analysis_path']

    background_ls_OLD = TreeHandler()
    background_ls_OLD.get_handler_from_large_file(file_name = data_path + 'DataTable_pp_LS_OLD.root',tree_name= "DataTable")
    background_ls_OLD.apply_preselections(presel_dict['background_presel'])

    background_ls_ROT = TreeHandler()
    background_ls_ROT.get_handler_from_large_file(file_name = data_path + filename_dict['background_filename'],tree_name= "DataTable")
    background_ls_ROT.apply_preselections(presel_dict['background_presel'])

    #utils.plot_distr_comparison(background_ls_OLD.get_data_frame(),background_ls_ROT.get_data_frame(), 'COMPARISON/', filename_dict, 'OLD', 'Track Rotation')
    plot_utils.plot_distr([background_ls_OLD, background_ls_ROT], alpha = 0.5,
                            bins = 100, labels = ['OLD', 'Track rotation'], figsize = ((20,20)), density = True)
    
    plt.savefig('comparison.png')
    

    if flag_dict['train_model']:
        print('Starting model training & application\n')
        train.train_model(filename_dict, presel_dict, flag_dict, eff_array)
        print('Model training & application complete\n')


    model_hdl = ModelHandler()
    model_hdl.load_model_handler(analysis_path + '/model/model_hdl')

    print('Model loaded\n')









    eff_array, scores = train.load_eff_scores(analysis_path + 'output_data/')

    data = train.load_data_with_scores(analysis_path + 'output_data/data_scores.parquet.gzip')                #pd dataframe already processed
    print('Data loaded\n')
    #data.query('model_output > -5', inplace = True)         ## PARAM!!!!!
    #print('Query on data applied\n')
    background_ls = train.load_data_with_scores(analysis_path + 'output_data/bckg_ls_scores.parquet.gzip')
    print('Background LS loaded\n')
    #background_ls.query('model_output > -5', inplace = True)            ## PARAM!!!!!
    #print('Query on background LS applied\n')
        

    if flag_dict['plot_m_mppivert']:
        print('Plotting scatter plot for m vs. mppi_vert\n')

        for score, i in zip(scores, eff_array):
            sel = data.query('model_output > ' + str(score))
            utils.scatter_with_hist(sel['m'],sel['mppi_vert'],[34,2.96,3.04],[34,1.08,1.13],
                                    x_label='Hypertriton mass [GeV/c$^2$]',
                                    y_label='$p - \pi$ mass$^2$ [(GeV/c$^2$)$^2$]', eff = i,
                                    path = 'm_mppi/', name = 'dalitz_eff_', filename_dict = filename_dict)
        
    if flag_dict['plot_mppi_mdpi']:
        print('Plotting Dalitz plot: mppi vs. mdpi\n')

        sel_m = data.query('m > 2.989 & m < 2.993')
        for score,i in zip(scores,eff_array):
            sel = sel_m.query('model_output > ' + str(score))
            utils.scatter_with_hist(sel['mppi'], sel['mdpi'],
                                        [50,1.16,1.26],[50,4.07,4.22], path = 'mppi_mdpi/', name = 'dalitz_eff_',
                                        filename_dict = filename_dict,
                                        x_label='$p - \pi$ mass$^2$ [(GeV/c$^2$)$^2$]',
                                        y_label='$d - \pi$ mass$^2$ [(GeV/c$^2$)$^2$]', eff=i)

        del sel_m

    if flag_dict['plot_mppi_mdpi_fine']:
        print('Plotting fine Dalitz plot: mppi vs. mdpi\n')

        sel_m = data.query('m > 2.989 & m < 2.993')
        for score,i in zip(scores,eff_array):
            sel = sel_m.query('model_output > ' + str(score))
            utils.scatter_with_hist(sel['mppi'], sel['mdpi'],
                                        [10,1.235,1.25],[20,4.15,4.21], path = 'mppi_mdpi_fine/', name = 'dalitz_eff_',
                                        filename_dict = filename_dict,
                                        x_label='$p - \pi$ mass$^2$ [(GeV/c$^2$)$^2$]',
                                        y_label='$d - \pi$ mass$^2$ [(GeV/c$^2$)$^2$]', eff=i)

        del sel_m

    if flag_dict['root_plots']:
        mass_fit.data_ls_comp_plots(data,background_ls,scores,eff_array, filename_dict)

   

