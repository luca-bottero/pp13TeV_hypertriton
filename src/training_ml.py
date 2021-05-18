import utils, mass_fit, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import score_from_efficiency_array


def save_data_with_scores(tree_handler, filename):
    print('Saving file: ' + filename)
    tree_handler.write_df_to_parquet_files(filename)      #to_parquet, get_handler_from_large_data, get_data_frame

def load_data_with_scores(filename):
    print('Loading file: ' + filename)
    return pd.read_parquet(filename)

def save_eff_scores(eff_array, scores, output_data_path):

    if output_data_path[-1] != '/':
        output_data_path += '/'
    
    eff_name =  output_data_path + 'eff_array.csv'
    scores_name =  output_data_path + 'scores.csv'

    with open(eff_name,'w') as f:
        for val in eff_array:
            f.write(str(np.round(val,4)))
            f.write('\n')
    f.close()

    with open(scores_name,'w') as f:
        for val in scores:
            f.write(str(np.round(val,4)))
            f.write('\n')
    f.close()

def load_eff_scores(output_data_path):
    if output_data_path[-1] != '/':
        output_data_path += '/'

    with open(output_data_path + 'eff_array.csv') as f:
        eff_array = np.array(f.read().splitlines()).astype(np.float)
    f.close()

    with open(output_data_path + 'scores.csv') as f:
        scores = np.array(f.read().splitlines()).astype(np.float)
    f.close()

    return eff_array, scores

def train_model(filename_dict, presel_dict, flag_dict, eff_array, train_vars):

    data_path = filename_dict['data_path']
    analysis_path = filename_dict['analysis_path']

    print('Loading MC signal')
    mc_signal = TreeHandler()
    mc_signal.get_handler_from_large_file(file_name = data_path + filename_dict['MC_signal_filename'],tree_name= "SignalTable")        
    print('MC signal loaded\n')

    #Efficiency plots
    if flag_dict['plot_presel_eff']:
        for var, gvar in zip(['ct','pt'],['gCt','gPt']):             
            utils.plot_efficiency(mc_signal.get_data_frame().query('rej_accept > 0')[gvar],
                                    mc_signal.get_data_frame().query('gReconstructed > 0 & ' + presel_dict['MC_presel'])[var],
                                    var, presel_dict['data_presel'], var, filename_dict, path = 'images/presel_eff/')
    
    mc_signal.apply_preselections(presel_dict['MC_presel'])

    #Scatter plot of the MC signal
    if flag_dict['plot_scatter']:
        utils.plot_scatter(mc_signal, filename_dict, 'signal')

    utils.save_data_description(filename_dict, mc_signal.get_data_frame(), append = False, name = 'MC signal')

    print('Loading background data')
    background_ls = TreeHandler()
    background_ls.get_handler_from_large_file(file_name = data_path + filename_dict['background_filename'],tree_name= "DataTable")
    background_ls.apply_preselections(presel_dict['background_presel'])
    background_ls.shuffle_data_frame(size = min(background_ls.get_n_cand(), mc_signal.get_n_cand() * 4))
    print('Background data loaded\n')

    #Scatter plot of background
    if flag_dict['plot_scatter']:
        utils.plot_scatter(background_ls, filename_dict, 'bckg')

    train_test_data, y_pred_test, model_hdl = utils.train_xgboost_model(mc_signal, background_ls, filename_dict, train_vars, 
                                                                            optimize_bayes = flag_dict['optimize_bayes'])
        
    print('Saving model handler')
    model_hdl.dump_model_handler(analysis_path + '/model/model_hdl')
    print('Model handler saved\n')

    scores = score_from_efficiency_array(train_test_data[3],y_pred_test,eff_array)

    del background_ls
    print('Deleted background data\n')

    print('Loading experimental data')
    data = TreeHandler()
    data.get_handler_from_large_file(file_name = data_path + filename_dict['data_filename'],tree_name= "DataTable",
                                         model_handler = model_hdl)

    data.apply_preselections(presel_dict['data_presel'])
    print('Data loaded\n')

    #Scatter plot of data
    if flag_dict['plot_scatter']:
        utils.plot_scatter(data, filename_dict, 'data')

    utils.save_data_description(filename_dict, data.get_data_frame(), name = 'Data')

    print('Loading background data')
    background_ls = TreeHandler()
    background_ls.get_handler_from_large_file(file_name = data_path + filename_dict['background_filename'],tree_name= "DataTable",
                                        preselection = presel_dict['background_presel'], model_handler = model_hdl)
    print('Background loaded\n')

    utils.save_data_description(filename_dict, background_ls.get_data_frame(), name = 'Background')
    

    #background_ls.apply_model_handler(model_hdl)
    #data.apply_model_handler(model_hdl)

    #print(background_ls)
    utils.plot_distr_comparison(mc_signal,background_ls,'signal_bckg/', 
                                filename_dict, 'MC signal', 'Background')
    utils.plot_distr_comparison(data,background_ls,'data_bckg/',
                                filename_dict, 'Data', 'Background')


    
    save_data_with_scores(background_ls, analysis_path + '/output_data/bckg_ls_scores')
    save_data_with_scores(data, analysis_path + '/output_data/data_scores')    

    save_eff_scores(eff_array, scores, analysis_path + '/output_data')

    del background_ls
    del data
    del mc_signal





