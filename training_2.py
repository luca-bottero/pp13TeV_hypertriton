#%%
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

def get_skimmed_large_data(data_path, table, cent_classes, pt_bins, ct_bins, training_columns, application_columns, mode, split=''):
    print('\n++++++++++++++++++++++++++++++++++++++++++++++++++')
    print ('\nStarting BDT appplication on large data')

    if mode == 3:
        handlers_path = os.environ['HYPERML_MODELS_3'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_3']

    if mode == 2:
        handlers_path = os.environ['HYPERML_MODELS_2'] + '/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_2']

    executor = ThreadPoolExecutor()
    iterator = uproot.pandas.iterate(data_path, table, executor=executor, reportfile=True)

    df_applied = pd.DataFrame()

    for current_file, data in iterator:
        rename_df_columns(data)
    
        print('current file: {}'.format(current_file))
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))
        
        for cclass in cent_classes:
            for ptbin in zip(pt_bins[:-1], pt_bins[1:]):
                for ctbin in zip(ct_bins[:-1], ct_bins[1:]):
                    info_string = '_{}{}_{}{}_{}{}'.format(cclass[0], cclass[1], ptbin[0], ptbin[1], ctbin[0], ctbin[1])

                    filename_handler = handlers_path + '/model_handler' + info_string + split + '.pkl'
                    filename_efficiencies = efficiencies_path + '/Eff_Score' + info_string + split + '.npy'

                    model_handler = ModelHandler()
                    model_handler.load_model_handler(filename_handler)

                    eff_score_array = np.load(filename_efficiencies)
                    tsd = eff_score_array[1][-1]

                    data_range = f'{ctbin[0]}<ct<{ctbin[1]} and {ptbin[0]}<pt<{ptbin[1]} and {cclass[0]}<=centrality<{cclass[1]}'

                    df_tmp = data.query(data_range)
                    df_tmp.insert(0, 'score', model_handler.predict(df_tmp[training_columns]))

                    df_tmp = df_tmp.query('score>@tsd')
                    df_tmp = df_tmp.loc[:, application_columns]

                    df_applied = df_applied.append(df_tmp, ignore_index=True, sort=False)

    print(df_applied.info(memory_usage='deep'))
    return df_applied

#%%

def get_large_data(data_path, table):

    executor = ThreadPoolExecutor()
    iterator = uproot.pandas.iterate(data_path, table, executor=executor, reportfile=True)

    selected_data = pd.DataFrame()

    for current_file, data in iterator:
        print('current file: {}'.format(current_file))
        print ('start entry chunk: {}, stop entry chunk: {}'.format(data.index[0], data.index[-1]))

        selected_data = selected_data.append(data.query('rej_accept > 0 & pt > 0'))

    

    hndl = TreeHandler(data_path, table)

    #hndl._full_data_frame = selected_data
    return hndl


data_path = os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_tofPID_mtexp.root'
data = get_large_data(data_path,'SignalTable')

print(len(data))
data.head()
# %%

# %%
