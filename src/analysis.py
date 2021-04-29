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
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
from hipe4ml import plot_utils

#CONFIG PARAMETERS
train_model      = False
optimize_bayes   = False
print_m_mppivert = True
print_mppi_mdpi  = True

data_path           = '../data/'
analysis_name       = 'NEW_trees'
MC_signal_filename  = 'SignalTable_pp13TeV_mtexp.root'
background_filename = 'DataTable_pp_LS.root'
data_filename       = 'DataTable_pp.root'



'''



parser = argparse.ArgumentParser()
parser.add_argument('config', help='Path to the YAML configuration file')
parser.add_argument('-syst', '--syst', help='Compute systematic uncertainties', action='store_true')
parser.add_argument('-s', '--significance', help='Use the BDT efficiency selection from the significance scan', action='store_true')
args = parser.parse_args()

with open(os.path.expandvars(args.config), 'r') as stream:
    try:
        params = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

SYSTEMATICS = args.syst
SIGNIFICANCE_SCAN = args.significance

# TRAINING_DIR = params["TRAINING_DIR"]
TRAINING_DIR = params["TRAINING_DIR"]
CENT_CLASS = params['CENTRALITY_CLASS'][0]
PT_BINS = params['PT_BINS']
CT_BINS = params['CT_BINS']
EFF_MIN, EFF_MAX, EFF_STEP = params['BDT_EFFICIENCY']
EFF_ARRAY = np.around(np.arange(EFF_MIN, EFF_MAX, EFF_STEP), 2)
FIX_EFF = 0.6 if not SIGNIFICANCE_SCAN else 0
SYSTEMATICS_COUNTS = 100000


'''







#CONFIG SETUP
analysis_path = '../analysis_results/' + analysis_name

if analysis_path[-1] != '/':
        analysis_path += '/'

if data_path[-1] != '/':
        data_path += '/'

utils.folder_setup(analysis_path = analysis_path)

filename_dict  =  {'analysis_path' : analysis_path,
                    'MC_signal_filename' : MC_signal_filename,
                    'background_filename' : background_filename,
                    'data_filename' : data_filename,
                    'data_path' : data_path}

print('\nHypertriton pp 3-body 13 Tev\n')

if train_model:
    print('Starting model training & application\n')
    train.train_model(filename_dict, optimize_bayes)
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
    

if print_m_mppivert:
    print('Plotting scatter plot for m vs. mppi_vert\n')

    for score, i in zip(scores, eff_array):
        sel = data.query('model_output > ' + str(score))
        utils.scatter_with_hist(sel['m'],sel['mppi_vert'],[34,2.96,3.04],[34,1.08,1.13],
                                x_label='Hypertriton mass [GeV/c$^2$]',
                                y_label='$p - \pi$ mass [GeV/c$^2$]', eff = i,
                                path = 'm_mppi/', name = 'dalitz_eff_', filename_dict = filename_dict)
    
if print_mppi_mdpi:
    print('Plotting Dalitz plot: mppi vs. mdpi\n')

    sel_m = data.query('m > 2.989 & m < 2.993')
    for score,i in zip(scores,eff_array):
        sel = sel_m.query('model_output > ' + str(score))
        utils.scatter_with_hist(sel['mppi'], sel['mdpi'],
                                    [50,1.16,1.26],[50,4.07,4.22], path = 'mppi_mdpi/', name = 'dalitz_eff_',
                                    filename_dict = filename_dict,
                                    x_label='$p - \pi$ mass [GeV/c$^2$]',
                                    y_label='$d - \pi$ mass [GeV/c$^2$]', eff=i)

    del sel_m

mass_fit.data_ls_comp_plots(data,background_ls,scores,eff_array, filename_dict)

