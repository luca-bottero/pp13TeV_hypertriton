import utils
import training_ml as train
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

#Config
test_run = True             #if _Test trees as used
train_model = True
optimize_bayes = True
print_m_mppivert = False
print_mppi_mdpi = False

'''
taglio centralità:
centrality: in p-Pb o Pb-Pb => overlap
            p-p => molteplicità di particelle cariche
            percentili di centralità (su estremi di molteplicità) (nucleoni spettatori, tanti = poca centralità e viceversa in Pb)
            [centrality] 

            cut: < 0.17 - 17% più centrali 

pt: cut > 1.5 GeV/c, ci aspettiamo disomogeneità tra dati e LS
    efficienza per questo taglio !!!!!!!!

cut sia bckg sia signal e poi ai dati

downscaling bckg: 4-5 volte il segnale

hyperparam di francesco

poi ottimizzazione

'''


if train_model:
    print('Starting model training & application\n')
    train.train_model(optimize_bayes, test_run)
    print('Model training & application complete\n')

print('\nHypertriton pp 3-body 13 Tev\n')

model_hdl = ModelHandler()
if test_run:
    model_hdl.load_model_handler('../model/model_hdl_Test')
else:
    model_hdl.load_model_handler('../model/model_hdl')


print('Model loaded\n')

eff_array, scores = train.load_eff_scores()

if test_run:
    data = train.load_data_with_scores('../data/data_scores_Test.csv')                #pd dataframe already processed
    print('Data loaded\n')
    #data.query('model_output > -5', inplace = True)         ## PARAM!!!!!
    #print('Query on data applied\n')
    background_ls = train.load_data_with_scores('../data/bckg_ls_scores_Test.csv')
    print('Background LS loaded\n')
    #background_ls.query('model_output > -5', inplace = True)            ## PARAM!!!!!
    #print('Query on background LS applied\n')
else:
    data = train.load_data_with_scores('../data/data_scores.csv')                #pd dataframe already processed
    print('Data loaded\n')
    #data.query('model_output > -5', inplace = True)         ## PARAM!!!!!
    #print('Query on data applied\n')
    background_ls = train.load_data_with_scores('../data/bckg_ls_scores.csv')
    print('Background LS loaded\n')
    #background_ls.query('model_output > -5', inplace = True)            ## PARAM!!!!!
    #print('Query on background LS applied\n')
    

if print_m_mppivert:
    print('Plotting scatter plot for m vs. mppi_vert\n')

    for score, i in zip(scores, eff_array):
        sel = data.query('model_output > ' + str(score))
        utils.scatter_with_hist(sel['m'],sel['mppi_vert'],[34,2.96,3.04],[34,1.08,1.13],
                                x_label='Hypertriton mass [GeV/c$^2$]',
                                y_label='$p - \pi$ mass [GeV/c$^2$]', eff = i,path = 'm_mppi/', name = 'dalitz_eff_')
    
if print_mppi_mdpi:
    print('Plotting Dalitz plot: mppi vs. mdpi\n')

    sel_m = data.query('m > 2.989 & m < 2.993')
    for score,i in zip(scores,eff_array):
        sel = sel_m.query('model_output > ' + str(score))
        utils.scatter_with_hist(sel['mppi'], sel['mdpi'],
                                    [50,1.16,1.26],[50,4.07,4.22], path = 'mppi_mdpi/', name = 'dalitz_eff_',
                                    x_label='$p - \pi$ mass [GeV/c$^2$]',
                                    y_label='$d - \pi$ mass [GeV/c$^2$]', eff=i)

    del sel_m



mass_fit.data_ls_comp_plots(data,background_ls,scores,eff_array)



