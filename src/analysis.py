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
train_model = True
print_m_mppivert = False
print_mppi_mdpi = False


if train_model:
    train.train_model()

model_hdl = ModelHandler()
model_hdl.load_model_handler('../model/model_hdl')
print('Model loaded\n')

data = train.load_data_with_scores('../data/data_scores.csv')                #pd dataframe already processed
background_ls = train.load_data_with_scores('../data/bckg_ls_scores.csv')

eff_array, scores = train.load_eff_scores()
print('Datas loaded\n')

if print_m_mppivert:
    print('Plotting scatter plot for m vs. mppi_vert')

    for score, i in zip(scores, eff_array):
        sel = data.query('model_output > ' + str(score))
        utils.scatter_with_hist(sel['m'],sel['mppi_vert'],[34,2.96,3.04],[34,1.08,1.13],
                                x_label='Hypertriton mass [GeV/c$^2$]',
                                y_label='$p - \pi$ mass [GeV/c$^2$]', eff = i,name = '/m_mppi/dalitz_eff_')
    
if print_mppi_mdpi:
    print('Plotting Dalitz plot: mppi vs. mdpi')

    sel_m = data.query('m > 2.989 & m < 2.993')
    for score,i in zip(scores,eff_array):
        sel = sel_m.query('model_output > ' + str(score))
        utils.scatter_with_hist(sel['mppi'], sel['mdpi'],
                                    [50,1.16,1.26],[50,4.07,4.22], name = '/mppi_mdpi/dalitz_eff_',
                                    x_label='$p - \pi$ mass [GeV/c$^2$]',
                                    y_label='$d - \pi$ mass [GeV/c$^2$]', eff=i)

    del sel_m



mass_fit.data_ls_comp_plots(data,background_ls,scores,eff_array)



