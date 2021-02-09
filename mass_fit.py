#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uproot
import os
import xgboost as xgb
import aghast
import ROOT
from hipe4ml.model_handler import ModelHandler
from hipe4ml import plot_utils

def mass_fitter(hist,score):
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Score_' + str(np.round(score,4)))
    canvas = ROOT.TCanvas()
    root_hist.Draw()
    canvas.Draw()
    gaus = ROOT.TF1('gaus','gaus',2.985,3.005)
    bkg = ROOT.TF1('linear','pol 2',2.96,3.04)
    total = ROOT.TF1('total','gaus + [3]',2.96,3.04)
    root_hist.Fit('gaus','R')
    root_hist.Fit('linear','R+')
    canvas.SaveAs('./images/mass_sys/score_' + str(np.round(score,4)) + '.png')


def systematic_estimate(data,scores):
    for score in scores:
        selected_data_hndl = data.get_subset('model_output > ' + str(score)).get_data_frame()
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hndl['m'],bins=34,range=(2.96,3.04))
        mass_fitter(hist=hist,score=score)




#%%
