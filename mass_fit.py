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
from array import array

def mass_fitter(hist,score):
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Score_' + str(np.round(score,4)))

    canvas = ROOT.TCanvas()
    root_hist.Draw()
    canvas.Draw()

    gaus = ROOT.TF1('gaus','gaus',2.96,3.04)
    bkg = ROOT.TF1('poly','pol 2',2.96,3.04)
    total = ROOT.TF1('total','gaus + pol 2',2.96,3.04)
    
    gaus.SetLineColor( 1 )
    bkg.SetLineColor( 2 )
    total.SetLineColor( 3 )

    root_hist.Fit('gaus','R','',2.985,3.005)
    root_hist.Fit('poly','R+','',2.96,3.04)

    par = array( 'd', 6*[0.] )
    gaus_par = gaus.GetParameters()
    poly_par = bkg.GetParameters()
    par[0], par[1], par[2] = gaus_par[0], gaus_par[1], gaus_par[2]
    par[3], par[4], par[5] = poly_par[0], poly_par[1], poly_par[2]

    total.SetParameters(par)
    root_hist.Fit('total', 'R+','',2.96,3.04 )

    ROOT.gStyle.SetOptFit(1111)
    canvas.SaveAs('./images/mass_sys/score_' + str(np.round(score,4)) + '.png')


def systematic_estimate(data,scores):
    for score in scores:
        selected_data_hndl = data.get_subset('model_output > ' + str(score)).get_data_frame()
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hndl['m'],bins=34,range=(2.96,3.04))
        mass_fitter(hist=hist,score=score)




#%%
