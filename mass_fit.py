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

def mass_fitter(hist,score,efficiency):
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    canvas = ROOT.TCanvas()
    root_hist.Draw()
    canvas.Draw()

    #gaus = ROOT.TF1('gaus','gaus',2.96,3.04)
    #bkg = ROOT.TF1('poly','pol 2',2.96,3.04)
    total = ROOT.TF1('total','pol1(0) + gaus(2)',2.96,3.04)
    total.SetParameter(3, 2.992)
    total.SetParLimits(3, 2.99, 2.995)
    total.SetParameter(2, 25)
    total.SetParameter(4, 0.003)
    #total.SetParLimits(4, 0.006, 0.00001)
    '''
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
    '''
    #total.SetParameters(par)
    root_hist.Fit('total', 'R+', '',2.96,3.04)

    ROOT.gStyle.SetOptFit(1111)
    canvas.SaveAs('./images/results/mass_distr/hist_eff_' + str(np.round(efficiency,4)) + '.png')

    count = total.GetParameters()[2]
    return count


def systematic_estimate(data,scores,efficiencies):

    count = []
    i = 0
    for score in scores:
        selected_data_hndl = data.get_subset('model_output > ' + str(score)).get_data_frame()
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hndl['m'],bins=34,range=(2.96,3.04))
        count.append(mass_fitter(hist=hist,score=score,efficiency=efficiencies[i]))
        i += 1

    plt.plot(efficiencies,count)
    plt.title('Count of signal (from fit) as a function of efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('Count')
    plt.annotate('Mean: ' + str(np.round(np.mean(count),4)) + "\n$\sigma$: " + str(np.round(np.std(count),4)) ,xy=(0.68,23))
    plt.savefig('./images/results/fit_count_eff.png',dpi=300,facecolor = 'white')

#%%
