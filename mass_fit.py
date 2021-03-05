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
import os
from ROOT import gROOT

gROOT.SetBatch(True)
gROOT.LoadMacro("cmb_fit_exp.C")
from ROOT import cmb_fit_exp
gROOT.LoadMacro("cmb_fit_erf.C")
from ROOT import cmb_fit_erf

def mass_fitter(hist,score,efficiency):
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    root_hist.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

    canvas = ROOT.TCanvas()
    root_hist.Draw()
    canvas.Draw()

    #gaus = ROOT.TF1('gaus','gaus',2.96,3.04)
    #bkg = ROOT.TF1('poly','pol 2',2.96,3.04)
    total = ROOT.TF1('total','pol1(0) + gaus(2)',2.96,3.04)
    total.SetParameter(3, 2.992)
    total.SetParLimits(3, 2.99, 2.995)
    total.SetParameter(2, 26)
    total.SetParameter(4, 0.0032)
    total.SetParLimits(4, 0.001, 0.01)
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

    total.SetNpx(1000)
    #root_hist.Fit('total', 'R+', '',2.96,3.04)

    ROOT.gStyle.SetOptFit(1111)
    canvas.SaveAs('./images/results/mass_distr_LS/LS_hist_eff_' + str(np.round(efficiency,4)) + '.png')

    count = total.GetParameters()[2]
    error = total.GetParError(2)
    return count, error


def data_ls_comp_plots(data, ls, scores, efficiencies):
   

    ff = ROOT.TFile('./images/results/data_ls.root','recreate')
    
    for efficiency, score in zip(efficiencies,scores):
        selected_data = data.query('model_output > ' + str(score))
        selected_ls = ls.query('model_output > ' + str(score))

        hist_data = np.histogram(selected_data['m'],bins=34,range=(2.96,3.04))
        hist_ls = np.histogram(selected_ls['m'],bins=34,range=(2.96,3.04))

        aghast_hist = aghast.from_numpy(hist_data)
        root_hist_data = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))
        root_hist_data_erf = aghast.to_root(aghast_hist,'Efficiency_erf ' + str(np.round(efficiency,4)))

        aghast_hist = aghast.from_numpy(hist_ls)
        root_hist_ls = aghast.to_root(aghast_hist, 'Efficiency_ls ' +str(np.round(efficiency,4)))
        root_hist_ls_erf = aghast.to_root(aghast_hist, 'Efficiency_ls_erf ' +str(np.round(efficiency,4)))
        root_hist_data.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

        canvas = ROOT.TCanvas('Efficiency ' + str(np.round(efficiency,4)))

        leg = ROOT.TLegend(.7,.8,.9,.9)
        leg.AddEntry(root_hist_data, 'Data', 'L')
        leg.AddEntry(root_hist_ls, 'Background LS', 'L')
        leg.SetTextSize(0.032)

        root_hist_ls.SetLineColor(ROOT.kRed)
        root_hist_ls.SetMarkerColor(ROOT.kRed)
        root_hist_ls.SetMarkerStyle(7)
        root_hist_data.SetMarkerStyle(7)

        root_hist_data.Draw('PE')
        root_hist_ls.Draw('PE SAME')
        leg.Draw()
        
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)

        canvas.Write()

        cmb_fit_exp(root_hist_ls,root_hist_data, 'Fit_exp_eff_' + str(np.round(efficiency,4)))
        cmb_fit_erf(root_hist_ls_erf,root_hist_data_erf, 'Fit_erf_eff_' + str(np.round(efficiency,4)))

        #canvas.SaveAs('./images/results/mass_distr_LS/LS_hist_eff_' + str(np.round(efficiency,4)) + '.png')

    ff.Close()


def systematic_estimate(data,scores,efficiencies):

    count = []
    errors = []
    i = 0
    for score in scores:
        selected_data_hndl = data.get_subset('model_output > ' + str(score)).get_data_frame()
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hndl['m'],bins=34,range=(2.96,3.04))
        cnt, err = mass_fitter(hist=hist,score=score,efficiency=efficiencies[i])
        count.append(cnt)
        errors.append(err)
        i += 1

    plt.plot(efficiencies,count)
    #plt.fill_between(efficiencies, np.array(count) - np.array(errors), np.array(count) + np.array(errors), alpha = 0.1)
    plt.title('Count of signal (from fit) as a function of efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('Counts')
    plt.annotate('Mean: ' + str(np.round(np.mean(count),4)) + "\n$\sigma$: " + str(np.round(np.std(count),4)) ,xy=(0.68,23))
    #plt.savefig('./images/results/fit_count_eff.png',dpi=300,facecolor = 'white')
    plt.show()

    plt.plot(efficiencies,np.round(count))
    #plt.fill_between(efficiencies, np.array(np.round(count)) - np.array(np.round(errors)), 
    #                    np.array(np.round(count)) + np.array(np.round(errors)), alpha = 0.1)
    plt.title('Count of signal (from fit) as a function of efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('Count')
    plt.annotate('Mean: ' + str(np.round(np.mean(count),4)) + "\n$\sigma$: " + str(np.round(np.std(count),4)) ,xy=(0.68,23))
    #plt.savefig('./images/results/fit_count_eff_rect.png',dpi=300,facecolor = 'white')

#%%
