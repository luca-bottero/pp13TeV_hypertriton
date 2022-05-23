#%%
import ROOT
ROOT.gROOT.SetBatch(True)

import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uproot
import xgboost as xgb
import aghast
from hipe4ml.model_handler import ModelHandler
from hipe4ml import plot_utils
from array import array


def normalize_ls(data_counts, ls_counts, bins):
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    side_region = np.logical_or(bin_centers<2.992-2*0.0025, bin_centers>2.992+2*0.0025)
    
    side_data_counts = np.sum(data_counts[side_region])
    side_ls_counts = np.sum(ls_counts[side_region])
    scaling_factor = side_data_counts/side_ls_counts

    return scaling_factor

def h1_invmass(counts, mass_range=[2.96, 3.04] , bins=34, name=''):
    th1 = ROOT.TH1D(f'{name}',f'{name}', int(bins), mass_range[0], mass_range[1])
    for index in range(0, len(counts)):
        th1.SetBinContent(index+1, counts[index])
        # th1.SetBinError(index + 1, np.sqrt(counts[index]))
    th1.SetDirectory(0)
    return th1

def mass_fitter(hist, score, efficiency, filename_dict):
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    ff = ROOT.TFile(filename_dict['analysis_path'] + 'results/data_ls_integral.root','update')

    #root_hist = h1_invmass(hist[0], name = 'eff_' + str(np.round(efficiency,4)), bins = 34)

    root_hist.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

    canvas = ROOT.TCanvas(name = 'eff_' + str(np.round(efficiency,4)))
    root_hist.Draw('PE')
    canvas.Draw('PE SAME')

    gaus = ROOT.TF1('gaus','gaus',2.96,3.04)
    bkg = ROOT.TF1('sigmoid','[0]/(1 + TMath::Exp(-[1]*(x-[2]))) + [3]',2.96,3.04)
    bkg.SetParLimits(3, 0., 5.)
    bkg.SetParameter(0, 15)
    bkg.SetParLimits(2, 2.99, 3.0)
    bkg.SetParLimits(1, 10., 500.)
    #bkg.SetParLimits(3, 2.)

    total = ROOT.TF1('total','gaus(0) + [3]/(1 + TMath::Exp(-[4]*(x-[5]))) + [6]',2.96,3.04)

    '''total.SetParameter(3, 2.992)
    total.SetParLimits(3, 2.99, 2.995)
    total.SetParameter(2, 100)  #26
    total.SetParameter(4, 0.0032)
    total.SetParLimits(4, 0.001, 0.01)'''
    '''total.SetParameter(1, 2.992)
    total.SetParLimits(1, 2.99, 2.995)
    total.SetParameter(0, 100)  #26
    total.SetParameter(2, 0.0032)
    total.SetParLimits(2, 0.001, 0.01)'''
    
    gaus.SetLineColor( 4 )
    bkg.SetLineColor( 2 )
    total.SetLineColor( 3 )

    gaus.SetLineStyle(7)
    bkg.SetLineStyle(7)

    m = 2.9912
    d_m = 3 * 0.002

    root_hist.Fit('gaus','R0','',m - d_m, m + d_m)
    root_hist.Fit('sigmoid','R0+','',2.96,3.04)

    #gaus.SetNpx(500)
    gaus.Draw('SAME')

    #bkg.SetNpx(500)
    bkg.Draw('SAME')
    
    par = array( 'd', 7*[0.] )
    gaus_par = gaus.GetParameters()
    bkg_par = bkg.GetParameters()
    par[0], par[1], par[2] = gaus_par[0], gaus_par[1], gaus_par[2]
    par[3], par[4], par[5], par[6] = bkg_par[0], bkg_par[1], bkg_par[2], bkg_par[3]
    
    s = gaus.Integral(m - d_m, m + d_m) * 100
    b = bkg.Integral(m - d_m, m + d_m) * 100
    significance = s/np.sqrt(s+b)
    #significance = s/np.sqrt(b)

    print('Significance at', efficiency, 'efficiency')
    print('Signal', s)
    print('Background', b)
    print('Signal-Background ratio', s/b)
    print('Significance', significance)
    print('Significance x efficiency', efficiency * significance)

    total.SetParameters(par)

    total.SetNpx(1000)
    #root_hist.Fit('total', 'R+', '',2.96,3.04)
    total.Draw('SAME')
    root_hist.SetMarkerStyle(7)

    ROOT.gStyle.SetOptFit(1111)
    #ROOT.gStyle.SetOptFit(0)

    canvas.Write()

    canvas.SaveAs(filename_dict['analysis_path'] + 'images/mass_distr_LS/LS_hist_eff_' + str(np.round(efficiency,4)) + '.png')

    '''count = total.GetParameters()[2]
    error = total.GetParError(2)'''

    count = root_hist.GetEntries()
    error = np.sqrt(count)


    ff.Close()

    return count, error, significance


def data_ls_comp_plots(data, ls, scores, efficiencies, filename_dict, flag_dict):
    ROOT.gROOT.LoadMacro("cmb_fit_exp.C")
    ROOT.gROOT.LoadMacro("cmb_fit_erf.C")
    from ROOT import cmb_fit_exp, cmb_fit_erf

    n_bins = 80     #PARAM !!!

    ff = ROOT.TFile(filename_dict['analysis_path'] + 'results/data_ls.root','recreate')

    sigs_erf = []
    sigs_exp = []
    
    for efficiency, score in zip(efficiencies,scores):
        selected_data = data.query('model_output > ' + str(score))
        selected_ls = ls.query('model_output > ' + str(score))

        hist_data = np.histogram(selected_data['m'],bins=n_bins,range=(2.96,3.04))

        if flag_dict['norm_on_sidebands']:
            hist_ls_TMP = np.histogram(selected_ls['m'],bins=n_bins,range=(2.96,3.04))
            ls_counts = np.array(hist_ls_TMP[0]) * normalize_ls(hist_data[0], hist_ls_TMP[0], hist_data[1])
            #hist_ls = np.histogram(list(np.round(ls_counts)) ,bins=n_bins,range=(2.96,3.04))
        else:
            hist_ls = np.histogram(selected_ls['m'],bins=n_bins,range=(2.96,3.04))
            ls_counts = np.array(hist_ls[0])

        ''' if flag_dict['norm_on_sidebands']:
            hist_ls = list(hist_ls)
            hist_ls[0] = np.array(hist_ls[0])*normalize_ls(hist_data[0], hist_ls[0], hist_data[1])
            #hist_ls = tuple(hist_ls)'''

        '''aghast_hist = aghast.from_numpy(hist_data)
        root_hist_data = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))
        root_hist_data_erf = aghast.to_root(aghast_hist,'Efficiency_erf ' + str(np.round(efficiency,4)))

        aghast_hist = aghast.from_numpy(hist_ls)
        root_hist_ls = aghast.to_root(aghast_hist, 'Efficiency_ls ' +str(np.round(efficiency,4)))
        root_hist_ls_erf = aghast.to_root(aghast_hist, 'Efficiency_ls_erf ' +str(np.round(efficiency,4)))
        root_hist_data.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')'''

        root_hist_data = h1_invmass(hist_data[0], name = 'eff_' + str(np.round(efficiency,4)), bins = n_bins)
        root_hist_data_erf = h1_invmass(hist_data[0], name = 'eff_erf_' + str(np.round(efficiency,4)), bins = n_bins)

        root_hist_ls = h1_invmass(np.round(ls_counts), name = 'eff_' + str(np.round(efficiency,4)), bins = n_bins)
        root_hist_ls_erf = h1_invmass(np.round(ls_counts), name = 'eff_erf_' + str(np.round(efficiency,4)), bins = n_bins)

        root_hist_data.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

        canvas = ROOT.TCanvas('eff_' + str(np.round(efficiency,4)))

        leg = ROOT.TLegend(.6,.8,.8,.9)
        leg.AddEntry(root_hist_data, 'Data', 'L')
        leg.AddEntry(root_hist_ls, 'Background LS', 'L')
        leg.SetTextSize(0.032)

        root_hist_ls.SetLineColor(ROOT.kRed)
        root_hist_ls.SetMarkerColor(ROOT.kRed)
        root_hist_ls.SetMarkerStyle(7)
        root_hist_data.SetMarkerStyle(7)

        if root_hist_data.GetMaximum() > root_hist_ls.GetMaximum():
            root_hist_data.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

            root_hist_data.Draw('PE')
            root_hist_ls.Draw('PE SAME')
        else:
            root_hist_ls.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

            root_hist_ls.Draw('PE')
            root_hist_data.Draw('PE SAME')

        leg.Draw()
        
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)

        canvas.Write()

        cmb_fit_exp(root_hist_ls,root_hist_data, 'fit_exp_eff_' + str(np.round(efficiency,4)))
        significance_erf = cmb_fit_erf(root_hist_ls_erf,root_hist_data_erf, 'fit_erf_eff_' + str(np.round(efficiency,4)))

        sigs_erf.append(significance_erf)

        #canvas.SaveAs('../analysis/images/mass_distr_LS/LS_hist_eff_' + str(np.round(efficiency,4)) + '.png')

    ff.Close()

    print(sigs_erf)

    utils.plot_significance(efficiencies, sigs_erf, filename_dict, 'erf')

def systematic_estimate(data,scores,efficiencies, filename_dict):

    count = []
    errors = []
    sigs = []
    i = 0

    ff = ROOT.TFile(filename_dict['analysis_path'] + 'results/data_ls_integral.root','recreate')
    ff.Close()

    for score in scores:
        selected_data_hndl = data.query('model_output > ' + str(score))
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hndl['m'],bins=36,range=(2.96,3.04))
        cnt, err, significance = mass_fitter(hist=hist ,score=score, efficiency=efficiencies[i], filename_dict=filename_dict)
        count.append(cnt)
        errors.append(err)
        sigs.append(significance)
        i += 1

    '''plt.plot(efficiencies,count)
    #plt.fill_between(efficiencies, np.array(count) - np.array(errors), np.array(count) + np.array(errors), alpha = 0.1)
    plt.title('Count of signal (from fit) as a function of efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('Counts')
    plt.annotate('Mean: ' + str(np.round(np.mean(count),4)) + "\n$\sigma$: " + str(np.round(np.std(count),4)) ,xy=(0.68,23))
    plt.savefig('../analysis/images/mass_distr/fit_count_eff.png',dpi=300,facecolor = 'white')
    plt.show()

    plt.plot(efficiencies,np.round(count))
    #plt.fill_between(efficiencies, np.array(np.round(count)) - np.array(np.round(errors)), 
    #                    np.array(np.round(count)) + np.array(np.round(errors)), alpha = 0.1)
    plt.title('Count of signal (from fit) as a function of efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('Count')
    plt.annotate('Mean: ' + str(np.round(np.mean(count),4)) + "\n$\sigma$: " + str(np.round(np.std(count),4)) ,xy=(0.68,23))
    plt.savefig('../analysis/images/mass_distr/fit_count_eff_rect.png',dpi=300,facecolor = 'white')'''

    sigs = [e*s for e,s in zip(efficiencies, sigs)]

    plt.close()
    plt.plot(efficiencies, sigs)
    plt.title('Significance as a function of BDT efficiency')
    plt.xlabel('BDT efficiency')
    plt.ylabel('Significance * BDT efficiency')
    plt.savefig(filename_dict['analysis_path'] + 'results/significance_plot.png', dpi = 300, facecolor = 'white')
    plt.close()




#%%
