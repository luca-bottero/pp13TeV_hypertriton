#%%
from logging import root
from re import S
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
    # side_region = np.logical_or(bin_centers<2.992-2*0.0025, bin_centers>2.992+2*0.0025)
    side_region = bin_centers>3.01
    print("NORMALISING LS")
    
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

def mass_fitter(hist, score, efficiency, filename_dict, presel_eff, N_ev = 900e6): #7.581079e9    8.119086e9 869085971
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    #root_hist = h1_invmass(hist[0], name = 'eff_' + str(np.round(efficiency,4)), bins = 34)

    root_hist.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

    canvas = ROOT.TCanvas()
    root_hist.Draw('PE')
    canvas.Draw('PE SAME')

    #gaus = ROOT.TF1('gaus','gaus',2.96,3.04)
    bkg = ROOT.TF1('erf','[0]/(1 + TMath::Exp(-[1]*(x-[2]))) + [3]',2.96,3.04)
    bkg.SetParLimits(3, 0., 5.)
    bkg.SetParameter(0, 15)
    bkg.SetParLimits(2, 2.99, 3.0)
    bkg.SetParLimits(1, 10., 500.)
    #bkg.SetParLimits(3, 2.)
    #total = ROOT.TF1('total','gaus(0) + poly(3)',2.96,3.04)
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
    
    #gaus.SetLineColor( 1 )
    bkg.SetLineColor( 2 )
    #total.SetLineColor( 3 )

    m = 2.9912
    d_m = 3 * 0.002

    low_bin = root_hist.FindBin(m-d_m)      # 12
    high_bin = root_hist.FindBin(m+d_m)     # 17

    original_bin_contents = []                  # remove mass peak from bkg fit
    for bin in range(low_bin, high_bin + 1):
        original_bin_contents.append(root_hist.GetBinContent(bin))
        root_hist.SetBinContent(bin,0)

    root_hist.Fit('gaus','R','',m - d_m, m + d_m)
    root_hist.Fit('erf','R+','',2.96,3.04)

    for bin in range(low_bin, high_bin + 1):    # restore mass peak counts
        root_hist.SetBinContent(bin,original_bin_contents[bin - low_bin])
    
    par = array( 'd', 6*[0.] )
    #gaus_par = gaus.GetParameters()
    bkg_par = bkg.GetParameters()
    '''par[0], par[1], par[2] = gaus_par[0], gaus_par[1], gaus_par[2]
    par[3], par[4], par[5] = bkg_par[0], bkg_par[1], bkg_par[2]'''
    
    #s = gaus.Integral(m - d_m, m + d_m) * 100
    s = 2.59e-7 * 0.4 * 2 * N_ev * presel_eff * efficiency
    sigma_s = 0.5e-7 * 0.4 * 2 * N_ev * presel_eff * efficiency
    b = bkg.Integral(m - d_m, m + d_m) / root_hist.GetBinWidth(1)
    significance = s/np.sqrt(s+b)
   
    print('Bin width', root_hist.GetBinWidth(1))
    print('Significance at', efficiency, 'efficiency')
    print('Preselection efficiency', presel_eff)
    print('Signal', s)
    print('Background', b)
    print('Signal-Background ratio', s/b)
    print('Significance', significance)
    print('Significance x efficiency', efficiency * significance)

    #total.SetParameters(par)

    #total.SetNpx(1000)
    #root_hist.Fit('total', 'R+', '',2.96,3.04)

    ROOT.gStyle.SetOptFit(1111)
    #ROOT.gStyle.SetOptFit(0)
    canvas.SaveAs(filename_dict['analysis_path'] + 'images/mass_distr_LS/LS_hist_eff_' + str(np.round(efficiency,4)) + '.png')

    '''count = total.GetParameters()[2]
    error = total.GetParError(2)'''

    count = root_hist.GetEntries()
    error = np.sqrt(count)

    return s, sigma_s, b, significance


def signal_fitter(hist, efficiency, significance, sig_err, bkg_hist=None, only_gaus=False): #7.581079e9    8.119086e9 869085971
    aghast_hist = aghast.from_numpy(hist)
    root_hist = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    if bkg_hist is not None:
        aghast_hist = aghast.from_numpy(bkg_hist)
        bkg_hist = aghast.to_root(aghast_hist, 'Efficiency ' + str(np.round(efficiency,4)))

        root_hist.Add(bkg_hist, -1)

    #root_hist = h1_invmass(hist[0], name = 'eff_' + str(np.round(efficiency,4)), bins = 34)

    root_hist.SetTitle('Counts as a function of mass;m [GeV/c^{2}];Counts')

    canvas = ROOT.TCanvas()
    root_hist.Draw('PE')
    canvas.Draw('PE SAME')

    gaus = ROOT.TF1('gaus','gaus',2.96,3.04)
    bkg = ROOT.TF1('bkg','[0]/(1 + TMath::Exp(-[1]*(x-[2]))) + [3]',2.96,3.04)
    total = ROOT.TF1('total','gaus(0) + [3]/(1 + TMath::Exp(-[4]*(x-[5]))) + [6]',2.96,3.04)
    bkg.SetParLimits(3, 0., 5.)
    bkg.SetParameter(0, 15)
    bkg.SetParLimits(2, 2.99, 3.0)
    bkg.SetParLimits(1, 10., 500.)
    #bkg.SetParLimits(3, 2.)

    total.SetParLimits(6, 0., 5.)
    total.SetParameter(3, 15)
    total.SetParLimits(5, 2.99, 3.0)
    total.SetParLimits(4, 10., 500.)

    '''gaus.SetParLimits(0, 0., 30.)
    gaus.SetParLimits(1, 10., 500.)
    gaus.SetParLimits(2, 2.99, 3.0)
    gaus.SetParLimits(3, 0., 5.)'''
   

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
    
    gaus.SetLineColor( ROOT.kGreen )
    bkg.SetLineColor( ROOT.kRed )
    total.SetLineColor( ROOT.kBlue )

    if not only_gaus:
        gaus.SetLineStyle(7)   #7 = dotted
    #bkg.SetLineStyle(7)

    m = 2.9912
    d_m = 3 * 0.002

    low_bin = root_hist.FindBin(m-d_m)      # 12
    high_bin = root_hist.FindBin(m+d_m)     # 17

    original_bin_contents = []                  # remove mass peak from bkg fit
    for bin in range(low_bin, high_bin + 1):
        original_bin_contents.append(root_hist.GetBinContent(bin))
        root_hist.SetBinContent(bin,0)

    root_hist.Fit('bkg','R0+','',2.96,3.04)

    for bin in range(low_bin, high_bin + 1):    # restore mass peak counts
        root_hist.SetBinContent(bin,original_bin_contents[bin - low_bin])


    '''# Remove background when fitting mass peak

    original_bin_contents = []                  # remove mass peak from bkg fit
    for bin in range(low_bin, high_bin + 1):
        original_bin_contents.append(root_hist.GetBinContent(bin))
        fit_bkg =  bkg.Eval(root_hist.GetXaxis().GetBinCenter(bin))
        root_hist.SetBinContent(bin, max(0, bin,root_hist.GetBinContent(bin) - fit_bkg))

        print(root_hist.GetBinContent(bin), fit_bkg)

    root_hist.Fit('gaus','R0+','',m - d_m, m + d_m)

    for bin in range(low_bin, high_bin + 1):    # restore mass peak counts
        root_hist.SetBinContent(bin,original_bin_contents[bin - low_bin])'''

    root_hist.Fit('gaus','R0+','',m-d_m, m+d_m)

    par = array( 'd', 7*[0.] )
    gaus_par = gaus.GetParameters()
    bkg_par = bkg.GetParameters()
    par[0], par[1], par[2] = gaus_par[0], gaus_par[1], gaus_par[2]
    par[3], par[4], par[5], par[6] = bkg_par[0], bkg_par[1], bkg_par[2], bkg_par[3]

    total.SetParameters(par)

    r = root_hist.Fit('total', 'RS0+', '', 2.96, 3.04)

    gaus.SetNpx(1000)
    bkg.SetNpx(1000)
    total.SetNpx(1000)

    #gaus.Draw('SAME')
    #bkg.Draw('SAME')

    if only_gaus:
        gaus.SetLineColor( ROOT.kBlue )
        gaus.Draw('SAME')
    else:
        total.Draw('SAME')

    root_hist.SetStats(0)

    count = root_hist.GetEntries()

    count = gaus.Integral(m - d_m, m + d_m) / root_hist.GetBinWidth(1)
    b = bkg.Integral(m - d_m, m + d_m) / root_hist.GetBinWidth(1)

    measured_significance = count/np.sqrt(count + b)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.6 ,0.88, "Exp. Signif. (3 #sigma) " + str(np.round(significance,1)) +  " #pm " + str(np.round(3*sig_err,1)) )
    latex.DrawLatex(0.6 ,0.83, "Calc. Signif. " + str(np.round(measured_significance,1)) )#+  " #pm " + str(np.round(3*sig_err,1)) )
    latex.DrawLatex(0.6 ,0.78, "S (3 #sigma) " + str(np.round(count)) +  " #pm " + str(np.round(3*np.sqrt(count))) )
    latex.DrawLatex(0.6 ,0.73, "B (3 #sigma) " + str(np.round(b)) +  " #pm " + str(np.round(3*np.sqrt(b))) )
    latex.DrawLatex(0.6 ,0.68, "S/B " + str(np.round(count/b)) )

    canvas.SetName('eff_' + str(efficiency))
    canvas.Write()

    #cov =  r.GetCovarianceMatrix()
    sigma_s = np.sqrt(count) #gaus.IntegralError(m - d_m, m + d_m, r.GetParams(), cov.GetMatrixArray()) / root_hist.GetBinWidth(1)

    fitted_mass = total.GetParameters()[1]
    fitted_mass_err = total.GetParError(1)
    mass_variance = total.GetParameters()[2]
    mass_variance_err = total.GetParError(2)
    
    print('Efficiency', efficiency)
    print('Count {0} +- {1}'.format(count, sigma_s))
    print('Background', b)
    print('Fitted mass ', fitted_mass, '+-', fitted_mass_err)
    print('Mass variance', mass_variance, '+-', mass_variance_err)

    return count, b, sigma_s, (fitted_mass, fitted_mass_err), (mass_variance, mass_variance_err)


def data_ls_comp_plots(data, ls, scores, efficiencies, filename_dict, flag_dict):
    ROOT.gROOT.LoadMacro("cmb_fit_exp.C")
    ROOT.gROOT.LoadMacro("cmb_fit_erf.C")
    from ROOT import cmb_fit_exp, cmb_fit_erf

    n_bins = 36     #PARAM !!!

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

        if flag_dict['subtract_bkg']:
            root_hist_data = root_hist_data - root_hist_ls

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

    #utils.plot_significance(efficiencies, sigs_erf, filename_dict, 'erf')

def systematic_estimate(data, bkg_hdl, scores, efficiencies, presel_eff, filename_dict, flag_dict, simult = False):

    s_arr = []
    sigma_s_arr = []
    b_arr = []
    sigs = []
    i = 0
    for score in scores:
        selected_data_hndl = data.query('model_output > ' + str(score))
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hndl['m'],bins=36,range=(2.96,3.04))
        if not simult:
            s, sigma_s, b, significance = mass_fitter(hist=hist ,score=score, efficiency=efficiencies[i], 
                                                        presel_eff=presel_eff, filename_dict=filename_dict)
        else:
            s, sigma_s, b; significance = 1,1
        s_arr.append(s)
        sigma_s_arr.append(sigma_s)
        b_arr.append(b)
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

    sigs = np.array([e*s for e,s in zip(efficiencies, sigs)])
    print('\n\nMax significance: ', sigs.max())
    print('BDT Efficiency at max significance: ', efficiencies[sigs.argmax()])

    def err(s,b, sigma_s):
        #sigma_s = 0.5 #np.sqrt(s)       da propagare (y)
        sigma_b = np.sqrt(b)

        error = (sigma_b**2 * s**2 + sigma_s**2 * (2*b + s)**2)/((b + s)**3)

        return 0.5 * np.sqrt(error)

    sigs_err = list(map(err, s_arr, b_arr, sigma_s_arr))
    #print(sigs_err)
    
    # Expected significance:
    plt.close()
    fig, ax = plt.subplots()
    plt.plot(efficiencies, sigs)
    plt.fill_between(efficiencies, sigs - sigs_err, sigs + sigs_err, alpha = 0.3)
    plt.axvline(efficiencies[sigs.argmax()], color ='r', ls = '--', ymax = sigs.max(), alpha = 0.8)
    plt.axhline(sigs.max(), color = 'r', alpha = 0.8)
    #plt.text(0.3, 3.5, 'Max significance: ' + str(sigs.max()) + '\nBDT Efficiency at max significance: ' + str(efficiencies[sigs.argmax()]))
    plt.title('Expected significance as a function of BDT efficiency')
    plt.xlabel('BDT efficiency')
    plt.ylabel('Significance * BDT efficiency')
    ax.minorticks_on()
    plt.savefig(filename_dict['analysis_path'] + 'results/significance_expected_plot.png', dpi = 300, facecolor = 'white')
    plt.close()

    counts = []
    bkgs = []
    sigma_s_arr = []
    masses = []
    masses_err = []
    mass_variances = []
    mass_variances_err = []

    ff = ROOT.TFile(filename_dict['analysis_path'] + 'results/mass_fit.root','recreate')

    for i in range(len(efficiencies)):
        selected_data_hdl = data.query('model_output > ' + str(scores[i]))
        #hist = plot_utils.plot_distr(selected_data_hndl, column='m', bins=34, colors='orange', density=False,fill=True, range=[2.96,3.04])
        hist = np.histogram(selected_data_hdl['m'], bins=36, range=(2.96,3.04))
        
        if flag_dict['subtract_bkg']:
            selected_bkg_hdl = bkg_hdl.query('model_output > ' + str(scores[i]))
            bkg_hist = np.histogram(selected_bkg_hdl['m'], bins=36, range=(2.96,3.04))
            count, bkg, sigma_s, fitted_mass, mass_variance = signal_fitter(hist=hist, bkg_hist=bkg_hist, efficiency=efficiencies[i], 
                                                                            significance=sigs[i], sig_err=sigs_err[i], only_gaus=flag_dict['only_gaus'])
        else:
            count, bkg, sigma_s, fitted_mass, mass_variance = signal_fitter(hist=hist, efficiency=efficiencies[i], 
                                                                            significance=sigs[i], sig_err=sigs_err[i])

        counts.append(count)
        bkgs.append(bkg)
        sigma_s_arr.append(sigma_s)
        masses.append(fitted_mass[0])
        masses_err.append(fitted_mass[1])
        mass_variances.append(mass_variance[0])
        mass_variances_err.append(mass_variance[1])

    ff.Close()

    masses = np.array(masses)
    masses_err = np.array(masses_err)
    mass_variances = np.array(mass_variances)
    mass_variances_err = np.array(mass_variances_err)


    idx_max = np.argwhere(efficiencies == 0.8)[0,0]


    # Signal/BDT_eff vs. BDT_eff:
    plt.close()
    efficiencies = np.array(efficiencies)
    counts = np.array(counts)
    fig, ax = plt.subplots()
    plt.plot(efficiencies[:idx_max], counts[:idx_max]/efficiencies[:idx_max])
    #plt.axvline(efficiencies[:idx_max][(count/efficiencies[:idx_max]).argmax()], color ='r', ls = '--', ymax = count.max(), alpha = 0.8)
    #plt.axhline((sigs/efficiencies[:idx_max]).max(), color = 'r', alpha = 0.8)
    plt.fill_between(efficiencies[:idx_max], np.maximum(np.zeros_like(counts[:idx_max]),(counts[:idx_max] - np.sqrt(counts[:idx_max]))/efficiencies[:idx_max]), 
                        (counts[:idx_max] + np.sqrt(counts[:idx_max]))/efficiencies[:idx_max], alpha = 0.3)
    plt.title('Efficiency-normalized Signal as a function of BDT efficiency')
    plt.xlabel('BDT efficiency')
    plt.ylabel(r'$\frac{S(3\sigma)}{BDT efficiency}$', fontsize=16)
    ax.minorticks_on()
    plt.tight_layout()
    #ax.yaxis.set_tick_params(which='minor', bottom=False)
    plt.savefig(filename_dict['analysis_path'] + 'results/signal_norm_plot.png', dpi=300, facecolor='white')
    plt.close()


    # Measured significance:
    plt.close()
    efficiencies = np.array(efficiencies)
    counts = np.array(counts)
    bkgs = np.array(bkgs)
    fig, ax = plt.subplots()
    sigs = counts/np.sqrt(counts + bkgs)
    sigs_err = list(map(err, counts, bkgs, np.array(sigma_s_arr)))
    plt.plot(efficiencies[:idx_max], sigs[:idx_max]*efficiencies[:idx_max])
    #plt.axvline(efficiencies[(sigs*efficiencies).argmax()], color ='r', ls = '--', ymax = sigs.max(), alpha = 0.8)
    #plt.axhline((sigs*efficiencies).max(), color = 'r', alpha = 0.8)
    plt.fill_between(efficiencies[:idx_max], np.maximum(np.zeros_like(sigs[:idx_max]), (sigs[:idx_max] - sigs_err[:idx_max])*efficiencies[:idx_max]), 
                                                            (sigs[:idx_max] + sigs_err[idx_max])*efficiencies[:idx_max], alpha = 0.3)
    plt.title('Measured Significance as a function of BDT efficiency')
    plt.xlabel('BDT efficiency')
    plt.ylabel('Significance*efficiency', fontsize=16)
    ax.minorticks_on()
    plt.tight_layout()
    #ax.yaxis.set_tick_params(which='minor', bottom=False)
    plt.savefig(filename_dict['analysis_path'] + 'results/significance_measured_plot.png', dpi=300, facecolor='white')
    plt.close()

    # If BDT eff is too high the fit does not perfom well
    max_idx = 4
    masses = np.array(masses[:-max_idx])
    masses_err = np.array(masses_err[:-max_idx])
    mass_variances = np.array(mass_variances[:-max_idx])
    mass_variances_err = np.array(mass_variances_err[:-max_idx])

    # Mass vs. BDT eff
    fig, ax = plt.subplots()
    plt.plot(efficiencies[:-max_idx], masses)
    #plt.fill_between(efficiencies[:-max_idx], masses - masses_err, masses + masses_err, alpha = 0.3)
    plt.title('Fitted mass as a function of BDT efficiency')
    plt.xlabel('BDT efficiency')
    plt.ylabel('Fitted mass')
    ax.minorticks_on()
    plt.tight_layout()
    plt.savefig(filename_dict['analysis_path'] + 'results/fitted_mass_vs_DBT_eff.png', dpi=300, facecolor='white')
    plt.close()

    # Mass variance vs. BDT eff
    fig, ax = plt.subplots()
    plt.plot(efficiencies[:-max_idx], mass_variances)
    #plt.fill_between(efficiencies[:-max_idx], mass_variances - mass_variances_err, mass_variances + mass_variances_err, alpha = 0.3)
    plt.title('Fitted mass variance as a function of BDT efficiency')
    plt.xlabel('BDT efficiency')
    plt.ylabel('Fitted mass variance')
    ax.minorticks_on()
    plt.tight_layout()
    plt.savefig(filename_dict['analysis_path'] + 'results/fitted_mass_variance_vs_BDT_eff.png', dpi=300, facecolor='white')
    plt.close()




#%%
