import ROOT
import numpy as np
import aghast
 
 
# definition of shared parameter background function
iparB = np.array([3,4,5,6], dtype=np.int32)  # exp amplitude in B histo and exp common parameter
 
# signal + background function
iparSB = np.array(
    [
        0,  # Gaussian amplitude
        1,  # Gaussian mean
        2,  # Gaussian sigma
        3,  # Sigmoid normalization
        4,  # Sigmoid slope
        5,  # Sigmoid center
        6,  # Constant offset
    ],
    dtype=np.int32,
)
 
# Create the GlobalCHi2 structure
 
class GlobalChi2(object):
    def __init__(self, f1, f2):
        self._f1 = f1
        self._f2 = f2
 
    def __call__(self, par):
        # parameter vector is first background (in common 1 and 2) and then is
        # signal (only in 2)
 
        # the zero-copy way to get a numpy array from a double *
        par_arr = np.frombuffer(par, dtype=np.float64, count=6)
 
        p1 = par_arr[iparB]
        p2 = par_arr[iparSB]
 
        return self._f1(p1) + self._f2(p2)
 
def cmb_fit_sigmoid(bkg_hist, data_hist, efficiency):

    aghast_hist = aghast.from_numpy(data_hist)
    hSB = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    aghast_hist = aghast.from_numpy(bkg_hist)
    hB = aghast.to_root(aghast_hist,'Efficiency ' + str(np.round(efficiency,4)))

    fB = ROOT.TF1('fB','[0]/(1 + TMath::Exp(-[1]*(x-[2]))) + [3]',2.96,3.04)
    fSB = ROOT.TF1('fSB','gaus(0) + [3]/(1 + TMath::Exp(-[4]*(x-[5]))) + [6]',2.96,3.04)

    fB.SetParLimits(3, 0., 5.)
    fB.SetParameter(0, 15)
    fB.SetParLimits(2, 2.99, 3.0)
    fB.SetParLimits(1, 10., 500.)
    #bkg.SetParLimits(3, 2.)

    fSB.SetParLimits(6, 0., 5.)
    fSB.SetParameter(3, 15)
    fSB.SetParLimits(5, 2.99, 3.0)
    fSB.SetParLimits(4, 10., 500.)
    
    wfB = ROOT.Math.WrappedMultiTF1(fB, 1)
    wfSB = ROOT.Math.WrappedMultiTF1(fSB, 1)
    
    opt = ROOT.Fit.DataOptions()
    rangeB = ROOT.Fit.DataRange()
    # set the data range
    rangeB.SetRange(10, 90)
    dataB = ROOT.Fit.BinData(opt, rangeB)
    ROOT.Fit.FillData(dataB, hB)
    
    rangeSB = ROOT.Fit.DataRange()
    rangeSB.SetRange(10, 50)
    dataSB = ROOT.Fit.BinData(opt, rangeSB)
    ROOT.Fit.FillData(dataSB, hSB)
    
    chi2_B = ROOT.Fit.Chi2Function(dataB, wfB)
    chi2_SB = ROOT.Fit.Chi2Function(dataSB, wfSB)
    
    globalChi2 = GlobalChi2(chi2_B, chi2_SB)
    
    fitter = ROOT.Fit.Fitter()
    
    '''Npar = 6
    par0 = np.array([5, 5, -0.1, 100, 30, 10])
    
    # create before the parameter settings in order to fix or set range on them
    fitter.Config().SetParamsSettings(6, par0)
    # fix 5-th parameter
    fitter.Config().ParSettings(4).Fix()
    # set limits on the third and 4-th parameter
    fitter.Config().ParSettings(2).SetLimits(-10, -1.0e-4)
    fitter.Config().ParSettings(3).SetLimits(0, 10000)
    fitter.Config().ParSettings(3).SetStepSize(5)
    
    fitter.Config().MinimizerOptions().SetPrintLevel(0)
    fitter.Config().SetMinimizer("Minuit2", "Migrad")'''
    
    # we can't pass the Python object globalChi2 directly to FitFCN.
    # It needs to be wrapped in a ROOT::Math::Functor.
    globalChi2Functor = ROOT.Math.Functor(globalChi2, 7)
    
    # fit FCN function
    # (specify optionally data size and flag to indicate that is a chi2 fit)
    fitter.FitFCN(globalChi2Functor, 0, dataB.Size() + dataSB.Size(), True)
    result = fitter.Result()
    result.Print(ROOT.std.cout)
    
    c1 = ROOT.TCanvas()
    c1.SetName('eff_' + str(efficiency))
    c1.Divide(1, 2)
    c1.cd(1)
    ROOT.gStyle.SetOptFit(1111)
    
    fB.SetFitResult(result, iparB)
    fB.SetRange(rangeB().first, rangeB().second)
    fB.SetLineColor(ROOT.kBlue)
    hB.GetListOfFunctions().Add(fB)
    hB.Draw()
    
    c1.cd(2)
    fSB.SetFitResult(result, iparSB)
    fSB.SetRange(rangeSB().first, rangeSB().second)
    fSB.SetLineColor(ROOT.kRed)
    hSB.GetListOfFunctions().Add(fSB)
    hSB.Draw()

    c1.SaveAs("combinedFit.png")

    fitted_mass = fSB.GetParameters()[1]
    fitted_mass_err = fSB.GetParError(1)
    mass_variance = fSB.GetParameters()[2]
    mass_variance_err = fSB.GetParError(2)

    m = 2.9912
    d_m = 3 * 0.002

    #count = gaus.Integral(m - d_m, m + d_m) / root_hist.GetBinWidth(1)
    b = fB.Integral(m - d_m, m + d_m) / data_hist.GetBinWidth(1)