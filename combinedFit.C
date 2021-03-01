#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "TH1.h"
#include "TList.h"
#include "Math/WrappedMultiTF1.h"
#include "HFitInterface.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <string>

using std::string;

// definition of shared parameter
// background function
int iparB[2] = { 0,      // exp amplitude in B histo
                 2    // exp common parameter
};
// signal + background function
int iparSB[5] = { 1, // exp amplitude in S+B histo
                  2, // exp common parameter
                  3, // gaussian amplitude
                  4, // gaussian mean
                  5  // gaussian sigma
};

// Create the GlobalCHi2 structure
struct GlobalChi2 {
   GlobalChi2(  ROOT::Math::IMultiGenFunction & f1,
                ROOT::Math::IMultiGenFunction & f2) :
      fChi2_1(&f1), fChi2_2(&f2) {}
   // parameter vector is first background (in common 1 and 2)
   // and then is signal (only in 2)
   double operator() (const double *par) const {
      double p1[2];
      for (int i = 0; i < 2; ++i) p1[i] = par[iparB[i] ];
      double p2[5];
      for (int i = 0; i < 5; ++i) p2[i] = par[iparSB[i] ];
      return (*fChi2_1)(p1) + (*fChi2_2)(p2);
   }

   const  ROOT::Math::IMultiGenFunction * fChi2_1;
   const  ROOT::Math::IMultiGenFunction * fChi2_2;
};

void combinedFit(TH1D * hB, TH1D * hSB, string name) {
 
   TF1 * fB = new TF1("fB","expo",2.96,3.04);
   fB->SetParameters(1,-0.05);
 
   TF1 * fS = new TF1("fS","gausn",2.96,3.04);
   fS->SetParameters(1,30,5);
 
   // perform now global fit
 
   TF1 * fSB = new TF1("fSB","expo + gausn(2)",2.96,3.04); 
 
   ROOT::Math::WrappedMultiTF1 wfB(*fB,1);
   ROOT::Math::WrappedMultiTF1 wfSB(*fSB,1);
 
   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange rangeB;
   // set the data range
   rangeB.SetRange(2.96,3.04);
   ROOT::Fit::BinData dataB(opt,rangeB);
   ROOT::Fit::FillData(dataB, hB);
 
   ROOT::Fit::DataRange rangeSB;
   rangeSB.SetRange(2.96,3.04);
   ROOT::Fit::BinData dataSB(opt,rangeSB);
   ROOT::Fit::FillData(dataSB, hSB);
 
   ROOT::Fit::Chi2Function chi2_B(dataB, wfB);
   ROOT::Fit::Chi2Function chi2_SB(dataSB, wfSB);
 
   GlobalChi2 globalChi2(chi2_B, chi2_SB);
 
   ROOT::Fit::Fitter fitter;
 
   const int Npar = 6;
   double par0[Npar] = { 5, 5, 0.1, 20, 2.991, 0.0032};
 
   // create before the parameter settings in order to fix or set range on them
   fitter.Config().SetParamsSettings(6,par0);
   // set limits on the third and 4-th parameter
   fitter.Config().ParSettings(2).SetLimits(-10,10);
   fitter.Config().ParSettings(3).SetLimits(0,100);
   fitter.Config().ParSettings(4).SetLimits(2.991 - 0.0032, 2.991 + 0.0032);
   fitter.Config().ParSettings(5).SetLimits(0.001, 0.01);
 
   fitter.Config().MinimizerOptions().SetPrintLevel(0);
   fitter.Config().SetMinimizer("Minuit2","Migrad");
 
   // fit FCN function directly
   // (specify optionally data size and flag to indicate that is a chi2 fit)
   fitter.FitFCN(6,globalChi2,0,dataB.Size()+dataSB.Size(),true);
   ROOT::Fit::FitResult result = fitter.Result();
   result.Print(std::cout);
 
   TCanvas * c1 = new TCanvas();
   c1->SetName(name.data());
   c1->Divide(1,2);
   c1->cd(1);
   gStyle->SetOptFit(1111);
 
   fB->SetFitResult( result, iparB);
   fB->SetRange(rangeB().first, rangeB().second);
   fB->SetLineColor(kBlue);
   hB->GetListOfFunctions()->Add(fB);
   hB->Draw("PE");
 
   c1->cd(2);
   fSB->SetFitResult( result, iparSB);
   fSB->SetRange(rangeSB().first, rangeSB().second);
   fSB->SetLineColor(kRed);
   hSB->GetListOfFunctions()->Add(fSB);
   hSB->Draw("PE");

   c1->Write();
}