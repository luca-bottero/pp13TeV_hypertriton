# 3-body decay of Hypertriton in p-p collision at 13 TeV with ALICE data

## How to run the analysis

- Create a folder named 'data' with the trees in it
- 'cd src'
- Create or modify the '.yaml' file in the 'config' folder
- Modify 'analysis.py' such that 'config = [NAME_OF_CONFIG]'
- 'python3 analysis.py

**Directories:**

* data: contains root trees and pandas dataframe with model output
* training: contains the model and the plots generated in the training process (train-test output, feature importance)
* analysis: contains the results of analysis.py (images):
  * mass_sys: result of mass_fitter in mass_fit.py
  * m_mppi: scatter plot with hist of total mass vs. mass of p + pi
  * mppi_mdpi: Dalitz plot of p+pi vs. d+pi
  * old_results: contains plots generated in the first weeks
  * results: folder that still needs to be properly formatted
* src: contains all the code:
  * training.py: trains the model and applies it to the data
  * analysis.py: performs the analysis using already processed data
  * mass_fit.py: mass distribution utilities
  * utils.py: general utilities
  * pp_visualization.py: old code used to generate the plots in old_results dir
  * cmb_fit_exp.C: ROOT macro used to fit a gausn signal with a exp background simultaneously
  * cmb_fit_erf.C: ROOT macro used to fit a gausn signal with a erf background simultaneously
  
