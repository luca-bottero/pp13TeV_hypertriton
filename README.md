# 3-body decay of Hypertriton in p-p collision at 13 TeV with ALICE data

## How to run the analysis

```
cd pp13Tev_hypertriton
mkdir data
```
- Move your ROOT trees into ```data``` folder
- Create a ```.yaml``` configuration file using one those present in the ```src/config``` folder
- Modify ```src/analysis.py``` such that ```config = [name of your config file]```
- Run the analysis using ```python3 analysis.py```

**Analysis results:**

The results of the analysis are stored in the ```analysis_results``` folder. An example of the ouput is the following:

```
TOF_PID_cut
    ├── images
    │   ├── presel_eff
    │   ├── scatter
    │   ├── training
    │   └── var_distribution
    │       ├── data_bckg
    │       └── signal_bckg
    ├── model
    ├── output_data
    └── results
```

**Source Code**

The code is organized as follows:

```
src
    ├── analysis.py
    ├── cmb_fit_erf.C
    ├── cmb_fit_exp.C
    ├── config
    │   ├── MIX_data_LS.yaml
    │   ├── NO_TOF_PID_cut.yaml
    │   ├── TOF_PID_cut.yaml
    │   └── train_on_sidebands
    │       ├── NO_TOF_PID_cut_SB.yaml
    │       └── TOF_PID_cut_SB.yaml
    ├── mass_fit.py
    ├── pp_visualization.py
    ├── prova.py
    ├── __pycache__
    │   ├── mass_fit.cpython-38.pyc
    │   ├── training_ml.cpython-38.pyc
    │   └── utils.cpython-38.pyc
    ├── training_ml.py
    └── utils.py

```
