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
