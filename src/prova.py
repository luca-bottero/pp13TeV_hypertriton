import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import xgboost as xgb
import mass_fit
import hist
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import *
from hipe4ml import plot_utils
import matplotlib


data = pd.read_parquet('../data/data_scores.parquet.gzip') 

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.LoadMacro("cmb_fit_exp.C")
ROOT.gROOT.LoadMacro("cmb_fit_erf.C")
from ROOT import cmb_fit_exp, cmb_fit_erf