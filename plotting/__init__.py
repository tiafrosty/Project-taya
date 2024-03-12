import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
#from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.pipeline import make_pipeline
# for CV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# my modules
from .get_all_roc import get_all_roc 
from .make_plots_compared import make_plots_compared
from .get_roc_and_ci import get_roc_and_ci, plot_for_every_model