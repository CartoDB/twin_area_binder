import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

import cartoframes
from cartoframes import *
from cartoframes import CartoContext, Layer, QueryLayer
from cartoframes.data import Dataset, DatasetInfo
from cartoframes.viz import *
from cartoframes.viz.helpers import *
from cartoframes.auth import set_default_context

cc = CartoContext(base_url='https://dfan.carto.com', api_key='default_public')
# dj=CartoContext()
set_default_context(context=cc)

import numpy as np
import pandas as pd
pd.set_option('max_colwidth', -1)
from shapely.geometry import Point, Polygon
from shapely import wkt, wkb
import geopandas as gpd

import os
import pickle
from time import sleep, time
import random
from itertools import permutations
import bisect
from tqdm import tqdm
from functools import partial


import missingno as msno
from sklearn.metrics import pairwise_distances, euclidean_distances
#import fastdist
from twin import compute_pca

from utils import *

import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from ipywidgets import Output, Tab
import seaborn as sns
from IPython.display import clear_output

from operator import itemgetter
import pysal
# print(pysal.__version__)