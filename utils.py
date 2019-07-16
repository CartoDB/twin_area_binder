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


import warnings
warnings.filterwarnings('ignore')

import ipywidgets as widgets
from ipywidgets import Output, Tab
import seaborn as sns
from IPython.display import clear_output



def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj]

def varname(p):
    import inspect, re
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def gpd_bbox(bounds):
    '''
    bounds (list): min_lng, min_lat, max_lng, max_lat
    '''
    from shapely.geometry import Polygon
    import geopandas as gpd
    import pandas as pd
    bbox = gpd.GeoDataFrame(pd.DataFrame([Polygon.from_bounds(*bounds)], columns=['geometry']))
    return bbox

def gpd_multiple_bbox(bounds_dict):
    '''
    bounds_dict (dict): {'tag': [min_lng, min_lat, max_lng, max_lat], ...}
    '''
    from shapely.geometry import Polygon, MultiPolygon
    import geopandas as gpd
    import pandas as pd
    bbox = pd.DataFrame()
    for tag, bounds in bounds_dict.items():
        if isinstance(bounds, Polygon) or isinstance(bounds, MultiPolygon):
            tmp_bbox = pd.DataFrame([(bounds, tag)], columns=['geometry', 'tag'])
            bbox = bbox.append(tmp_bbox)
        elif isinstance(bounds, list) and len(bounds)==4: 
            tmp_bbox = pd.DataFrame([(Polygon.from_bounds(*bounds), tag)], columns=['geometry', 'tag'])
            bbox = bbox.append(tmp_bbox)
        else:
            raise TypeError('''
                            bounds should be either 
                            shapely.Polygon (or shapely.MultiPolygon) 
                            or list (min_lng, min_lat, max_lng, max_lat)')
                            ''')
    bbox.reset_index(drop=True, inplace=True)
    return gpd.GeoDataFrame(bbox)

def tile_cover(geometry, z):
    import geopandas as gpd
    from supermercado import burntiles, super_utils
    from pygeotile.tile import Tile
    geo = gpd.GeoSeries([geometry]).__geo_interface__['features'][0]
    geo = [f for f in super_utils.filter_polygons([geo])]
    return [Tile.from_google(*geo).quad_tree for geo in [f for f in burntiles.burn(geo, z)]]

def quad_to_poly(quad):
    from shapely.geometry import Polygon
    from pygeotile.tile import Tile
    from pygeotile import tile
    bbox = tile.Tile.from_quad_tree(quad).bounds
    # min_lng, min_lat, max_lng, max_lat
    poly = Polygon.from_bounds(bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]) 
    return poly

def gpd_grid(geometry, z, tag=None):
    '''
    draw grid and return gpd.GeoDataFrame()
    '''
    from shapely.geometry import Polygon, MultiPolygon
    import pandas as pd
    import geopandas as gpd
    
    if isinstance(geometry, Polygon) or isinstance(geometry, MultiPolygon):
        qt_list = tile_cover(geometry, z)
    elif isinstance(geometry, list) and len(geometry)==4:
        qt_list = tile_cover(gpd_bbox(geometry).geometry.values[0], z)
    else:
        raise TypeError('''
        geometry should be either 
        shapely.Polygon (or shapely.MultiPolygon) 
        or list (min_lng, min_lat, max_lng, max_lat)')
        ''')
    grid = pd.DataFrame(qt_list, columns=['qt'])
    grid['geometry'] = grid['qt'].apply(lambda x: quad_to_poly(x))
    grid['tag'] = [tag] * len(grid)
    return gpd.GeoDataFrame(grid)

def geom2geometry(the_geom):
    return wkb.loads(the_geom, hex=True)

all_diff = lambda x: sum([np.abs(_[0]-_[1]) for _ in list(permutations(x, 2))])

# def display_side_by_side(*args):
#     from IPython.display import display_html
#     html_str=''
#     for df in args:
#         html_str+=df.to_html()
#     display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
def draw_boxplot_df(df, axes, fig, variables_names):
    _= df[variables_names].dropna().plot(kind='box',grid=True,
                                         color={'medians': 'orange', 'boxes': 'blue'},
                                         medianprops={'linestyle': '-', 'linewidth': 4},
                                         showfliers=False,patch_artist=True, ax = axes)
    _ = axes.set_xticklabels(axes.get_xticklabels(),rotation=90)
    _ = axes.set_facecolor('white')
    fig.suptitle("")
    fig.tight_layout()
    
def qt_convert(lat, lng, zoom=19):
    '''
    lat, lng, zoom=18(default)
    '''
    from pygeotile.tile import Tile
    try:
        return Tile.for_latitude_longitude(lat, lng, zoom=zoom).quad_tree
    except:
        pass
    
    

def calc_ens_pc(data, selected_col_prep, source:str, target:str, n_ens_member=10, seed=112358):
    data_prep = data.copy()
    data_prep = data_prep[data_prep.tag.isin([source, target])]
    ens_pca_complete = min(data_prep[selected_col_prep].shape[1], data_prep.shape[0])
    complete_explained_variance = compute_pca(data_prep, 
                                              selected_col_prep, 
                                              n_components=ens_pca_complete, 
                                              how = 'ppca', 
                                              use_corr = True, 
                                              what = False)[1]
    logging.info(complete_explained_variance)

    ens_var_max = 1
    ens_var_min = 0.9
    ens_pca_max = ens_pca_complete
    ens_pca_min = np.min([np.max([1, bisect.bisect(np.cumsum(complete_explained_variance), ens_var_min)+1]), data_prep.shape[0]])
    logging.info(f"[Ensemble Explained Variance] Max: {ens_var_max}; Min: {ens_var_min}")
    logging.info(f"[Ensemble n_component Config] Max: {ens_pca_max}; Min: {ens_pca_min}")

    ## n_ens_member = 10
    logging.info(f"[# Ensemble Members] {n_ens_member}")
    
    reconstructed_data = []
    ens_n_component = []
    ## explained_variance_ratio = {}

    if seed:
        random.seed(seed)
        
    for i in tqdm(range(n_ens_member)):
        data_pre = data.copy()
        data_prep = data_prep[data_prep.tag.isin([source, target])]
        n = np.random.randint(ens_pca_min, ens_pca_max)
        logging.info(n)
        #n = random.choices(list(range(ens_pca_min, ens_pca_max+1)), weights=np.cumsum(complete_explained_variance)[ens_pca_min-1:ens_pca_max])[0]
        
        returned_data = compute_pca(data_pre,
                                    selected_col_prep, 
                                    n_components = n, 
                                    how = 'ppca',
                                    use_corr = True,
                                    what = False)
        pc_columns = [_ for _ in returned_data[0].columns if _.startswith('pca')]
        logging.info(len(pc_columns))
        ens_n_component.append(len(pc_columns))
        reconstructed_data.append(returned_data)
        
        ## explained_variance_ratio[n] = explained_variance_ratio.get(n, sum(returned_data[1]))
    logging.info(f"[Ensemble n_component] {list(ens_n_component)}")
    return reconstructed_data




def distance_matrix(data, source_qtid:str, source:str, target:str, ens_iter=None, show_time=True):
    import time
    #import fastdist
    start = time.time()
    data_copy = data.copy()
    ## index_source_city = list(data['tag'] == source)
    index_source_city = list((data_copy['tag'] == source) & (data_copy['qt_id'] == source_qtid))
    index_target_city = list(data_copy['tag'] == target)
    # logging.info(f"[# Grid Cells] Source:{sum(index_source_city)}  Target:{sum(index_target_city)}  Total:{len(index_source_city)}")
    
    pc_columns = [_ for _ in data.columns if _.startswith('pca')]
    
    if ens_iter != None:
        pass
        logging.info(f"[Ensenble Member NO.{ens_iter}] n_component: {len(pc_columns)}")

    data_source_city = data_copy.loc[index_source_city, pc_columns].values
    data_target_city = data_copy.loc[index_target_city, pc_columns].values
    
    dist_matrix = pairwise_distances(data_source_city, data_target_city)
    if show_time:
        logging.info(f"[Time] {time.time()-start} seconds")
    return dist_matrix


def similarity_score(reconstructed_data, source_qtid, source='nyc', target='la'):
    import numpy as np
    from itertools import permutations
    all_diff = lambda x: sum([np.abs(_[0]-_[1]) for _ in list(permutations(x, 2))])
    
    index_source_city = list(reconstructed_data[0][0]['tag'] == source) & (reconstructed_data[0][0]['qt_id'] == source_qtid)
    index_target_city = list(reconstructed_data[0][0]['tag'] == target)
    n_ens_member = len(reconstructed_data)
    
    dist_ens = np.zeros((n_ens_member, sum(index_source_city), sum(index_target_city)))
    
    for i in range(n_ens_member):
        tmp_data = reconstructed_data[i][0]
        tmp_res = distance_matrix(tmp_data, source_qtid, source, target, ens_iter=i+1, show_time=False)
        dist_ens[i,:,:] = tmp_res
    
    if n_ens_member > 1:
        score = dist_ens[:,:,:].mean(axis=0) - np.apply_along_axis(all_diff, 0, dist_ens)/(2*n_ens_member*(n_ens_member-1))
        score_mean_each_source = dist_ens.mean(axis=2).mean(axis=0) - np.apply_along_axis(all_diff, 0, dist_ens.mean(axis=2))/(2*n_ens_member*(n_ens_member-1))
    elif n_ens_member == 1:
        score = dist_ens[:,:,:].mean(axis=0)
        score_mean_each_source = dist_ens.mean(axis=2).mean(axis=0)
    else:
        raise Exception('n_ens_member should be >= 1')
    similarity_skill_score = 1 - (score / score_mean_each_source[:,None])
    
    return pd.DataFrame(similarity_skill_score[0], index=reconstructed_data[0][0][index_target_city]['qt_id'], columns=['similarity_skill_score']).sort_values(by='similarity_skill_score', ascending=False).reset_index()
    
