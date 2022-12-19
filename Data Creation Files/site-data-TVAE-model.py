import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import random as rn
import math
from inspect import getsourcefile
from os.path import abspath

from sdv.timeseries import PAR
from sdv import Metadata

from scipy.io import loadmat
from sdv.evaluation import evaluate
from sdv.metrics.tabular import BNLikelihood
from sdv.metrics.tabular import LogisticDetection

from sdv.metrics.timeseries import LSTMDetection, TSFCDetection

from sdv.tabular import TVAE

breakpoint()
### ----------------Load site data to synethesize------------------###
data_set_folder = "Original Data"
base_folder = abspath(getsourcefile(lambda: 0))
site_data_geo = gp.read_file(
    base_folder[:-7]+data_set_folder + "/Moore_2022-11-17/site_data/Moore_2022-11-17.gpkg")

# indicate whether to plot descriptive figs or not
plot_figs = 0

### ----------------Preprocess data for sdv.TVAE model fit------------------###
# simplify to dataframe
site_data = site_data_geo[site_data_geo.columns[:-1]]
site_data["long"] = site_data_geo.centroid.x
site_data["lat"] = site_data_geo.centroid.y

# get rid of identifying ids
site_data = site_data.drop('reef_siteid', axis=1)
site_ids = pd.DataFrame(
    {'site_id': [i for i in range(1, len(site_data['site_id'])+1)]})
site_data = site_data.drop('site_id', axis=1)
site_data = pd.concat([site_ids, site_data], axis=1)

# change categorical to strings
for k in range(len(site_data['site_id'])):
    site_data['Reef'][k] = str(site_data['Reef'][k])
    site_data['habitat'][k] = str(site_data['habitat'][k])
    site_data['rubble'][k] = str(site_data['rubble'][k])

site_data['long'] = abs(site_data['long'])
site_data['lat'] = abs(site_data['lat'])
# define table metadata
metadata = {
    'fields': {
        'site_id': {'type': 'id', 'subtype': 'integer'},
        'habitat': {'type': 'categorical'},
        'area': {'type': 'numerical', 'subtype': 'float'},
        'benthic': {'type': 'categorical'},
        'rubble': {'type': 'categorical'},
        'k': {'type': 'numerical', 'subtype': 'float'},
        'sitedepth': {'type': 'numerical', 'subtype': 'float'},
        'lat': {'type': 'numerical', 'subtype': 'float'},
        'long': {'type': 'numerical', 'subtype': 'float'}
    },
    'constraints': [],
    'primary_key': 'site_id'
}

### ----------------Fit and save TVAE model for site data------------------###
N = 300
N2 = len(site_data['site_id'])
# set up TVAE, fit and save
model = TVAE(primary_key='site_id')
model.fit(site_data)
model.save('site_data_brick_model.pkl')
# model = TVAE.load('site_data_brick_model.pkl')

### ----------------Sample data and test utility------------------###
# create sample data
new_data_site_data = model.sample(num_rows=N)

# evaluate utility measures
# statistical - KS indicates 1-dist between distributions, CS indicates probability of synth and orig data
# being selected from the same distribution
evaluate(new_data_site_data, site_data, metrics=[
         'CSTest', 'KSTest'], aggregate=False)

# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, new_data_site_data)

breakpoint()
### ----------------Re-sample using conditional sampling to emulated site spatial clustering------------------###
N3 = 30
# randomly select 5 points in the synthetic data set
nrand_sites = 10
rand_sites = np.random.randint(
    0, len(new_data_site_data.lat), size=(1, nrand_sites))[0]
max_lat = max(new_data_site_data.lat)
min_lat = min(new_data_site_data.lat)
max_long = max(new_data_site_data.long)
min_long = min(new_data_site_data.long)
lats = new_data_site_data.lat[rand_sites]
longs = new_data_site_data.long[rand_sites]

rand_lats = []
rand_longs = []
R = 0.01
# generate random radii and theta around these points
rand_radii = np.sqrt(np.random.uniform(0, 1, size=(1, N3))[0])*R
rand_theta = 2*math.pi*np.random.uniform(0, 1, size=(1, N3))[0]

breakpoint()
for rr in range(len(rand_radii)):
    site = np.random.randint(0, nrand_sites-1, size=(1, 1))[0]
    rlat_temp = lats[rand_sites[site]] + \
        rand_radii[rr] * np.cos(rand_theta[rr])
    if rlat_temp[rand_sites[site][0]] < max_lat and rlat_temp[rand_sites[site][0]] > min_lat:
        rand_lats.append(rlat_temp[rand_sites[site][0]])
    rlong_temp = longs[rand_sites[site]] + \
        rand_radii[rr] * np.sin(rand_theta[rr])
    if rlong_temp[rand_sites[site][0]] < max_long and rlong_temp[rand_sites[site][0]] > min_long:
        rand_longs.append(rlong_temp[rand_sites[site][0]])

breakpoint()
nsites_samp = min(len(rand_lats), len(rand_longs))
conditions = pd.DataFrame(
    {'lat': rand_lats[0:nsites_samp], 'long': rand_longs[0:nsites_samp]})
breakpoint()
sample_sites = model.sample_remaining_columns(conditions)

# evaluate Chi-squared and K-S score
evaluate(sample_sites, new_data_site_data, metrics=[
         'CSTest', 'KSTest'], aggregate=False)

# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, sample_sites)

### ----------------Plot data comparisons------------------###
breakpoint()
N3 = sample_sites.shape[0]
if plot_figs == 1:
    fig1, axes = plt.subplots(1, 3)
    l1 = axes[0].scatter(new_data_site_data['lat'], new_data_site_data['long'],
                         s=new_data_site_data['k'], c=new_data_site_data['area'])
    l2 = axes[1].scatter(site_data['lat'], -site_data['long'],
                         s=site_data['k'], c=site_data['area'])
    l3 = axes[2].scatter(sample_sites['lat'], -sample_sites['long'],
                         s=sample_sites['k'], c=sample_sites['area'])
    axes[1].set_title('Original')
    axes[0].set_title('Synthetic')
    axes[2].set_title('Sampled')
    axes[1].set(xlabel='lat', ylabel='long')
    axes[0].set(xlabel='lat', ylabel='long')
    axes[2].set(xlabel='lat', ylabel='long')
    fig1.show()

    fig2, axes = plt.subplots(1, 3)
    axes[0].hist(new_data_site_data['area'], bins=round(np.sqrt(N)))
    axes[1].hist(site_data['area'], bins=round(np.sqrt(N2)))
    axes[2].hist(sample_sites['area'], bins=round(np.sqrt(N3)))
    axes[1].set_title('Original')
    axes[0].set_title('Synthetic')
    axes[2].set_title('Sampled')
    axes[1].set(xlabel='site area', ylabel='counts')
    axes[0].set(xlabel='site_area', ylabel='counts')
    axes[2].set(xlabel='site_area', ylabel='counts')
    fig2.show()

    fig3, axes = plt.subplots(1, 3)
    axes[0].hist(new_data_site_data['k'], bins=round(np.sqrt(N)))
    axes[1].hist(site_data['k'], bins=round(np.sqrt(N2)))
    axes[2].hist(sample_sites['k'], bins=round(np.sqrt(N3)))
    axes[1].set_title('Original')
    axes[0].set_title('Synthetic')
    axes[2].set_title('Sampled')
    axes[2].set(xlabel='k', ylabel='counts')
    axes[1].set(xlabel='k', ylabel='counts')
    axes[0].set(xlabel='k', ylabel='counts')
    fig3.show()

    fig4, axes = plt.subplots(1, 3)
    axes[0].hist(new_data_site_data['Reef'])
    axes[1].hist(site_data['Reef'])
    axes[2].hist(sample_sites['Reef'])
    axes[1].set_title('Original')
    axes[2].set_title('Sampled')
    axes[0].set_title('Synthetic')
    axes[1].set(xlabel='Reef', ylabel='counts')
    axes[2].set(xlabel='Reef', ylabel='counts')
    axes[0].set(xlabel='Reef', ylabel='counts')
    fig4.show()

    fig5, axes = plt.subplots(1, 3)
    axes[0].hist(new_data_site_data['habitat'])
    axes[1].hist(site_data['habitat'])
    axes[2].hist(sample_sites['habitat'])
    axes[1].set_title('Original')
    axes[0].set_title('Synthetic')
    axes[2].set_title('Sampled')
    axes[1].set(xlabel='habitat', ylabel='counts')
    axes[2].set(xlabel='habitat', ylabel='counts')
    axes[0].set(xlabel='habitat', ylabel='counts')
    fig5.show()

    fig6, axes = plt.subplots(1, 3)
    axes[0].hist(new_data_site_data['sitedepth'])
    axes[1].hist(site_data['sitedepth'])
    axes[2].hist(sample_sites['sitedepth'])
    axes[1].set_title('Original')
    axes[2].set_title('Sampled')
    axes[0].set_title('Synthetic')
    axes[1].set(xlabel='site_depth', ylabel='counts')
    axes[2].set(xlabel='site_depth', ylabel='counts')
    axes[0].set(xlabel='site_depth', ylabel='counts')
    fig6.show()

    fig7, axes = plt.subplots(1, 3)
    axes[0].hist(new_data_site_data['rubble'])
    axes[1].hist(site_data['rubble'])
    axes[2].hist(sample_sites['rubble'])
    axes[1].set_title('Original')
    axes[2].set_title('Sampled')
    axes[0].set_title('Synthetic')
    axes[1].set(xlabel='rubble', ylabel='counts')
    axes[0].set(xlabel='rubble', ylabel='counts')
    axes[2].set(xlabel='rubble', ylabel='counts')
    fig7.show()

sample_sites.to_csv('site_data_'+data_set_folder +
                    '_numsamps_'+str(nsites_samp)+'.csv')
