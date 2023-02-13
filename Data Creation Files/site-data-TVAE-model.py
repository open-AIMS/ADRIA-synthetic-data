import pandas as pd
import geopandas as gp

from sdv.evaluation import evaluate
from sdv.metrics.tabular import BNLikelihood,LogisticDetection
from sdv.metrics.timeseries import LSTMDetection

from sdv.tabular import TVAE
from sdv.lite import TabularPreset

import preprocess_functions
from data_comparison_plots import plot_comparison_scatter, plot_comparison_hist
from sample_sites_functions import sample_rand_radii
breakpoint()

### ----------------Load site data to synethesize------------------###
data_set_folder = "Original Data"
site_data_geo = gp.read_file(
    data_set_folder+"\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.gpkg")

# indicate whether to plot descriptive figs or not
plot_figs = 0

### ----------------Preprocess data for sdv.TVAE model fit------------------###
# simplify to dataframe
site_data, metadata = preprocess_functions.preprocess_site_data(site_data_geo)

breakpoint()
### ----------------Fit and save TVAE model for site data------------------###
N = 300
N2 = len(site_data['site_id'])
# set up TVAE, fit and save
# model = TVAE(primary_key='site_id')
# model.fit(site_data)

model = TabularPreset(name='FAST_ML', metadata=metadata)
model.fit(site_data)
model.save('site_data_synth_model.pkl')
# model =TabularPreset.load('site_data_synth_model.pkl')
# model = TVAE.load('site_data_synth_model.pkl')

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
conditions = sample_rand_radii(new_data_site_data,10,N3)
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
    fig1, axes = plot_comparison_scatter(sample_sites,new_data_site_data,site_data,'area','k')

    fig2, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'area')
    fig3, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'k')
    fig4, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'Reef')
    fig5, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'habitat')
    fig6, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'sitedepth')
    fig7, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'rubble')

sample_sites.to_csv('site_data_'+data_set_folder +
                    '_numsamps_'+str(N3)+'.csv')
