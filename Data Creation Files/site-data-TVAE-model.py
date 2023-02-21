import pandas as pd
import geopandas as gp

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

from sdv.tabular import TVAE

import preprocess_functions
from data_comparison_plots import plot_comparison_scatter, plot_comparison_hist
from postprocess_functions import sample_rand_radii, anonymize_spatial, generate_timestamp

from package_synth_data import initialize_data_package
breakpoint()

### ---------------------------------------Load site data to synethesize-----------------------------------------###
site_data_geo = gp.read_file("Original Data\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.gpkg")

# indicate whether to plot descriptive figs or not
plot_figs = 0

### -----------------------------------Preprocess data for sdv.TVAE model fit-------------------------------------###
# simplify to dataframe
site_data, metadata_site = preprocess_functions.preprocess_site_data(site_data_geo)

breakpoint()
### -----------------------------------Fit and save TVAE model for site data-------------------------------------###
N = 300
N2 = len(site_data['site_id'])
# set up TVAE, fit and save
model = TVAE(primary_key='site_id')
model.fit(site_data)

#model.save('site_data_synth_model.pkl')
# model = TVAE.load('site_data_synth_model.pkl')

### ----------------------------------------Sample data and test utility-----------------------------------------###
# create sample data
new_data_site_data = model.sample(num_rows=N)

# evaluate utility measures
evaluate(new_data_site_data, site_data)
# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, new_data_site_data)

breakpoint()
report = QualityReport()
report.generate(site_data, new_data_site_data, metadata_site)
report.get_details(property_name='Column Shapes')
report.get_details(property_name='Pair Trends')

breakpoint()
### ----------------Re-sample using conditional sampling to emulated site spatial clustering------------------###
N3 = 30
conditions = sample_rand_radii(new_data_site_data,10,N3)
sample_sites = model.sample_remaining_columns(conditions)

# evaluate Chi-squared and K-S score
evaluate(sample_sites, new_data_site_data)

# Ml ability to detec difference between (1 minus ROC AUC score for ML classifier)
LogisticDetection.compute(site_data, sample_sites)

### ---------------------------------------Plot data comparisons----------------------------------------------###
breakpoint()
if plot_figs == 1:
    fig1, axes = plot_comparison_scatter(sample_sites,new_data_site_data,site_data,'area','k')

    fig2, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'area')
    fig3, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'k')
    fig4, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'Reef')
    fig5, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'habitat')
    fig6, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'sitedepth')
    fig7, axes = plot_comparison_hist(sample_sites,new_data_site_data,site_data,'rubble')

### ------------------------------------------Save site data to csv-------------------------------------------###
time_stamp = generate_timestamp()
sample_sites.to_csv('Synthetic Data\\site_data_'+time_stamp+'_numsamps_'+str(N3)+'.csv')
sample_sites_anon = anonymize_spatial(sample_sites)

initialize_data_package(time_stamp+'_numsamps_'+str(N3))

sample_sites_anon.to_csv('Synthetic Data\\Site Data Packages\\'+time_stamp+'_numsamps_'+
                        str(N3)+'\\site_data\\site_data_anon_'+time_stamp+'_numsamps_'+str(N3)+'.csv')
