import pandas as pd
import geopandas as gp

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

from sdv.lite import TabularPreset

from preprocess_functions import preprocess_site_data, convert_to_csv
from data_comparison_plots import plot_comparison_scatter, plot_comparison_hist
from sampling_functions import sample_rand_radii
from postprocess_functions import anonymize_spatial, generate_timestamp
from package_synth_data import initialize_data_package,retrieve_orig_site_data_fp

### ---------------------------------------Key Inputs-----------------------------------------###
orig_data_package = "Moore_2022-11-17"

### ---------------------------------------Load site data to synethesize-----------------------------------------###

site_data_geo_fn = retrieve_orig_site_data_fp(orig_data_package)
site_data_geo = gp.read_file(site_data_geo_fn)

# convert to csv for use with connectivity model (package clash with GAN model packages)
convert_to_csv(site_data_geo,site_data_geo_fn)

# indicate whether to plot descriptive figs or not
plot_figs = 0

### -----------------------------------Preprocess data for sdv.TVAE model fit-------------------------------------###
# simplify to dataframe
site_data, metadata_site = preprocess_site_data(site_data_geo)

breakpoint()
### -----------------------------------Fit and save TVAE model for site data-------------------------------------###
N = 300
N2 = len(site_data['site_id'])
# set up TVAE, fit and save
model = TabularPreset(name='FAST_ML', metadata=metadata_site)
model.fit(site_data)

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
sample_sites['lat'] = -1*sample_sites['lat']
sample_sites.to_csv('Synthetic Data\\site_data_'+time_stamp+'_numsamps_'+str(N3)+'.csv')
sample_sites_anon = anonymize_spatial(sample_sites)

initialize_data_package(time_stamp+'_numsamps_'+str(N3))

sample_sites_anon.to_csv('Synthetic Data\\Synthetic Data Packages\\'+time_stamp+'_numsamps_'+str(N3)+'\\site_data\\site_data_anon_'+time_stamp+'_numsamps_'+str(N3)+'.csv')
