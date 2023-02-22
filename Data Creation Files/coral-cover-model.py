import pandas as pd
import geopandas as gp
import netCDF4

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdmetrics.reports.single_table import QualityReport

from sdv.lite import TabularPreset
from sdv.sampling import Condition
from sdv.constraints import Unique, FixedCombinations

from preprocess_functions import preprocess_cover_data
from data_comparison_plots import plot_comparison_scatter, plot_comparison_hist

### ------------------------------------------------Key Inputs---------------------------------------------------###
root_original_file = 'Moore_2022-11-17'

###---------------------------------------Load site data to synethesize------------------------------------------###

root_site_data_synth = 'site_data_22-2-2023_92546_numsamps_30.csv'
original_cover_data_fn = 'Original Data\\'+root_original_file+'\\site_data\\coral_cover.nc'
original_site_data_fn = 'Original Data\\'+root_original_file+'\\site_data\\'+root_original_file+".csv"
synth_site_data_fn = "Synthetic Data\\"+root_site_data_synth

breakpoint()
cover_orig = netCDF4.Dataset(original_cover_data_fn, 'r')
site_data_orig = pd.read_csv(original_site_data_fn)
site_data_synth = pd.read_csv(synth_site_data_fn)

###----------------------------------Preprocess data for sdv.fastML model fit------------------------------------###
# simplify to dataframe
breakpoint()
cover_df, metadata_cover = preprocess_cover_data(cover_orig,site_data_orig)

###----------------------------------Fit and save fastML model for site data-------------------------------------###
model = TabularPreset(name='FAST_ML', metadata=metadata_cover)
model.fit(cover_df)
synth_cover = model.sample(num_rows=300)
breakpoint()
###----------Sample conditional dist, based on synthetic lats and longs and requirement of species types---------###
conditions = pd.DataFrame({"species": cover_df["species"]})
model.sample_remaining_columns(conditions)
