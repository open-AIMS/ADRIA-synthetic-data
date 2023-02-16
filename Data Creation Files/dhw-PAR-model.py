import pandas as pd
import geopandas as gp

import plotly.graph_objects as go

from deepecho import PARModel

import netCDF4
from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdv.metrics.timeseries import LSTMDetection
from sdmetrics.reports.single_table import QualityReport

import preprocess_functions
from data_comparison_plots import create_timeseries, get_data_quantiles
from sample_sites_functions import find_NN_dhw_data

### ----------------------------------Load site data and dhw data to synethesize--------------------------------###
data_set_folder = "Original Data"
synth_data_set_folder = "Synthetic Data"
DHW_45 = netCDF4.Dataset(
    data_set_folder+"\\Moore_2022-11-17\\DHWs\\dhwRCP45.nc", 'r')

site_data_synth = pd.read_csv(
    synth_data_set_folder+"\\site_data_Original Data_numsamps_29.csv")

### --------------------------------------Reshape data to fit PAR model------------------------------------------###
dhw_45_df, data_types, metadata_dhw, old_years, nyears = preprocess_functions.preprocess_dhw_data(DHW_45)

### ----------------------------------------Set up and fit PAR model---------------------------------------------###
model = PARModel(epochs=1024, cuda=False)

model.fit(data=dhw_45_df,
          context_columns=['Lat', 'Long'],
          entity_columns=['Site'],
          data_types=data_types,
          sequence_index='Year')

### -----------------------------------------Sample data to synthesize-------------------------------------------###
N_s = 200
new_data_dhw = model.sample(N_s)
new_years = [str(yr+2025) for yr in range(nyears)]
new_years = new_years*N_s
new_data_dhw['Year'] = new_years

### ----------------------------------------------Evaluate data --------------------------------------------------###

# create metadata dictionary
metadata_dhw = {'fields': {'Year': {'type': 'datetime'},
                            'Lat': {'type': 'numerical', 'subtype': 'float'},
                            'Long': {'type': 'numerical', 'subtype': 'float'},
                            'Site': {'type': 'categorical'},
                            'Dhw': {'type': 'numerical', 'subtype': 'float'}},
                 'context_columns': ['Site'],
                 'entity_columns': ['Lat', 'Long'],
                 'sequence_index': 'Year'}

# detection values should be low
LogisticDetection.compute(dhw_45_df, new_data_dhw)
LSTMDetection.compute(dhw_45_df, new_data_dhw, metadata_dhw)

# evaluate and report values should be high for quality data
evaluate(new_data_dhw, dhw_45_df)
report = QualityReport()
report.generate(dhw_45_df, new_data_dhw, metadata_dhw )
report.get_details(property_name='Column Shapes')

### ----------------------------------------------Plot data --------------------------------------------------###

outcomes_data,outcomes_synth = get_data_quantiles(dhw_45_df,new_data_dhw,nyears,old_years,new_years)

fig_data = create_timeseries(outcomes_data, label="Original DHW Data")
fig_data = go.Figure(fig_data)
fig_data.show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic DHW Data")
fig_synth = go.Figure(fig_synth)
fig_synth.show()

### --------------------------Find nearest negihbours to synthetic site data------------------------------------###
selected_dhws = find_NN_dhw_data(site_data_synth,new_data_dhw,nyears)

