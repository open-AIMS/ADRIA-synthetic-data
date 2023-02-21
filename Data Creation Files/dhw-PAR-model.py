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
from postprocess_functions import sample_dhw_ensemble, create_dhw_nc

### ----------------------------------Load site data and dhw data to synethesize--------------------------------###
original_data_fn = "Original Data\\Moore_2022-11-17\\DHWs\\dhwRCP45.nc"
synth_data_fn = "Synthetic Data\\site_data_Original Data_numsamps_29.csv"
DHW_45 = netCDF4.Dataset(original_data_fn, 'r')
site_data_synth = pd.read_csv(synth_data_fn)

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

# detection values should be low
LogisticDetection.compute(dhw_45_df, new_data_dhw)
LSTMDetection.compute(dhw_45_df, new_data_dhw, metadata_dhw)

# evaluate and report values should be high for quality data
evaluate(new_data_dhw, dhw_45_df)
report = QualityReport()
report.generate(dhw_45_df, new_data_dhw, metadata_dhw )
report.get_details(property_name='Column Shapes')

### ----------------------------------------------Plot data --------------------------------------------------###
breakpoint()
outcomes_data,outcomes_synth = get_data_quantiles(dhw_45_df,new_data_dhw,nyears,old_years,new_years)

fig_data = create_timeseries(outcomes_data, label="Original DHW Data")
fig_data = go.Figure(fig_data)
fig_data.show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic DHW Data")
fig_synth = go.Figure(fig_synth)
fig_synth.show()
breakpoint()
### -------------------------Sample synthetic dhws at synthetic site data locations -----------------------------###
Lat =-1.*site_data_synth.lat.values
Long = site_data_synth.long.values
data = {'Lat':Lat,'Long':Long}
context = pd.DataFrame(data)
selected_dhw_data = model.sample(context=context)
selected_years = [str(yr+2025) for yr in range(nyears)]
selected_years = selected_years*len(site_data_synth.lat.values)
selected_dhw_data.insert(4,"Year",selected_years,True)

outcomes_data,outcomes_synth_selected = get_data_quantiles(dhw_45_df,selected_dhw_data,nyears,old_years,selected_years)

fig_synth = create_timeseries(outcomes_synth_selected, label="Synthetic sampled DHW Data")
fig_synth = go.Figure(fig_synth)
fig_synth.show()

selected_dhw_ensemble = sample_dhw_ensemble(model,context)
synth_dhw_fn = "Synthetic Data\\Synthetic Data Packages\\"+synth_data_fn.split("\\")[1][9:-3]
+"\\DHWs\\dhw_data"+synth_data_fn.split("\\")[1][9:-3]+"nc"

create_dhw_nc(selected_dhw_ensemble,site_data_synth.lat.values,site_data_synth.long.values,site_data_synth.site_ids,synth_dhw_fn)
