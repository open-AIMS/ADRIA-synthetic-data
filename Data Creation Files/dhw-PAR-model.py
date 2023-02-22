import pandas as pd

import plotly.graph_objects as go

from deepecho import PARModel

import netCDF4
from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdv.metrics.timeseries import LSTMDetection
from sdmetrics.reports.single_table import QualityReport

from preprocess_functions import preprocess_env_data
from data_comparison_plots import create_timeseries, get_data_quantiles
from postprocess_functions import sample_env_ensemble, create_env_nc

### ---------------------------------------------Key inputs ------------------------------------------------------###

layer = 'Ub'
rcp = '45'
root_original_file = 'Moore_2022-11-17'
root_site_data_synth = 'site_data_22-2-2023_92546_numsamps_30.csv'

### ----------------------------------Load site data and env data to synethesize----------------------------------###


if layer=='dhw':
    file_loc = "\\DHWs\\"
    file = 'dhwRCP'+rcp+'.nc'
elif layer=='Ub':
    file_loc = "\\waves\\"
    file = 'wave_RCP'+rcp+'.nc'
else:
    ValueError("Unrecognised environmental data layer.")

original_data_fn = "Original Data\\"+root_original_file+file_loc+file
synth_data_fn = "Synthetic Data\\"+root_site_data_synth
ENV = netCDF4.Dataset(original_data_fn, 'r')

site_data_synth = pd.read_csv(synth_data_fn)
breakpoint()
### --------------------------------------Reshape data to fit PAR model------------------------------------------###
env_df, data_types, metadata_env, old_years, nyears = preprocess_env_data(ENV,layer)

### ----------------------------------------Set up and fit PAR model---------------------------------------------###
model = PARModel(epochs=1024, cuda=False)

model.fit(data=env_df,context_columns=['lat', 'long'],entity_columns=['site'],data_types=data_types,sequence_index='year')

### -----------------------------------------Sample data to synthesize--------------------------------------------###
N_s = 200
new_data_env = model.sample(N_s)
new_years = [str(yr+2025) for yr in range(nyears)]
new_years = new_years*N_s
new_data_env['year'] = new_years

### ----------------------------------------------Evaluate data --------------------------------------------------###

# detection values should be low
LogisticDetection.compute(env_df, new_data_env)
LSTMDetection.compute(env_df, new_data_env, metadata_env)

# evaluate and report values should be high for quality data
# report values should be high
evaluate(new_data_env, env_df)
report = QualityReport()
report.generate(env_df, new_data_env, metadata_env)
report.get_details(property_name='Column Shapes')

### -------------------------------------------Plot data as timeseries--------------------------------------------###
outcomes_data,outcomes_synth = get_data_quantiles(env_df,new_data_env,nyears,old_years,new_years)

fig_data = create_timeseries(outcomes_data, label="Original "+layer+" Data")
fig_data = go.Figure(fig_data)
fig_data.show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic "+layer+" Data")
fig_synth = go.Figure(fig_synth)
fig_synth.show()
breakpoint()

### -------------------------Sample synthetic env data at synthetic site data locations --------------------------###
lat =-1.*site_data_synth.lat.values
long = site_data_synth.long.values
data = {'lat':lat,'long':long}
context = pd.DataFrame(data)
selected_env_data = model.sample(context=context)
selected_years = [str(yr+2025) for yr in range(nyears)]
selected_years = selected_years*len(site_data_synth.lat.values)
selected_env_data.insert(4,"year",selected_years,True)

outcomes_data,outcomes_synth_selected = get_data_quantiles(env_df,selected_env_data,nyears,old_years,selected_years)

fig_synth = create_timeseries(outcomes_synth_selected, label="Synthetic sampled "+layer+" data")
fig_synth = go.Figure(fig_synth)
fig_synth.show()

selected_env_ensemble = sample_env_ensemble(model,context)

synth_env_fn = "Synthetic Data\\Synthetic Data Packages\\"+synth_data_fn.split("\\")[1][9:-3]
+file_loc+layer+"_"+synth_data_fn.split("\\")[1][9:-3]+"nc"

create_env_nc(selected_env_ensemble,site_data_synth.lat.values,site_data_synth.long.values,site_data_synth.site_ids,synth_env_fn)
