import sys
import time
import plotly.graph_objects as go

sys.path.append("..")

from src.plotting.data_comparison_plots import create_timeseries, get_data_quantiles
from src.data_processing.package_synth_data import save_csv_plotting
from src.models.env_PAR_model import env_data_model

from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### ------------------ Use model to generate synthetic env data and sampled synthetic env data ----------------- ###

# layer = "dhw"
layer = "Ub"
rcp = "45"
root_original_file = "Moore_2022-11-17"
root_site_data_synth = "synth_2023-7-24_152038.csv"
nsamples = 10
# replicates = [10, 38, 41, 42, 45]  # dhw
replicates = [1, 5, 13, 16, 45]  # waves
tic = time.perf_counter()
(
    env_df,
    new_data_env,
    selected_env_data,
    metadata_env,
    nyears,
    old_years,
) = env_data_model(
    root_original_file, root_site_data_synth, nsamples, replicates, rcp, layer
)
toc = time.perf_counter()
print(f"Model learnt in {toc - tic:0.4f} seconds")
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# # evaluate and report values should be high for quality data
# # report values should be high
env_df["year"] = [yr.year for yr in env_df.year]
new_data_env["year"] = [int(yr) for yr in new_data_env.year]

metadata_env = SingleTableMetadata()
metadata_env.detect_from_dataframe(data=env_df[env_df.columns[[0, 1, 2, 3, 5]]])
quality_report = evaluate_quality(real_data=env_df[env_df.columns[[0, 1, 2, 3, 5]]],synthetic_data=new_data_env,metadata=metadata_env,)


### ------------------------------------------ Plot data as timeseries ----------------------------------------- ###

outcomes_data, outcomes_synth = get_data_quantiles(env_df, new_data_env, nyears, layer)
breakpoint()
fig_data = create_timeseries(outcomes_data, np.unique(old_years), layer, label="Original " + layer + " Data")
go.Figure(fig_data).show()

fig_synth = create_timeseries(outcomes_synth, np.unique(old_years), layer, label="Original " + layer + " Data")
go.Figure(fig_synth).show()

### ----------------------------------------------- Plot sampled data ------------------------------------------ ###
new_years = [str(int(yr)) for yr in selected_env_data["year"]]
outcomes_data, outcomes_synth_selected = get_data_quantiles(env_df, selected_env_data, nyears, layer)

fig_synth = create_timeseries(
    outcomes_synth_selected, np.unique(old_years), layerlabel="Synthetic sampled " + layer + " data"
)
go.Figure(fig_synth).show()

save_csv_plotting(env_df, new_data_env, selected_env_data, root_site_data_synth, layer)
