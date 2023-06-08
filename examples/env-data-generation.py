import sys
import plotly.graph_objects as go

sys.path.append("..")

from src.plotting.data_comparison_plots import create_timeseries, get_data_quantiles
from src.data_processing.package_synth_data import save_csv_plotting
from src.models.env_PAR_model import env_data_model

from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection
from sdv.metrics.timeseries import LSTMDetection
from sdmetrics.reports.single_table import QualityReport

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### ------------------ Use model to generate synthetic env data and sampled synthetic env data ----------------- ###

layer = "dhw"
# layer = "Ub"
rcp = "45"
root_original_file = "Moore_2022-11-17"
root_site_data_synth = "site_data_synth_6-6-2023_144117.csv"
nsamples = 50

(
    env_df,
    new_data_env,
    selected_env_data,
    metadata_env,
    nyears,
    old_years,
) = env_data_model(root_original_file, root_site_data_synth, nsamples, rcp, layer)
breakpoint()
### --------------- Evaluate synthetic data and sampled data utility and generate quality report --------------- ###

# detection values should be low
# LogisticDetection.compute(env_df, new_data_env)
# LSTMDetection.compute(env_df, new_data_env, metadata_env)

# # evaluate and report values should be high for quality data
# # report values should be high
evaluate(new_data_env, env_df[env_df.columns[[0, 1, 2, 3, 5]]])
env_df["year"] = [yr.year for yr in env_df.year]
new_data_env["year"] = [int(yr) for yr in new_data_env.year]
report = QualityReport()
report.generate(
    env_df[env_df.columns[[0, 1, 2, 3, 5]]],
    new_data_env,
    metadata_env,
)
report.get_details(property_name="Column Shapes")

### ------------------------------------------ Plot data as timeseries ----------------------------------------- ###
outcomes_data, outcomes_synth = get_data_quantiles(
    env_df, new_data_env, nyears, old_years, new_data_env["year"], layer
)

fig_data = create_timeseries(outcomes_data, label="Original " + layer + " Data")
go.Figure(fig_data).show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic " + layer + " Data")
go.Figure(fig_synth).show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
years = [yr.year for yr in env_df.year]
ax.bar3d(years, env_df.site, np.zeros(len(env_df.Ub)), 1, 1, env_df.Ub)
ax.set_ylim([0, 215])
plt.show()
### ----------------------------------------------- Plot sampled data ------------------------------------------ ###

outcomes_data, outcomes_synth_selected = get_data_quantiles(
    env_df, selected_env_data, nyears, old_years, selected_env_data["year"], layer
)

fig_synth = create_timeseries(
    outcomes_synth_selected, label="Synthetic sampled " + layer + " data"
)
go.Figure(fig_synth).show()

save_csv_plotting(
    env_df, new_data_env, selected_env_data, root_site_data_synth[10:], layer
)
