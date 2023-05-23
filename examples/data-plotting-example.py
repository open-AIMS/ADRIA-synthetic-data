import pandas as pd
import geopandas as gpd
import json
import plotly.graph_objects as go
import plotly.express as px
import sklearn
import numpy as np
import sys

sys.path.append("..")
from src.plotting.data_comparison_plots import (
    comparison_plots_site_data,
    compared_cover_species_hist,
    get_data_quantiles,
    create_timeseries,
)

c_dir = "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\synthetic_data\\"


# site data plotting
site_data_orig = pd.read_csv(
    c_dir + "site_data_orig_plotting_site_data.csv", index_col=False
)
site_data_sampled = pd.read_csv(
    c_dir + "site_data_samp_plotting_site_data.csv", index_col=False
)
site_data_synth = pd.read_csv(
    c_dir + "site_data_synth_plotting_site_data.csv", index_col=False
)

figs_site_data = comparison_plots_site_data(
    site_data_sampled, site_data_synth, site_data_orig
)

breakpoint()


# cover data plotting
cover_data_orig = pd.read_csv(
    c_dir + "covers_orig_plotting_synth_3-5-2023_93547.csv", index_col=False
)
cover_data_sampled = pd.read_csv(
    c_dir + "covers_samp_plotting_synth_3-5-2023_93547.csv", index_col=False
)
cover_data_synth = pd.read_csv(
    c_dir + "covers_synth_plotting_synth_3-5-2023_93547.csv", index_col=False
)

figs_cover = compared_cover_species_hist(
    cover_data_orig, cover_data_synth, cover_data_sampled
)
site_ids = np.unique(cover_data_sampled["reef_siteid"])
covers_sum = np.zeros(len(site_ids))
for k in range(len(site_ids)):
    covers_sum[k] = sum(
        cover_data_sampled["cover"][cover_data_sampled["reef_siteid"] == site_ids[k]]
    )

site_ids = np.unique(cover_data_orig["site_id"])
covers_sum_orig = np.zeros(len(site_ids))
for k in range(len(site_ids)):
    covers_sum_orig[k] = sum(
        cover_data_orig["cover"][cover_data_orig["site_id"] == site_ids[k]]
    )

breakpoint()
# connectivity data plotting
conn_data_orig = pd.read_csv(
    c_dir + "connectivity_orig_plotting_synth_3-5-2023_93547.csv", index_col=False
)
conn_data_sampled = pd.read_csv(
    c_dir + "connectivity_samp_plotting_synth_3-5-2023_93547.csv", index_col=False
)
conn_data_synth = pd.read_csv(
    c_dir + "connectivity_synth_plotting_synth_3-5-2023_93547.csv", index_col=False
)
breakpoint()
# dhw data plotting
dhw_data_orig = pd.read_csv(
    c_dir + "dhw_orig_plotting_synth_3-5-2023_93547.csv", index_col=False
)
dhw_data_sampled = pd.read_csv(
    c_dir + "dhw_samp_plotting_synth_3-5-2023_93547.csv", index_col=False
)
dhw_data_synth = pd.read_csv(
    c_dir + "dhw_synth_plotting_synth_3-5-2023_93547.csv", index_col=False
)
breakpoint()
old_years = np.array([year[0:4] for year in dhw_data_orig["year"]])
layer = "dhw"

outcomes_data, outcomes_synth = get_data_quantiles(
    dhw_data_orig, dhw_data_synth, 50, old_years, dhw_data_synth["year"], layer
)
fig_data = create_timeseries(outcomes_data, label="Original " + layer + " Data")
go.Figure(fig_data).show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic " + layer + " Data")
go.Figure(fig_synth).show()

outcomes_data, outcomes_synth_selected = get_data_quantiles(
    dhw_data_orig, dhw_data_sampled, 50, old_years, dhw_data_sampled["year"], layer
)

fig_synth_sel = create_timeseries(
    outcomes_synth_selected, label="Synthetic sampled " + layer + " data"
)
go.Figure(fig_synth_sel).show()

breakpoint()
# wave data plotting
wave_data_orig = pd.read_csv(
    c_dir + "wave_orig_plotting_synth_3-5-2023_93547.csv", index_col=False
)
wave_data_sampled = pd.read_csv(
    c_dir + "wave_samp_plotting_synth_3-5-2023_93547.csv", index_col=False
)
wave_data_synth = pd.read_csv(
    c_dir + "wave_synth_plotting_synth_3-5-2023_93547.csv", index_col=False
)
breakpoint()
layer = "Ub"

outcomes_data, outcomes_synth = get_data_quantiles(
    wave_data_orig, wave_data_synth, 50, old_years, wave_data_synth["year"], layer
)
fig_data = create_timeseries(outcomes_data, label="Original " + layer + " Data")
go.Figure(fig_data).show()

fig_synth = create_timeseries(outcomes_synth, label="Synthetic " + layer + " Data")
go.Figure(fig_synth).show()

outcomes_data, outcomes_synth_selected = get_data_quantiles(
    wave_data_orig, wave_data_sampled, 50, old_years, wave_data_sampled["year"], layer
)

fig_synth_sel = create_timeseries(
    outcomes_synth_selected, label="Synthetic sampled " + layer + " data"
)
go.Figure(fig_synth_sel).show()

breakpoint()
orig_sitedata_gpkg = gpd.read_file(
    "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\original_data\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.gpkg"
)
with open("site_data_Moore_2022-11-17.json") as response:
    polygons = json.load(response)
orig_sitedata_gpkg["covers"] = covers_sum_orig
fig = px.choropleth(
    orig_sitedata_gpkg,
    geojson=polygons,
    locations="reef_siteid",
    featureidkey="properties.reef_siteid",
    color="covers",
    color_continuous_scale="Viridis",
    labels={"covers": "coral cover"},
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()

synth_sitedata_gpkg = gpd.read_file(
    "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\synthetic_data\\synthetic_data_packages\\synth_3-5-2023_93547\\site_data\\synth_3-5-2023_93547.gpkg"
)
with open("site_data_synth_3-5-2023_93547.json") as response:
    polygons_synth = json.load(response)
synth_sitedata_gpkg["covers"] = covers_sum
fig = px.choropleth(
    synth_sitedata_gpkg,
    geojson=polygons_synth,
    locations="reef_siteid",
    featureidkey="properties.reef_siteid",
    color="covers",
    color_continuous_scale="Viridis",
    labels={"area": "area"},
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()
# convert gpkgs to geojsons

# orig_sitedata_gpkg.to_file("site_data_Moore_2022-11-17.json", driver="GeoJSON")

# synth_sitedata_gpkg.to_file("site_data_synth_3-5-2023_93547.json", driver="GeoJSON")
