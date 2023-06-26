import pandas as pd

# import geopandas as gpd
import json
import plotly.graph_objects as go
import plotly.express as px
import sklearn
import numpy as np
import sys

from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from datetime import datetime

sys.path.append("..")
from src.plotting.data_comparison_plots import (
    comparison_plots_site_data,
    plot_comparison_scatter_covers,
    compared_cover_species_hist,
    plot_pca,
    plot_mean_std,
    get_data_quantiles,
    create_timeseries,
)

metadata_site_data = {
    "fields": {
        "site_id": {"type": "id"},
        "habitat": {"type": "categorical"},
        "k": {"type": "numerical", "subtype": "float"},
        "area": {"type": "numerical", "subtype": "float"},
        "rubble": {"type": "numerical", "subtype": "float"},
        "sand": {"type": "numerical", "subtype": "float"},
        "rock": {"type": "numerical", "subtype": "float"},
        "coral_algae": {"type": "numerical", "subtype": "float"},
        "na_proportion": {"type": "numerical", "subtype": "float"},
        "depth_mean": {"type": "numerical", "subtype": "float"},
        "depth_sd": {"type": "numerical", "subtype": "float"},
        "depth_med": {"type": "numerical", "subtype": "float"},
        "zone_type": {"type": "categorical"},
        "long": {"type": "numerical", "subtype": "float"},
        "lat": {"type": "numerical", "subtype": "float"},
    },
    "primary_key": "site_id",
}

metadata_cc = {
    "fields": {
        "site_id": {"type": "id"},
        "species": {"type": "categorical"},
        "lat": {"type": "numerical", "subtype": "float"},
        "long": {"type": "numerical", "subtype": "float"},
        "cover": {"type": "numerical", "subtype": "float"},
    },
    "primary_key": "site_id",
}

metadata_dhw = {
    "columns": {
        "year": {"sdtype": "datetime"},
        "lat": {"sdtype": "numerical", "subtype": "float"},
        "long": {"sdtype": "numerical", "subtype": "float"},
        "site": {"sdtype": "id"},
        "dhw": {"sdtype": "numerical", "subtype": "float"},
    },
    "sequence_key": "site",
    "entity_columns": ["lat", "long"],
    "sequence_index": "year",
}

metadata_ub = {
    "fields": {
        "year": {"type": "numerical"},
        "lat": {"type": "numerical", "subtype": "float"},
        "long": {"type": "numerical", "subtype": "float"},
        "site": {"type": "categorical"},
        "Ub": {"type": "numerical", "subtype": "float"},
    },
    "context_columns": ["site"],
    "entity_columns": ["lat", "long"],
    "sequence_index": "year",
}

c_dir = "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\synthetic_data\\"


# site data plotting
site_data_orig = pd.read_csv(
    c_dir + "site_data_orig_plotting_16-6-2023_91140.csv", index_col=False
)
site_data_sampled = pd.read_csv(
    c_dir + "site_data_samp_plotting_16-6-2023_91140.csv", index_col=False
)
site_data_synth = pd.read_csv(
    c_dir + "site_data_synth_plotting_16-6-2023_91140.csv", index_col=False
)

figs_site_data = comparison_plots_site_data(
    site_data_sampled, site_data_synth, site_data_orig
)
breakpoint()
report = QualityReport()
report.generate(site_data_orig, site_data_synth, metadata_site_data)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

report = DiagnosticReport()
report.generate(
    site_data_orig,
    site_data_synth,
    metadata_site_data,
)
report.get_properties()
breakpoint()

# cover data plotting
cover_data_orig = pd.read_csv(
    c_dir + "covers_orig_plotting_synth_16-6-2023_91140.csv", index_col=False
)
cover_data_sampled = pd.read_csv(
    c_dir + "covers_samp_plotting_synth_16-6-2023_91140.csv", index_col=False
)
cover_data_synth = pd.read_csv(
    c_dir + "covers_synth_plotting_synth_16-6-2023_91140.csv", index_col=False
)

figs_cover = compared_cover_species_hist(
    cover_data_orig, cover_data_synth, cover_data_sampled
)

report = QualityReport()
report.generate(cover_data_orig, cover_data_synth, metadata_cc)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")
report = DiagnosticReport()
report.generate(
    cover_data_orig,
    cover_data_sampled,
    metadata_cc,
)
report.get_properties()

breakpoint()
site_ids = np.unique(cover_data_synth["site_id"])
covers_sum_synth = np.zeros(len(site_ids))
for k in range(len(site_ids)):
    covers_sum_synth[k] = sum(
        cover_data_synth["cover"][cover_data_synth["site_id"] == site_ids[k]]
    )

site_ids = np.unique(cover_data_sampled["reef_siteid"])
covers_sum_samp = np.zeros(len(site_ids))
for k in range(len(site_ids)):
    covers_sum_samp[k] = sum(
        cover_data_sampled["cover"][cover_data_sampled["reef_siteid"] == site_ids[k]]
    )

site_ids = np.unique(cover_data_orig["site_id"])
covers_sum_orig = np.zeros(len(site_ids))
for k in range(len(site_ids)):
    covers_sum_orig[k] = sum(
        cover_data_orig["cover"][cover_data_orig["site_id"] == site_ids[k]]
    )
figs_cover_2 = plot_comparison_scatter_covers(
    covers_sum_orig,
    site_data_orig,
    covers_sum_synth,
    site_data_synth,
    covers_sum_samp,
    site_data_sampled,
)
breakpoint()
# connectivity data plotting
conn_data_orig = pd.read_csv(
    c_dir + "connectivity_orig_plotting_synth_16-6-2023_91140.csv",
    index_col=False,
    dtype=np.float64,
)
conn_data_sampled = pd.read_csv(
    c_dir + "connectivity_samp_plotting_synth_16-6-2023_91140.csv", index_col=False
)
conn_data_synth = pd.read_csv(
    c_dir + "connectivity_synth_plotting_synth_16-6-2023_91140.csv",
    index_col=False,
    dtype=np.float64,
)

plot_pca(conn_data_orig, conn_data_synth[conn_data_orig.columns])

fig = plot_mean_std(conn_data_orig, conn_data_synth[conn_data_orig.columns])
fig.show()

breakpoint()
# dhw data plotting
dhw_data_orig = pd.read_csv(
    c_dir + "dhw_orig_plotting_16-6-2023_91140.csv", index_col=False
)
dhw_data_sampled = pd.read_csv(
    c_dir + "dhw_samp_plotting_16-6-2023_91140.csv", index_col=False
)
dhw_data_synth = pd.read_csv(
    c_dir + "dhw_synth_plotting_16-6-2023_91140.csv", index_col=False
)

report = QualityReport()
report.generate(
    dhw_data_orig[dhw_data_orig.columns[[0, 1, 2, 3, 5]]],
    dhw_data_synth[dhw_data_orig.columns[[0, 1, 2, 3, 5]]],
    metadata_dhw,
)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

report = DiagnosticReport()
report.generate(
    dhw_data_orig[dhw_data_orig.columns[[0, 1, 2, 3, 5]]],
    dhw_data_sampled[dhw_data_orig.columns[[0, 1, 2, 3, 5]]],
    metadata_dhw,
)
report.get_properties()

old_years = np.array([str(year) for year in dhw_data_orig["year"]])
new_years = np.array([str(year) for year in dhw_data_synth["year"]])
layer = "dhw"

outcomes_data, outcomes_synth = get_data_quantiles(
    dhw_data_orig, dhw_data_synth, 75, old_years, new_years, layer
)
fig_data = create_timeseries(
    outcomes_data, np.unique(old_years), layer, label="Original " + layer + " Data"
)
go.Figure(fig_data).show()
breakpoint()
fig_synth = create_timeseries(
    outcomes_synth, np.unique(new_years), layer, label="Synthetic " + layer + " Data"
)
go.Figure(fig_synth).show()

new_years_samps = np.array([str(int(year)) for year in dhw_data_sampled["year"]])
outcomes_data, outcomes_synth_selected = get_data_quantiles(
    dhw_data_orig, dhw_data_sampled, 75, old_years, new_years_samps, layer
)

fig_synth_sel = create_timeseries(
    outcomes_synth_selected,
    np.unique(new_years_samps),
    layer,
    label="Synthetic sampled " + layer + " data",
)
go.Figure(fig_synth_sel).show()

breakpoint()
# wave data plotting
wave_data_orig = pd.read_csv(
    c_dir + "Ub_orig_plotting_16-6-2023_91140.csv", index_col=False
)
wave_data_sampled = pd.read_csv(
    c_dir + "Ub_samp_plotting_16-6-2023_91140.csv", index_col=False
)
wave_data_synth = pd.read_csv(
    c_dir + "Ub_synth_plotting_16-6-2023_91140.csv", index_col=False
)
report = QualityReport()
report.generate(
    wave_data_orig[wave_data_orig.columns[[0, 1, 2, 3, 5]]],
    wave_data_synth,
    metadata_ub,
)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

report = DiagnosticReport()
report.generate(
    wave_data_orig[wave_data_orig.columns[[0, 1, 2, 3, 5]]],
    wave_data_synth,
    metadata_ub,
)
report.get_properties()

breakpoint()
layer = "Ub"
new_years = np.array([str(year) for year in wave_data_synth["year"]])
outcomes_data, outcomes_synth = get_data_quantiles(
    wave_data_orig, wave_data_synth, 75, old_years, new_years, layer
)
fig_data = create_timeseries(
    outcomes_data, np.unique(old_years), layer, label="Original " + layer + " Data"
)
go.Figure(fig_data).show()

fig_synth = create_timeseries(
    outcomes_synth, np.unique(new_years), layer, label="Synthetic " + layer + " Data"
)
go.Figure(fig_synth).show()


breakpoint()
new_years_samps = np.array([str(int(year)) for year in wave_data_sampled["year"]])
outcomes_data, outcomes_synth_selected = get_data_quantiles(
    wave_data_orig, wave_data_sampled, 75, old_years, new_years_samps, layer
)

fig_synth_sel = create_timeseries(
    outcomes_synth_selected,
    np.unique(new_years_samps),
    layer,
    label="Synthetic sampled " + layer + " data",
)
go.Figure(fig_synth_sel).show()

breakpoint()
# orig_sitedata_gpkg = gpd.read_file(
#     "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\original_data\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.gpkg"
# )
# with open("site_data_Moore_2022-11-17.json") as response:
#     polygons = json.load(response)
# orig_sitedata_gpkg["covers"] = covers_sum_orig
# fig = px.choropleth(
#     orig_sitedata_gpkg,
#     geojson=polygons,
#     locations="reef_siteid",
#     featureidkey="properties.reef_siteid",
#     color="covers",
#     color_continuous_scale="Viridis",
#     labels={"covers": "coral cover"},
# )
# fig.update_geos(fitbounds="locations", visible=False)
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# fig.show()

# synth_sitedata_gpkg = gpd.read_file(
#     "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\synthetic_data\\synthetic_data_packages\\synth_3-5-2023_93547\\site_data\\synth_3-5-2023_93547.gpkg"
# )
# with open("site_data_synth_3-5-2023_93547.json") as response:
#     polygons_synth = json.load(response)
# synth_sitedata_gpkg["covers"] = covers_sum
# fig = px.choropleth(
#     synth_sitedata_gpkg,
#     geojson=polygons_synth,
#     locations="reef_siteid",
#     featureidkey="properties.reef_siteid",
#     color="covers",
#     color_continuous_scale="Viridis",
#     labels={"area": "area"},
# )
# fig.update_geos(fitbounds="locations", visible=False)
# fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# fig.show()
# convert gpkgs to geojsons

# orig_sitedata_gpkg.to_file("site_data_Moore_2022-11-17.json", driver="GeoJSON")

# synth_sitedata_gpkg.to_file("site_data_synth_3-5-2023_93547.json", driver="GeoJSON")
