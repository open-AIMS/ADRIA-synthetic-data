import pandas as pd
import geopandas as gp

# import geopandas as gpd
import json
import plotly.graph_objects as go
import plotly.express as px
import sklearn
import numpy as np
import sys

from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import NewRowSynthesis
from sdmetrics.single_column import RangeCoverage, BoundaryAdherence
from datetime import datetime

sys.path.append("..")
from src.plotting.data_comparison_plots import (
    comparison_plots_site_data,
    plot_comparison_hist_covers,
    compared_cover_species_hist,
    pca_correlation,
    plot_pca,
    correlation_distance,
    correlation_heatmap,
    plot_mean_std,
    get_data_quantiles,
    create_timeseries,
)


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

metadata_ub = {"columns": { "year": {"sdtype": "datetime"},"lat": {"sdtype": "numerical", "subtype": "float"},"long": {"sdtype": "numerical", "subtype": "float"},"site": {"sdtype": "id"},"Ub": {"sdtype": "numerical", "subtype": "float"},},"sequence_key": "site","entity_columns": ["lat", "long"],"sequence_index": "year",}

c_dir = "C:\\Users\\rcrocker\\Documents\\GitHub\\ADRIA-synthetic-data\\synthetic_data\\"
# synth_id = "synth_2024-4-4_123411"
synth_id = "synth_2024-3-5_81235"
synth_lat_longs = gp.read_file(
    c_dir
    + "synthetic_data_packages\\"
    + synth_id
    + "\\site_data\\"
    + synth_id
    + ".gpkg"
)
anon_long = synth_lat_longs["geometry"].centroid.x
anon_lat = synth_lat_longs["geometry"].centroid.y

breakpoint()
# site data plotting
site_data_orig = pd.read_csv(c_dir + "site_data_orig_plotting_" + synth_id + ".csv", index_col=False)
site_data_sampled = pd.read_csv(c_dir + "site_data_samp_plotting_" + synth_id + ".csv", index_col=False)
site_data_synth = pd.read_csv(c_dir + "site_data_synth_plotting_" + synth_id + ".csv", index_col=False)
breakpoint()
metadata_site = SingleTableMetadata()
metadata_site.detect_from_dataframe(data=site_data_orig[site_data_orig.columns[1:]])
quality_report = evaluate_quality(real_data=site_data_orig[site_data_orig.columns[1:]],synthetic_data=site_data_synth[site_data_orig.columns[1:]],metadata=metadata_site)
diag_report = run_diagnostic(
    real_data=site_data_orig,
    synthetic_data=site_data_synth[site_data_orig.columns],
    metadata=metadata_site,
)

breakpoint()
site_data_sampled["lat"] = anon_lat
site_data_sampled["long"] = anon_long
site_data_orig["lat"] = -site_data_orig["lat"]
site_data_synth["lat"] = -site_data_synth["lat"]
figs_site_data = comparison_plots_site_data(
    site_data_sampled, site_data_synth, site_data_orig
)
# cover data plotting
cover_data_orig = pd.read_csv(
    c_dir + "covers_orig_plotting_" + synth_id + ".csv", index_col=False
)
cover_data_sampled = pd.read_csv(
    c_dir + "covers_samp_plotting_" + synth_id + ".csv", index_col=False
)
cover_data_synth = pd.read_csv(
    c_dir + "covers_synth_plotting_" + synth_id + ".csv", index_col=False
)

figs_cover = compared_cover_species_hist(
    cover_data_orig, cover_data_synth, cover_data_sampled
)
breakpoint()
metadata_cc = SingleTableMetadata()
metadata_cc.detect_from_dataframe(data=cover_data_orig)
metadata_cc.update_column(column_name='species',sdtype='categorical')
quality_report = evaluate_quality(real_data=cover_data_orig,synthetic_data=cover_data_synth,metadata=metadata_cc,)
    synthetic_data=cover_data_synth,
    metadata=metadata_cc,
)
diag_report = run_diagnostic(
    real_data=cover_data_orig,
    synthetic_data=cover_data_synth,
    metadata=metadata_cc,
)
breakpoint()

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
figs_cover_2 = plot_comparison_hist_covers(
    covers_sum_orig,
    covers_sum_samp,
)

breakpoint()
# connectivity data plotting
conn_data_orig = pd.read_csv(
    c_dir + "connectivity_orig_plotting_"+synth_id+".csv",
    index_col=False,
    dtype=np.float64,
)
conn_data_sampled = pd.read_csv(
    c_dir + "connectivity_samp_plotting_"+synth_id+".csv", index_col=False
)
conn_data_synth = pd.read_csv(
    c_dir + "connectivity_synth_plotting_"+synth_id+".csv",
    index_col=False,
    dtype=np.float64,
)

conn_fields = {kk: {"sdtype": "numerical", "subtype": "float"} for kk in conn_data_orig.columns}
}
# create metadata dictionary
metadata_conn = {"columns": conn_fields, "primary_key": "recieving_site"}

plot_pca(
    conn_data_orig,
    conn_data_synth[conn_data_orig.columns],
    conn_data_sampled[conn_data_sampled.columns[1:]],
)
PCA_mean_error, pca_orig, pca_synth = pca_correlation(
    conn_data_orig.to_numpy(), conn_data_synth[conn_data_orig.columns].to_numpy()
)

fig = plot_mean_std(conn_data_orig, conn_data_synth[conn_data_orig.columns])
fig.show()
breakpoint()
report = QualityReport()
report.generate(
    conn_data_orig,
    conn_data_synth[conn_data_orig.columns],
    metadata_conn,
)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

breakpoint()
corr_orig, corr_synth, corr_samp = correlation_distance(
    conn_data_orig,
    conn_data_synth[conn_data_orig.columns],
    conn_data_sampled[conn_data_sampled.columns[1:]],
    [],
)

correlation_heatmap(
    corr_orig.astype(float).values,
    corr_synth.astype(float).values,
    corr_samp.astype(float).values,
)

# coverage = np.zeros((len(conn_data_orig.columns), 1))
# boundaries = np.zeros((len(conn_data_orig.columns), 1))
# for kk in range(len(coverage)):
#     coverage[kk] = RangeCoverage.compute(
#         real_data=conn_data_orig[conn_data_orig.columns[kk]],
#         synthetic_data=conn_data_synth[conn_data_orig.columns[kk]],
#     )
#     boundaries[kk] = BoundaryAdherence.compute(
#         real_data=conn_data_orig[conn_data_orig.columns[kk]],
#         synthetic_data=conn_data_synth[conn_data_orig.columns[kk]],
#     )

breakpoint()
# dhw data plotting
dhw_data_orig = pd.read_csv(
    c_dir + "dhw_orig_plotting_"+synth_id+".csv", index_col=False
)
dhw_data_sampled = pd.read_csv(
    c_dir + "dhw_samp_plotting_"+synth_id+".csv", index_col=False
)
dhw_data_synth = pd.read_csv(
    c_dir + "dhw_synth_plotting_"+synth_id+".csv", index_col=False
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
    c_dir + "Ub_orig_plotting_"+synth_id+".csv", index_col=False
)
wave_data_sampled = pd.read_csv(
    c_dir + "Ub_samp_plotting_"+synth_id+".csv", index_col=False
)
wave_data_synth = pd.read_csv(
    c_dir + "Ub_synth_plotting_"+synth_id+".csv", index_col=False
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
    wave_data_orig, wave_data_synth, 75, layer
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
    wave_data_orig, wave_data_sampled, 75, layer
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
