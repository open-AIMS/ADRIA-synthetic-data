import sys

sys.path.append("..")

from src.models.connectivity_GAN_model import connectivity_model
from src.data_processing.package_synth_data import save_csv_plotting

from sdmetrics.reports.single_table import QualityReport

### ------------------------------------------------Key Inputs---------------------------------------------------###
root_original_file = "Moore_2022-11-17"
root_site_data_synth = "synth_16-6-2023_91140.csv"
years = ["2015", "2016", "2017"]  # connectivity data years to use
num = ["1", "2", "3"]  # connectivity data sample number to use

(
    conn_orig,
    conn_samples,
    selected_conn_data,
    metadata_conn,
    synth_conn_fn,
) = connectivity_model(root_original_file, root_site_data_synth, years, num)
breakpoint()

### ------------------------------------Test synthetic data utility-------------------------------------------- ###
report = QualityReport()
report.generate(
    conn_orig[conn_orig.columns[0:216]],
    conn_samples[conn_orig.columns[0:216]],
    metadata_conn,
)
report.get_details(property_name="Column Shapes")
report.get_details(property_name="Column Pair Trends")

save_csv_plotting(
    conn_orig[conn_orig.columns[0:216]],
    conn_samples[conn_orig.columns[0:216]],
    selected_conn_data,
    root_site_data_synth,
    "connectivity",
)
