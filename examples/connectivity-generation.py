import sys
import time
sys.path.append("..")

from src.models.connectivity_GAN_model import connectivity_model
from src.data_processing.package_synth_data import save_csv_plotting

from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

### ------------------------------------------------Key Inputs---------------------------------------------------###
root_original_file = "Moore_2022-11-17"
root_site_data_synth = "synth_2023-7-24_152038.csv"
years = ["2015", "2016", "2017"]  # connectivity data years to use
num = ["1", "2", "3"]  # connectivity data sample number to use
model_type = "GaussianCopula" # "GAN"
tic = time.perf_counter()
(
    conn_orig,
    conn_samples,
    selected_conn_data,
    metadata_conn,
    synth_conn_fn,
) = connectivity_model(root_original_file, root_site_data_synth, years, num, model_type)
toc = time.perf_counter()
print(f"Model learnt in {toc - tic:0.4f} seconds")
breakpoint()

### ------------------------------------Test synthetic data utility-------------------------------------------- ###
# Beware - there will be a lot of warning messages for the correlation statistic because the connnectivity matrix has so many zeros
metadata_conn = SingleTableMetadata()
metadata_conn.detect_from_dataframe(data=conn_orig[conn_orig.columns[0:216]])
quality_report = evaluate_quality(real_data=conn_orig[conn_orig.columns[0:216]],synthetic_data=conn_samples[conn_orig.columns[0:216]],metadata=metadata_conn,)

# Save original, sampled and synthetic as csvs if plotting later (only sampled is saved in data package)
save_csv_plotting(
    conn_orig[conn_orig.columns[0:216]],
    conn_samples[conn_orig.columns[0:216]],
    selected_conn_data,
    root_site_data_synth,
    "connectivity",
)
