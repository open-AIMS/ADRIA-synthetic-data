import sys

sys.path.append("..")

from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

from src.models.site_data_fastML_model import site_data_model
from src.models.site_data_fastML_model import site_data_model
from src.models.coral_cover_TVAE_model import coral_cover_model
from src.models.env_PAR_model import env_data_model
from src.models.connectivity_GAN_model import connectivity_model
from src.data_processing.package_synth_data import (
    create_dp_jason,
    retrieve_orig_data_package_path,
)

orig_data_package = "Moore_2023-08-17"

### --------------------------------------- Create synthetic site data ----------------------------------------- ###

(
    site_data,
    new_site_data,
    sample_sites,
    metadata_site,
    synth_site_data_fn,
) = site_data_model(orig_data_package, 300, 30, 10)


### ------------------------------------- Create synthetic coral covers ---------------------------------------- ###
breakpoint()
synth_fn = synth_site_data_fn.split("\\")[-1][:-5] + ".csv"
(cover_df, synth_cover, synth_sampled, metadata_cover, synth_fn) = coral_cover_model(
    orig_data_package, synth_fn, 300
)

### ----------------------------------- Create synthetic environmental layers ---------------------------------- ###

layer1 = "dhw"
layer2 = "Ub"
rcp = "45"
samples = 10
replicates_dhw = [10, 38, 41, 42, 45]
replicates_waves = [1, 5, 13, 16, 45]

(
    dhw_df,
    new_data_dhw,
    selected_dhw_data,
    metadata_dhw,
    nyears,
    old_years,
) = env_data_model(orig_data_package, synth_fn, samples, replicates_dhw, rcp, layer1)

(
    wave_df,
    new_data_wave,
    selected_wave_data,
    metadata_wave,
    nyears,
    old_years,
) = env_data_model(orig_data_package, synth_fn, samples, replicates_waves, rcp, layer2)

### ----------------------------------- Create synthetic connectivity layers ---------------------------------- ###
years = ["2015", "2016", "2017"]  # connectivity data years to use
num = ["1", "2", "3"]  # connectivity data sample number to use
model_type = "GAN"  # "GaussianCopula"
(
    conn_orig,
    conn_samples,
    selected_conn_data,
    metadata_conn,
    synth_conn_fn,
) = connectivity_model(orig_data_package, synth_fn, years, num, model_type)


### ----------------------------------------- Add json to data package ----------------------------------------- ###

orig_data_package_path = retrieve_orig_data_package_path(orig_data_package)
create_dp_jason(orig_data_package_path, synth_fn[:-4])
