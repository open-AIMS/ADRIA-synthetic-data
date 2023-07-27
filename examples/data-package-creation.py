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

orig_data_package = "Moore_2022-11-17"

### --------------------------------------- Create synthetic site data ----------------------------------------- ###

(
    site_data,
    new_site_data,
    sample_sites,
    metadata_site,
    synth_site_data_fn,
) = site_data_model(orig_data_package, 300, 30, 10)

quality_report = evaluate_quality(
    real_data=site_data, synthetic_data=sample_sites, metadata=metadata_site
)

### ------------------------------------- Create synthetic coral covers ---------------------------------------- ###

(
    cover_df,
    synth_cover,
    synth_sampled,
    metadata_cover,
    root_site_data_synth,
) = coral_cover_model(orig_data_package, synth_site_data_fn, 300)

quality_report = evaluate_quality(
    real_data=cover_df, synthetic_data=synth_cover, metadata=metadata_cover
)

### ----------------------------------- Create synthetic environmental layers ---------------------------------- ###

layer1 = "dhw"
layer2 = "Ub"
rcp = "45"

(
    dhw_df,
    new_data_dhw,
    selected_dhw_data,
    metadata_dhw,
    nyears,
    old_years,
) = env_data_model(orig_data_package, synth_site_data_fn, rcp, layer1)

(
    wave_df,
    new_data_wave,
    selected_wave_data,
    metadata_wave,
    nyears,
    old_years,
) = env_data_model(orig_data_package, synth_site_data_fn, rcp, layer2)

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
) = connectivity_model(orig_data_package, synth_site_data_fn, years, num, model_type)

metadata_conn = SingleTableMetadata()
metadata_conn.detect_from_dataframe(data=conn_orig[conn_orig.columns[0:216]])
quality_report = evaluate_quality(
    real_data=conn_orig[conn_orig.columns[0:216]],
    synthetic_data=conn_samples[conn_orig.columns[0:216]],
    metadata=metadata_conn,
)

### ----------------------------------------- Add json to data package ----------------------------------------- ###

orig_data_package_path = retrieve_orig_data_package_path(orig_data_package)
create_dp_jason(orig_data_package_path, synth_site_data_fn)
