import sys

sys.path.append("..")

from src.data_processing.package_synth_data import (
    create_dp_jason,
    retrieve_orig_data_package_path,
)

orig_data_package = "Moore_2022-11-17"
synth_data_package_name = "synth_6-6-2023_144117"
orig_data_package_path = retrieve_orig_data_package_path(orig_data_package)
create_dp_jason(orig_data_package_path, synth_data_package_name)
