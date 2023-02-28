import geopandas as gp

from src.data_processing.preprocess_functions import convert_to_csv
from src.data_processing.package_synth_data import retrieve_orig_site_data_fp

orig_data_package = "Moore_2022-11-17"

# convert to csv to use in synthetic data generation models (package clash with connectivity model and geopandas)
site_data_geo_fn = retrieve_orig_site_data_fp(orig_data_package,".gpkg")
site_data_geo = gp.read_file(site_data_geo_fn)
convert_to_csv(site_data_geo,site_data_geo_fn)