import sys
sys.path.append("..")
from src.models.connectivity_GAN_model import connectivity_model

from sdmetrics.reports.single_table import QualityReport

### ------------------------------------------------Key Inputs---------------------------------------------------###
root_original_file = 'Moore_2022-11-17'
root_site_data_synth = 'site_data_7-3-2023_154918_numsamps_100.csv'
year = "2015" # connectivity data year to use
num = "3" # connectivity data sample number to use

conn_orig, conn_samples, selected_conn_data, metadata_conn, synth_conn_fn = connectivity_model(root_original_file, root_site_data_synth, year, num)

### ------------------------------------Test synthetic data utility--------------------------------------------###
report = QualityReport()
report.generate(conn_orig, conn_samples[conn_orig.columns], metadata_conn)
report.get_details(property_name='Column Shapes')
report.get_details(property_name='Pair Trends')

