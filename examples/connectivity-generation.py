import sys
sys.path.append("..")

from src.models.connectivity_GAN_model import connectivity_model
from src.data_processing.package_synth_data import save_csv_plotting
#from sdmetrics.reports.single_table import QualityReport

### ------------------------------------------------Key Inputs---------------------------------------------------###
root_original_file = 'Moore_2022-11-17'
root_site_data_synth = 'site_data_synth_3-5-2023_93547.csv'
year = "2017" # connectivity data year to use
num = "3" # connectivity data sample number to use

conn_orig, conn_samples, selected_conn_data, metadata_conn, synth_conn_fn = connectivity_model(root_original_file, root_site_data_synth, year, num)
breakpoint()
### ------------------------------------Test synthetic data utility--------------------------------------------###
# report = QualityReport()
# report.generate(conn_orig, conn_samples[conn_orig.columns], metadata_conn)
# report.get_details(property_name='Column Shapes')
# report.get_details(property_name='Pair Trends')
save_csv_plotting(conn_orig,conn_samples,selected_conn_data,root_site_data_synth[10:],"connectivity")
