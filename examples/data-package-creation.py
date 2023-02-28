from src.models.site_data_fastML_model import site_data_model
from src.models.coral_cover_TVAE_model import coral_cover_model
from src.models.env_PAR_model import env_data_model
from src.models.connectivity_GAN_model import connectivity_model

orig_data_package = "Moore_2022-11-17"

### --------------------------------------- Create synthetic site data ----------------------------------------- ###

site_data, new_site_data, sample_sites, metadata_site, synth_site_data_fn = site_data_model(orig_data_package, 300, 30, 10)

### ------------------------------------- Create synthetic coral covers ---------------------------------------- ###

cover_df, synth_cover, synth_sampled, metadata_cover, root_site_data_synth = coral_cover_model(orig_data_package, synth_site_data_fn, 300)

### ----------------------------------- Create synthetic environmental layers ---------------------------------- ###

layer1 = 'dhw'
layer2 = 'Ub'
rcp = '45'

dhw_df, new_data_dhw, selected_dhw_data, metadata_dhw, nyears, old_years = env_data_model(orig_data_package, synth_site_data_fn, rcp, layer1)

wave_df, new_data_wave, selected_wave_data, metadata_wave, nyears, old_years = env_data_model(orig_data_package, synth_site_data_fn, rcp, layer2)

### ----------------------------------- Create synthetic connectivity layers ---------------------------------- ###
year = "2015" # connectivity data year to use
num = "3" # connectivity data sample number to use

conn_orig, conn_samples, selected_conn_data, metadata_conn, synth_conn_fn = connectivity_model(orig_data_package, synth_site_data_fn, year, num)
