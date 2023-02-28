import pandas as pd
import geopandas as gp
import netCDF4

from sdv.tabular import TVAE

from src.data_processing.preprocess_functions import preprocess_cover_data
from src.data_processing.package_synth_data import retrieve_synth_site_data_fp, retrieve_orig_site_data_fp, retrieve_orig_cover_fp
from src.data_processing.sampling_functions import create_cover_conditional_struct
from src.data_processing.postprocess_functions import make_cover_array, create_cover_nc

###---------------------------------------Load site data to synethesize------------------------------------------###
def coral_cover_model(root_original_file, root_site_data_synth, N):
    original_cover_data_fn = retrieve_orig_cover_fp(root_original_file)
    original_site_data_fn = retrieve_orig_site_data_fp(root_original_file,'.csv')
    synth_site_data_fn = retrieve_synth_site_data_fp(root_site_data_synth)

    cover_orig = netCDF4.Dataset(original_cover_data_fn, 'r')
    site_data_orig = pd.read_csv(original_site_data_fn)
    site_data_synth = pd.read_csv(synth_site_data_fn)

    ###----------------------------------Preprocess data for sdv.fastML model fit------------------------------------###
    # simplify to dataframe
    cover_df, metadata_cover = preprocess_cover_data(cover_orig,site_data_orig)

    ###----------------------------------Fit and save fastML model for site data-------------------------------------###
    model = TVAE(primary_key='site_id')
    cover_df['lat'] = -1*cover_df['lat']
    model.fit(cover_df)
    synth_cover = model.sample(num_rows=N)

    ###----------Sample conditional dist, based on synthetic lats and longs and requirement of species types---------###
    conditions = create_cover_conditional_struct(site_data_synth,max(cover_df['species']))
    synth_sampled = model.sample_remaining_columns(conditions,max_tries_per_batch=300)

    array_synth_sampled = make_cover_array(synth_sampled)
    create_cover_nc(array_synth_sampled, root_site_data_synth)

    return cover_df, synth_cover, synth_sampled, metadata_cover, root_site_data_synth
