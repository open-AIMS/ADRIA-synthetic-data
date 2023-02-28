import pandas as pd

from sdv.lite import TabularPreset

from src.data_processing.preprocess_functions import preprocess_site_data
from src.data_processing.sampling_functions import sample_rand_radii
from src.data_processing.postprocess_functions import anonymize_spatial, generate_timestamp
from src.data_processing.package_synth_data import initialize_data_package,retrieve_orig_site_data_fp, create_synth_site_fp,create_synth_site_data_package_fp

def site_data_model(orig_data_package, N, N2, N3):
    ### --------------------------------------Load site data to synethesize--------------------------------------###

    site_data_fn = retrieve_orig_site_data_fp(orig_data_package,".csv")
    site_data = pd.read_csv(site_data_fn)

    ### ---------------------------------Preprocess data for sdv.TVAE model fit----------------------------------###
    # simplify to dataframe
    site_data, metadata_site = preprocess_site_data(site_data)

    ### -----------------------------------Fit and save TVAE model for site data-------------------------------------###
    # set up TVAE, fit and save
    model = TabularPreset(name='FAST_ML', metadata=metadata_site)
    model.fit(site_data)

    ### ----------------------------------------Sample data and test utility-----------------------------------------###
    # create sample data
    new_site_data = model.sample(num_rows=N)
    
    ### ----------------Re-sample using conditional sampling to emulated site spatial clustering------------------###
    conditions = sample_rand_radii(new_site_data,N3,N2)
    sample_sites = model.sample_remaining_columns(conditions)

    ### ------------------------------------------Save site data to csv-------------------------------------------###
    time_stamp = generate_timestamp()
    sample_sites['lat'] = -1*sample_sites['lat']
    breakpoint()
    sample_sites_fn = create_synth_site_fp(time_stamp,N2)
    sample_sites.to_csv(sample_sites_fn)
    sample_sites_anon = anonymize_spatial(sample_sites)

    initialize_data_package(time_stamp+'_numsamps_'+str(N2))
    synth_site_data_fn = create_synth_site_data_package_fp(time_stamp,N2)
    sample_sites_anon.to_csv(synth_site_data_fn)

    return site_data, new_site_data, sample_sites, metadata_site, synth_site_data_fn