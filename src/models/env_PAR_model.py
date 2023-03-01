import pandas as pd

from deepecho import PARModel
import netCDF4

from src.data_processing.preprocess_functions import preprocess_env_data
from src.data_processing.sampling_functions import sample_env_ensemble
from src.data_processing.postprocess_functions import create_env_nc
from src.data_processing.package_synth_data import retrieve_orig_env_fp, retrieve_synth_site_data_fp,retrieve_synth_env_data_fp

### ----------------------------------Load site data and env data to synethesize----------------------------------###
def env_data_model(root_original_file, root_site_data_synth, nsamples,rcp, layer):
    original_data_fn = retrieve_orig_env_fp(root_original_file,rcp,layer)
    synth_data_fn = retrieve_synth_site_data_fp(root_site_data_synth)

    ENV = netCDF4.Dataset(original_data_fn, 'r')
    site_data_synth = pd.read_csv(synth_data_fn)

    ### --------------------------------------Reshape data to fit PAR model------------------------------------------###
    env_df, data_types, metadata_env, old_years, nyears = preprocess_env_data(ENV,layer)

    ### ----------------------------------------Set up and fit PAR model---------------------------------------------###
    model = PARModel(epochs=1024, cuda=False)
    model.fit(data=env_df,context_columns=['lat', 'long'],entity_columns=['site'],data_types=data_types,sequence_index='year')

    ### -----------------------------------------Sample data to synthesize--------------------------------------------###
    N_s = 200
    new_data_env = model.sample(N_s)
    new_years = [str(yr+2025) for yr in range(nyears)]
    new_years = new_years*N_s
    new_data_env['year'] = new_years

    ### -------------------------Sample synthetic env data at synthetic site data locations --------------------------###
    lat =-1.*site_data_synth.lat.values
    long = site_data_synth.long.values
    data = {'lat':lat,'long':long}
    context = pd.DataFrame(data)
    selected_env_data = model.sample(context=context)

    ### -------------------------------------Save sample data in sitedata package-------------------------------------###
    selected_years = [str(yr+2025) for yr in range(nyears)]
    selected_years = selected_years*len(site_data_synth.lat.values)
    selected_env_data.insert(4,"year",selected_years,True)

    nsites = len(site_data_synth.lat.values)

    selected_env_ensemble = sample_env_ensemble(model,context,nsamples,nsites,nyears,layer)

    synth_env_fn = retrieve_synth_env_data_fp(synth_data_fn,layer)

    create_env_nc(selected_env_ensemble,site_data_synth.lat.values,site_data_synth.long.values,site_data_synth.site_id,layer,synth_env_fn)

    return env_df, new_data_env, selected_env_data, metadata_env, nyears, old_years