import pandas as pd
import numpy as np

from deepecho import PARModel
#from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata
import netCDF4

from src.data_processing.preprocess_functions import (
    preprocess_env_data,
    initialize_env_data,
)
from src.data_processing.sampling_functions import sample_env_ensemble
from src.data_processing.postprocess_functions import create_env_nc
from src.data_processing.package_synth_data import (
    retrieve_orig_env_fp,
    retrieve_synth_site_data_fp,
    retrieve_synth_env_data_fp,
)


### ----------------------------------Load site data and env data to synethesize----------------------------------###
def env_data_model(
    root_original_file, root_site_data_synth, nsamples, replicates, rcp, layer
):
    original_data_fn = retrieve_orig_env_fp(root_original_file, rcp, layer)
    synth_data_fn = retrieve_synth_site_data_fp(root_site_data_synth)

    ENV = netCDF4.Dataset(original_data_fn, "r")
    site_data_synth = pd.read_csv(synth_data_fn)

    ### --------------------------------------Reshape data to fit PAR model------------------------------------------###
    data_types, metadata_env = initialize_env_data(layer)
    nreps, nsites, nyears = ENV[layer].shape
    reps = len(replicates)
    store_env_synth = np.zeros([nyears, len(site_data_synth.site_id), nsamples * reps])

    env_df, old_years, nyears = preprocess_env_data(
        ENV, layer, nyears, nsites, replicates
    )

    total_reps = 0
    ### ----------------------------------------Set up and fit PAR model---------------------------------------------###
    for rep in replicates:
        temp_env_df = env_df[env_df["rep"] == rep]
        model = PARModel(epochs=1024, cuda=False)
        # metadata_env = SingleTableMetadata()
        # metadata_env.detect_from_dataframe(data=temp_env_df.loc[:, temp_env_df.columns != "rep"])
        # metadata_env.set_sequence_index(column_name='year')
        # metadata_env.update_column(column_name='site',sdtype='id')
        # metadata_env.set_sequence_key(column_name='site')
        # breakpoint()
        # model = PARSynthesizer(metadata_env, epochs=1024, cuda=False, context_columns=["lat", "long"])

        model.fit(data=temp_env_df.loc[:, temp_env_df.columns != "rep"],
            context_columns=["lat", "long"],
            entity_columns=["site"],
            data_types=data_types,
            sequence_index="year",
        )

        ### -----------------------------------------Sample data to synthesize--------------------------------------------###
        N_s = 100
        new_data_env = model.sample(N_s)
        new_data_env[new_data_env[layer] < 0.0] = 0.0
        new_years = [str(yr + 2025) for yr in range(nyears)]
        new_years = new_years * N_s
        new_data_env["year"] = new_years

        for rr in range(nsamples):
            ### -------------------------Sample synthetic env data at synthetic site data locations --------------------------###
            lat = site_data_synth.lat.values
            long = site_data_synth.long.values
            data = {"lat": lat, "long": long}
            context = pd.DataFrame(data)
            selected_env_data = model.sample(context=context)

            ### -------------------------------------Save sample data in sitedata package-------------------------------------###
            selected_years = [str(yr + 2025) for yr in range(nyears)]
            selected_years = selected_years * len(site_data_synth.lat.values)
            selected_env_data.insert(4, "year", selected_years, True)

            nsites = len(site_data_synth.lat.values)

            for si in range(nsites):
                store_env_synth[:, si, total_reps] = selected_env_data[layer][
                    selected_env_data["site"] == si
                ]

            total_reps += 1

    store_env_synth[store_env_synth < 0] = 0.0

    storage_env = np.zeros((np.product(store_env_synth.shape), 5))
    synth_selected_df = pd.DataFrame(
        storage_env, columns=["lat", "long", "site", layer, "year"]
    )

    lats = site_data_synth["lat"]
    longs = site_data_synth["long"]
    sites = [int(si) for si in range(1, nsites + 1)]
    count = 0
    store_env_synth[store_env_synth < 0] = 0
    for rep in range(reps * nsamples):
        for yr in range(nyears):
            synth_selected_df["lat"][count : count + nsites] = lats
            synth_selected_df["long"][count : count + nsites] = longs
            synth_selected_df["site"][count : count + nsites] = sites
            synth_selected_df["year"][count : count + nsites] = int(yr + 2025)
            synth_selected_df[layer][count : count + nsites] = store_env_synth[
                yr, :, rep
            ]
            count += nsites

    synth_env_fn = retrieve_synth_env_data_fp(root_site_data_synth, layer, rcp)

    create_env_nc(
        store_env_synth,
        site_data_synth.lat.values,
        site_data_synth.long.values,
        site_data_synth.site_id,
        layer,
        synth_env_fn,
    )

    return (
        env_df,
        new_data_env,
        synth_selected_df,
        metadata_env,
        nyears,
        old_years,
    )
