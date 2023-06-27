import pandas as pd
import numpy as np

from src.data_processing.preprocess_functions import (
    add_distances_conn_data,
)

# from sdv.tabular import TVAE

from src.models.GAN_model import GAN
from src.data_processing.postprocess_functions import anonymize_conn
from src.data_processing.sampling_functions import find_NN_conn_data
from src.data_processing.package_synth_data import (
    retrieve_synth_site_data_fp,
    retrieve_orig_site_data_fp,
    retrieve_orig_conn_fp,
    retrieve_synth_conn_data_fp,
)


### -----------------------------Load site data and connectivity data to synethesize----------------------------###
def connectivity_model(root_original_file, root_site_data_synth, years, num):
    synth_data_fn = retrieve_synth_site_data_fp(root_site_data_synth)
    orginal_site_data_fn = retrieve_orig_site_data_fp(root_original_file, ".csv")
    site_data = pd.read_csv(orginal_site_data_fn)
    site_data_synth = pd.read_csv(synth_data_fn)

    conn_data_store = pd.DataFrame(
        np.zeros((len(site_data.lat), len(site_data.lat))),
        columns=site_data["reef_siteid"],
    )

    # stack all replicates to be used
    for yr in years:
        for nn in num:
            original_conn_fn = retrieve_orig_conn_fp(root_original_file, yr, nn)
            conn_orig = pd.read_csv(original_conn_fn, skiprows=3)
            conn_orig.drop(conn_orig.columns[0], axis=1, inplace=True)
            conn_data_store = conn_data_store + conn_orig

    # add NS and EW tidal distances + lats and longs to training data
    conn_data_store, scaler, metadata_conn = add_distances_conn_data(
        conn_data_store, conn_orig, site_data
    )
    data_cols = conn_data_store.columns
    ### ---------------------------------------Train GAN model-------------------------------------------------------###

    # Define the GAN and training parameters
    noise_dim = 32
    dim = 128
    batch_size = 32

    log_step = 100
    epochs = 2000 + 1
    learning_rate = 5e-4
    models_dir = "model"

    gan_args = [batch_size, learning_rate, noise_dim, conn_data_store.shape[1], dim]
    train_args = ["", epochs, log_step]

    # run training to learn from data
    model = GAN

    # Training the GAN model
    synthesizer = model(gan_args)
    synthesizer.train(conn_data_store, train_args)
    # synthesizer.save('generator_connectivity')

    # look at generator and discriminator summary
    # synthesizer.generator.summary()
    # synthesizer.discriminator.summary()

    models = {"GAN": ["GAN", False, synthesizer.generator]}

    # Setup parameters visualization parameters
    seed = 17
    test_size = conn_data_store.shape[0]  # number of sites
    noise_dim = 32

    ### -----------------------------Sample data and transform to original data space--------------------------------###
    np.random.seed(seed)
    real = synthesizer.get_data_batch(
        train=conn_data_store, batch_size=test_size, seed=seed
    )
    real_samples = pd.DataFrame(real, columns=data_cols)
    conn_samples = pd.DataFrame(
        scaler.inverse_transform(real_samples[data_cols]), columns=data_cols
    )

    ### -------------------------------Select conn data closest to site data spatially-------------------------------###

    selected_conn_data = find_NN_conn_data(
        site_data_synth, conn_samples, conn_data_store
    )

    selected_conn_data = anonymize_conn(site_data_synth, selected_conn_data)
    synth_conn_fn = retrieve_synth_conn_data_fp(root_site_data_synth)

    selected_conn_data.to_csv(synth_conn_fn, index=False)

    return (
        conn_data_store,
        conn_samples,
        selected_conn_data,
        metadata_conn,
        synth_conn_fn,
    )
