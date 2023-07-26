import pandas as pd

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


from src.data_processing.preprocess_functions import preprocess_site_data
from src.data_processing.sampling_functions import sample_rand_radii
from src.data_processing.postprocess_functions import (
    anonymize_spatial,
    generate_timestamp,
    convert_to_geo,
)
from src.data_processing.package_synth_data import (
    initialize_data_package,
    retrieve_orig_site_data_fp,
    create_synth_site_data_fp,
    create_synth_site_data_package_fp,
)


def site_data_model(orig_data_package, N, N2, N3):
    ### --------------------------------------Load site data to synethesize--------------------------------------###

    site_data_fn = retrieve_orig_site_data_fp(orig_data_package, ".csv")
    site_data = pd.read_csv(site_data_fn, index_col=False)

    ### ---------------------------------Preprocess data for sdv model fit----------------------------------###
    # simplify to dataframe
    site_data = preprocess_site_data(site_data)

    ### -----------------------------------Fit GaussianCopula model for site data-------------------------------------###
    # set up GaussianCopula, fit
    metadata_site = SingleTableMetadata()
    metadata_site.detect_from_dataframe(data=site_data)
    metadata_site.update_column(column_name="site_id", sdtype="id")
    metadata_site.set_primary_key(column_name="site_id")

    model = GaussianCopulaSynthesizer(
        metadata_site,
        enforce_min_max_values=True,
        enforce_rounding=False,
        default_distribution="gaussian_kde",
    )

    model.fit(site_data)

    ### ----------------------------------------Sample data and test utility-----------------------------------------###
    # create sample data
    new_site_data = model.sample(num_rows=N)

    ### ----------------Re-sample using conditional sampling to emulated site spatial clustering------------------###
    conditions = sample_rand_radii(new_site_data, N3, N2)
    sample_sites = model.sample_remaining_columns(conditions)

    ### ------------------------------------------Save site data to csv-------------------------------------------###
    time_stamp = generate_timestamp()
    sample_sites["lat"] = -1 * sample_sites["lat"]

    sample_sites["reef_siteid"] = [
        "reef_" + str(k) for k in range(1, len(sample_sites["site_id"]) + 1)
    ]
    sample_sites_fn = create_synth_site_data_fp(time_stamp)
    sample_sites.to_csv(sample_sites_fn, index=False)
    sample_sites_geo = convert_to_geo(sample_sites)
    sample_sites_anon = anonymize_spatial(sample_sites_geo)

    initialize_data_package(time_stamp)
    synth_site_data_fn = create_synth_site_data_package_fp(time_stamp)
    sample_sites_anon.to_file(synth_site_data_fn, driver="GPKG", index=False)

    return site_data, new_site_data, sample_sites, metadata_site, synth_site_data_fn
