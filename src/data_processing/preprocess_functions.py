import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_site_data(site_data):
    """
    Preprocess site data into correct format for TVAE model.

    :param dataframe site_data_geo: contains site data with :geometry column.

    """
    # get rid of identifying ids
    site_data = site_data.drop("reef_siteid", axis=1)
    site_ids = pd.DataFrame(
        {"site_id": [i for i in range(0, len(site_data["site_id"]))]}
    )
    site_data = site_data.drop("site_id", axis=1)
    site_data = site_data.drop("UNIQUE_ID", axis=1)
    site_data = site_data.drop("Reef", axis=1)
    site_data = pd.concat([site_ids, site_data], axis=1)

    site_data["long"] = abs(site_data["long"])
    site_data["lat"] = abs(site_data["lat"])

    return site_data


def convert_to_csv(site_data_geo, csv_fn):
    """
    Convert geo site data to csv for use with the connectivity model (environment incompatible with geopandas).

    :param dataframe site_data_geo: contains site data with :geometry column.
    :param str csv_fn: file name for new site data csv.

    """
    site_data = site_data_geo[site_data_geo.columns[:-1]]
    site_data["long"] = site_data_geo.centroid.x
    site_data["lat"] = site_data_geo.centroid.y
    site_data.to_csv(csv_fn[:-4] + "csv", index=False)

    return site_data


def initialize_env_data(layer):
    """
    Initializes metsdata for PAR model.

    :param str layer: string indicating whether the layer is 'dhw' data or 'wave' data.

    """

    data_types = {
        "long": "continuous",
        "lat": "continuous",
        "site": "categorical",
        layer: "continuous",
    }

    # create metadata dictionary
    metadata_env = {
        "fields": {
            "year": {"type": "datetime"},
            "lat": {"type": "numerical", "subtype": "float"},
            "long": {"type": "numerical", "subtype": "float"},
            "site": {"type": "categorical"},
            layer: {"type": "numerical", "subtype": "float"},
        },
        "context_columns": ["site"],
        "entity_columns": ["lat", "long"],
        "sequence_index": "year",
    }

    return data_types, metadata_env


def preprocess_env_data(env_data, layer, nyears, nsites, replicates):
    size = nyears * nsites * len(replicates)

    lats = env_data["latitude"][:]
    longs = env_data["longitude"][:]
    sites = np.array(range(1, nsites + 1))

    env_df = pd.DataFrame(
        {
            "year": [0] * size,
            "lat": [0.0] * size,
            "long": [0.0] * size,
            "site": [0] * size,
            "rep": [0] * size,
            layer: [0.0] * size,
        }
    )

    count = 0
    for rep in replicates:
        for yr in range(nyears):
            env_df["year"][count : count + nsites] = pd.to_datetime(
                str(yr + 2025), format="%Y"
            )
            env_df["lat"][count : count + nsites] = lats
            env_df["long"][count : count + nsites] = longs
            env_df["site"][count : count + nsites] = sites
            env_df["rep"][count : count + nsites] = rep
            env_df[layer][count : count + nsites] = env_data[layer][rep - 1, :, yr]
            count += nsites

    old_years = env_df["year"]

    return env_df, old_years, nyears


def preprocess_cover_data(cc_data, site_data):
    """
    Preprocesses coral cover into correct form for tabular model.

    :param numpy array cc_data: contains original dhw or wave data, loaded from nc file.

    """
    nsites, nspecies = cc_data["covers"].shape
    ngroups = int((nspecies/6))
    size =  ngroups * nsites

    cc_df = pd.DataFrame(
        {
            "site_id": [0] * size,
            "species": [0] * size,
            "lat": [0.0] * size,
            "long": [0.0] * size,
            "cover": [0.0] * size,
        }
    )

    count = 0

    for si in range(nsites):
        for sp in range(ngroups):
            cc_df["site_id"][count] = si + 1
            cc_df["lat"][count] = site_data["lat"][si]
            cc_df["long"][count] = site_data["long"][si]
            cc_df["species"][count] = sp + 1
            cc_df["cover"][count] = sum(cc_data["covers"][si, int(sp*6):int(sp*6)+5])
            count += 1

    # create metadata dictionary
    metadata_cc = {
        "fields": {
            "site_id": {"type": "id"},
            "species": {"type": "categorical"},
            "lat": {"type": "numerical", "subtype": "float"},
            "long": {"type": "numerical", "subtype": "float"},
            "cover": {"type": "numerical", "subtype": "float"},
        },
        "primary_key": "site_id",
    }

    return cc_df, metadata_cc


def tide_dist(latlong_site1, latlong_site2):
    """
    Calculates matrix of tital distances given input lat and long.
    (used to complement connectivity data).

    :param vec latlong_site1: [lat, long] for first site.
    :param vec latlong_site2: [lat, long] for 2nd site.

    """
    a2 = latlong_site1 * np.pi / 180  # convert to radians
    b2 = latlong_site2 * np.pi / 180
    a = (np.sin((b2 - a2) / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6378.145  # radius of the earth
    d = R * c
    return d


def add_distances_conn_data(conn_data, conn_orig, site_data):
    lats = site_data.lat
    longs = site_data.long
    east_west = np.zeros(conn_data.shape)
    north_south = np.zeros(conn_data.shape)

    for i in range(conn_data.shape[0]):
        for j in range(conn_data.shape[0]):
            east_west[i, j] = tide_dist(longs[i], longs[j])
            north_south[i, j] = tide_dist(lats[i], lats[j])

    east_west_cols = [n for n in conn_orig.columns + "_EW"]
    east_west = pd.DataFrame(east_west, columns=east_west_cols)

    north_south_cols = [n for n in conn_orig.columns + "_NS"]
    north_south = pd.DataFrame(north_south, columns=north_south_cols)

    conn_data = pd.concat([conn_data, lats, longs, east_west, north_south], axis=1)
    cols = conn_data.columns
    scaler = MinMaxScaler().fit(conn_data)
    conn_data = scaler.transform(conn_data)
    conn_data = pd.DataFrame(conn_data, columns=cols)

    conn_fields = {
        kk: {"type": "numerical", "subtype": "float"} for kk in conn_orig.columns
    }
    # create metadata dictionary
    metadata_conn = {"fields": conn_fields, "primary_key": "recieving_site"}
    return conn_data, scaler, metadata_conn

    # lats = site_data.lat
    # longs = site_data.long
    # ### ----------------Preprocessing data and appending tidal distance matrices------------------###
    # conn_orig.rename(columns={"Unnamed: 0": "recieving_site"}, inplace=True)
    # conn_data = conn_orig
    # rec_sites = conn_orig["recieving_site"]
    # conn_data.drop(conn_data.columns[0], axis=1, inplace=True)

    # count = 0
    # for lat in range(len(lats)):
    #     for long in range(len(longs)):
    #         conn_data_store["lat"][count] = lats[lat]
    #         conn_data_store["long"][count] = longs[long]
    #         conn_data_store["conn"][count] = conn_data[conn_data.columns[long]][lat]
    #         count += 1

    # conn_data = pd.DataFrame(conn_data, columns=conn_orig.columns)
