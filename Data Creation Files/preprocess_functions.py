import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_site_data(site_data_geo):
    """
    Preprocess site data into correct format for TVAE model.

    :param dataframe site_data_geo: contains site data with :geometry column.

    """
    site_data = site_data_geo[site_data_geo.columns[:-1]]
    site_data['long'] = site_data_geo.centroid.x
    site_data['lat'] = site_data_geo.centroid.y
    # get rid of identifying ids
    site_data = site_data.drop('reef_siteid', axis=1)
    site_ids = pd.DataFrame(
        {'site_id': [i for i in range(1, len(site_data['site_id'])+1)]})
    site_data = site_data.drop('site_id', axis=1)
    site_data = site_data.drop('UNIQUE_ID', axis=1)
    site_data = site_data.drop('Reef', axis=1)
    site_data = pd.concat([site_ids, site_data], axis=1)

    site_data['long'] = abs(site_data['long'])
    site_data['lat'] = abs(site_data['lat'])
    # define table metadata
    metadata = {'fields': {'site_id': {'type': 'id'},
                            'habitat': {'type': 'categorical'},
                            'k': {'type': 'numerical', 'subtype': 'float'},
                            'area': {'type': 'numerical', 'subtype': 'float'},
                            'rubble': {'type': 'numerical', 'subtype': 'float'},
                            'sand': {'type': 'numerical', 'subtype': 'float'},
                            'rock': {'type': 'numerical', 'subtype': 'float'},
                            'coral_algae': {'type': 'numerical', 'subtype': 'float'},
                            'na_proportion': {'type': 'numerical', 'subtype': 'float'},
                            'depth_mean': {'type': 'numerical', 'subtype': 'float'},
                            'depth_sd': {'type': 'numerical', 'subtype': 'float'},
                            'depth_med': {'type': 'numerical', 'subtype': 'float'},
                            'zone_type': {'type': 'categorical'},
                             'long': {'type': 'numerical', 'subtype': 'float'},
                              'lat': {'type': 'numerical', 'subtype': 'float'}},
                'primary_key':'site_id'}
    return site_data, metadata

def preprocess_dhw_data(dhw_data):
    """
    Preprocesses dhw into correct form for PAR model.

    :param numpy array dhw_data: contains original dhw data, loaded from nc file.

    """
    nyears, nsites, nreps = dhw_data['dhw'].shape
    size = nyears*nsites

    lats = dhw_data['latitude'][:]
    longs = dhw_data['longitude'][:]
    dhw_df = pd.DataFrame({"Year": [0]*size, "Lat": [0.0]*size,
                            "Long": [0.0]*size, "Site": [0]*size, "Dhw": [0.0]*size})

    count = 0

    for yr in range(nyears):
        for si in range(nsites):
            dhw_df['Year'][count] = str(yr+2025)
            dhw_df['Lat'][count] = lats[si]
            dhw_df['Long'][count] = longs[si]
            dhw_df['Site'][count] = si+1
            dhw_df['Dhw'][count] = dhw_data['dhw'][yr,si, np.random.randint(nreps-1)]
            count += 1

    data_types = {
        'Long': 'continuous',
        'Lat': 'continuous',
        'Site': 'categorical',
        'Dhw': 'continuous'
    }

    # create metadata dictionary
    metadata_dhw = {'fields': {'Year': {'type': 'datetime'},
                           'Lat': {'type': 'numerical', 'subtype': 'float'},
                           'Long': {'type': 'numerical', 'subtype': 'float'},
                           'Site': {'type': 'categorical'},
                           'Dhw': {'type': 'numerical', 'subtype': 'float'}},
                'context_columns': ['Site'],
                'entity_columns': ['Lat', 'Long'],
                'sequence_index': 'Year'}

    old_years = dhw_df["Year"]
    dhw_df["Year"] = pd.to_datetime(dhw_df["Year"], format='%Y')
    return dhw_df, data_types, metadata_dhw, old_years, nyears


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

def preprocess_conn_data(site_data,conn_orig):
    """
    Preprocesses connectivity data byt appending NS and ES tidal data.

    :param dataframe site_data: original site data.
    :param dataframe conn_orig: original connectivity data.

    """
    lats = site_data.lats
    longs = site_data.longs
    ### ----------------Preprocessing data and appending tidal distance matrices------------------###
    conn_orig.rename(columns={"Unnamed: 0": "recieving_site"}, inplace=True)
    conn_data = conn_orig
    rec_sites = conn_orig["recieving_site"]

    conn_data.drop(conn_data.columns[0], axis=1, inplace=True)

    conn_data = pd.DataFrame(conn_data, columns=conn_orig.columns)
    print(conn_data.isnull().values.any())
    conn_data.fillna(0)
    print(conn_data.isnull().values.any())

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

    rec_sites = rec_sites.astype("category").cat.codes

    conn_data = pd.concat([conn_data, lats, longs, east_west, north_south], axis=1)
    cols = conn_data.columns
    scaler = MinMaxScaler().fit(conn_data)
    conn_data = scaler.transform(conn_data)
    conn_data = pd.DataFrame(conn_data, columns=cols)
    conn_data = pd.concat([rec_sites, conn_data], axis=1)
    conn_data.rename(columns={0: "recieving_site"}, inplace=True)

    conn_fields = {kk : {'type':'numerical','subtype':'float'} for kk in conn_orig.columns}
    # create metadata dictionary
    metadata_conn = {'fields': conn_fields,
                'primary_key':'recieving_site'}
    return conn_data, conn_orig, scaler, metadata_conn