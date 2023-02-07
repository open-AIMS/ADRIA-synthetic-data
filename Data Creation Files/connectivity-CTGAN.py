import os
import pandas as pd
import geopandas as gp
import numpy as np

# from sdv.tabular import CTGAN
import ctgan
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sdv.evaluation import evaluate

breakpoint()
data_set_folder = "Original Data"
synth_data_set_folder = "Synthetic Data"
conn_orig = pd.read_csv(
    data_set_folder
    + "\\Moore_2022-11-17\\connectivity\\2015\\connect_matrix_2015_3.csv",
    skiprows=3,
)
site_data_geo = gp.read_file(
    data_set_folder + "\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.gpkg"
)
site_data_synth = pd.read_csv(
    synth_data_set_folder + "\\site_data_" + data_set_folder + "_numsamps_29.csv"
)

breakpoint()

site_data = site_data_geo[site_data_geo.columns[:-1]]
lats = site_data_geo.centroid.x
longs = site_data_geo.centroid.y


# tidal distance function
def tide_dist(latlong_site1, latlong_site2):
    a2 = latlong_site1 * np.pi / 180  # convert to radians
    b2 = latlong_site2 * np.pi / 180
    a = (np.sin((b2 - a2) / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6378.145  # radius of the earth
    d = R * c
    return d

    ### ----------------Preprocessing data and appending tidal distance matrices------------------###


conn_orig.rename(columns={"Unnamed: 0": "recieving_site"}, inplace=True)
conn_data = conn_orig
rec_sites = conn_orig["recieving_site"]

conn_data.drop(conn_data.columns[0], axis=1, inplace=True)
scaler_conn = MinMaxScaler().fit(conn_data)
conn_data = scaler_conn.transform(conn_data)
conn_data = pd.DataFrame(conn_data, columns=conn_orig.columns)
print(conn_data.isnull().values.any())
conn_data.fillna(0)
print(conn_data.isnull().values.any())

breakpoint()

east_west = np.zeros(conn_data.shape)
north_south = np.zeros(conn_data.shape)

for i in range(conn_data.shape[0]):
    for j in range(conn_data.shape[0]):
        east_west[i, j] = tide_dist(longs[i], longs[j])
        north_south[i, j] = tide_dist(lats[i], lats[j])

breakpoint()
east_west_cols = [n for n in conn_orig.columns + "_EW"]
east_west = pd.DataFrame(east_west, columns=east_west_cols)
breakpoint()

north_south_cols = [n for n in conn_orig.columns + "_NS"]
north_south = pd.DataFrame(north_south, columns=north_south_cols)

rec_sites = rec_sites.astype("category").cat.codes
breakpoint()

conn_data = pd.concat([conn_data, lats, longs, east_west, north_south], axis=1)
conn_data.columns = conn_data.columns.astype(str)
cols = conn_data.columns
breakpoint()
scaler = MinMaxScaler().fit(conn_data)
conn_data = scaler.transform(conn_data)
conn_data = pd.DataFrame(conn_data, columns=cols)
conn_data = pd.concat([rec_sites, conn_data], axis=1)
conn_data.rename(columns={0: "recieving_site"}, inplace=True)
breakpoint()
discrete_columns = ["recieving_site"]
ctgan_mod = ctgan.CTGAN(batch_size=50, epochs=5, verbose=False)
ctgan_mod.fit(conn_data, discrete_columns)
# model = CTGAN(primary_key="recieving_site")
conn_sample = ctgan_mod.sample(conn_data.shape[0])
conn_sample_trans = pd.DataFrame(
    scaler.inverse_transform(conn_sample[cols]), columns=cols
)

breakpoint()
evaluate(conn_sample_trans[cols[1:216]], conn_orig)
