import pandas as pd
import geopandas as gp
from shapely.geometry import Point, Polygon
import numpy as np
import netCDF4 as nc

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

def generate_timestamp():
    """
    Generate current time and date as string for file names.

    """
    ts = dt.now()
    ts_string = str(ts.day)+"-"+str(ts.month)+"-"+str(ts.year)+"_"+str(ts.hour)+str(ts.minute)+str(ts.second)
    return ts_string

def anonymize_spatial(spatial):
    """
    Normalise latitude and longitude data to remove physical reference.

    :param dataframe spatial: contains columns 'lat'- latitudes and 'long' - longitudes.

    """
    scaler = MinMaxScaler().fit(spatial[['lat','long']])
    spatial[['lat','long']] = scaler.transform(spatial[['lat','long']])
    return spatial

def anonymize_conn(site_data_synth,conn_data_synth):
    """
    Anonymise connectivity data by replacing identifyable site names in rows and column names.

    :param dataframe site_data_synth: synthetic, anonymized site data.
    :param dataframe conn_data_synth: synthetic connectivity data.

    """
    conn_data_md = {'recieving_site':site_data_synth.site_id.values}
    columns = [str(sid) for sid in site_data_synth.site_id]
    for kk in range(len(columns)):
        conn_data_md[columns[kk]] = conn_data_synth[:,kk]

    conn_data_synth_df = pd.DataFrame(conn_data_md)
    return conn_data_synth_df

def convert_to_geo(site_data_synth):
    R = 6371008.7714 # assumed radius of Earth in m
    radius = R*np.arccos(1-site_data_synth.area/(2*np.pi*R**2))/(1000^2)
    lat = -1*site_data_synth.lat
    long = site_data_synth.long

    # Create circle centre points
    polygons = []
    for (x,y,r) in zip(long,lat,radius):

        s_temp = gp.GeoSeries([Point(x,y)])
        polygons.append(s_temp.buffer(r))

    return polygons

def create_env_nc(store_env,lats,longs,site_ids,layer,fn):
    """
    Save environmental data as a net cdf file for site data packaging.

    :param numpy array store_env: Synthetic data model for dhw or wave data.
    :param numpy array lats: lats for synthesized env data layer.    
    :param numpy array longs: longs for synthesized env data layer. 
    :param numpy array site_ids: anonymized site_ids for synthesized env data layer.
    :param str layer: indicates type of env data ('dhw' or 'wave').
    :param fn: filename for nc data set to be saved as.

    """
    ds = nc.Dataset(fn, 'w', format='NETCDF4')
    ds.createDimension('sites', len(site_ids))
    ds.createDimension('member',store_env.shape[0])
    ds.createDimension('timesteps', store_env.shape[2])

    longitude = ds.createVariable('longitude', 'f4', ('sites',))
    latitude = ds.createVariable('latitude', 'f4', ('sites',))
    reef_siteid = ds.createVariable('reef_siteid', 'f4', ('sites',))
    UNIQUE_ID = ds.createVariable('UNIQUE_ID', 'f4', ('sites',))
    dhw = ds.createVariable(layer, 'f4', ('member', 'sites', 'timesteps',))

    longitude[:] = lats
    latitude[:] = longs
    reef_siteid[:] = site_ids
    UNIQUE_ID[:] = site_ids
    dhw[:,:,:] = store_env
    ds.close()

def make_cover_array(cover_df):
    """
    Create array from synthetically generated coral cover dataframe to be packages as netcdf.

    :param dataframe cover_df: Synthetic coral cover dataframe.

    """

    sites = cover_df['lat'].unique()
    species = cover_df['species'].unique()
    store_cover = np.zeros((len(sites),len(species)))
    for si in range(len(sites)):
        store_cover[si,:] = cover_df['cover'][cover_df['lat']==sites[si]]

    return store_cover

def create_cover_nc(store_cover,fn):
    """
    Save cover data as a net cdf file for site data packaging.

    :param numpy array store_cover: Synthetic data array containing cover data (dims = (n_sites,n_species)).
    :param fn: filename for nc data set to be saved as.

    """
    n_sites = store_cover.shape[0]
    n_species = store_cover.shape[1]
    ds = nc.Dataset(fn, 'w', format='NETCDF4')
    ds.createDimension('reef_siteid',n_sites)
    ds.createDimension('species',n_species)

    covers = ds.createVariable('covers', 'f4', ('reef_siteid','species'))
    reef_siteid = ds.createVariable('reef_siteid', 'f4', ('reef_siteid',))

    covers[:,:] = store_cover
    reef_siteid[:] = [ii for ii in range(1,n_sites+1)]

    ds.close()