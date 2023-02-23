import pandas as pd
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