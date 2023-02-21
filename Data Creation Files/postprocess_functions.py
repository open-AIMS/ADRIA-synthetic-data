import numpy as np
import math
import pandas as pd
import netCDF4 as nc

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

def sample_rand_radii(new_site_data, nrand_sites, n_gen):
    """
   Calculate new site lats and longs within randomised 
   radius, within the same domain as old sites.

   :param dataframe new_site_data: synthetic site data.
   :param int nrand_sites: number of site positions to generate.
   :param int n_gen: number of sites to generate around site positions.
   """
    rand_sites = np.random.randint(
        0, len(new_site_data.lat), size=(1, nrand_sites))[0]
    max_lat = max(new_site_data.lat)
    min_lat = min(new_site_data.lat)
    max_long = max(new_site_data.long)
    min_long = min(new_site_data.long)
    lats = new_site_data.lat[rand_sites]
    longs = new_site_data.long[rand_sites]

    rand_lats = []
    rand_longs = []
    R = 0.01
    # generate random radii and theta around these points
    rand_radii = np.sqrt(np.random.uniform(0, 1, size=(1, n_gen))[0])*R
    rand_theta = 2*math.pi*np.random.uniform(0, 1, size=(1, n_gen))[0]

    for rr in range(len(rand_radii)):
        site = np.random.randint(0, nrand_sites-1, size=(1, 1))[0]
        rlat_temp = lats[rand_sites[site]] + \
            rand_radii[rr] * np.cos(rand_theta[rr])
        if rlat_temp[rand_sites[site][0]] < max_lat and rlat_temp[rand_sites[site][0]] > min_lat:
            rand_lats.append(rlat_temp[rand_sites[site][0]])
        rlong_temp = longs[rand_sites[site]] + \
            rand_radii[rr] * np.sin(rand_theta[rr])
        if rlong_temp[rand_sites[site][0]] < max_long and rlong_temp[rand_sites[site][0]] > min_long:
            rand_longs.append(rlong_temp[rand_sites[site][0]])

    nsites_samp = min(len(rand_lats), len(rand_longs))
    conditions = pd.DataFrame(
        {'lat': rand_lats[0:nsites_samp], 'long': rand_longs[0:nsites_samp]})

    return conditions

def find_NN_dhw_data(site_data_synth,new_data_dhw,nyears):
    """
   Find closest neighbours in synthetic dhw data to synthetic site_data.

   :param dataframe site_data_synth: synthetic site data.
   :param dataframe new_data_dhw: synthetic dhw data.
   :param int n_gen: number of sites to generate around site positions.
   """
    synth_lats = site_data_synth['lat']
    synth_longs = site_data_synth['long']
    samples = np.zeros((len(new_data_dhw['Lat']), 2))
    site_data_vals = np.zeros((len(synth_lats), 2))

    for l in range(len(new_data_dhw['Lat'])):
        samples[l][:] = [new_data_dhw['Lat'][l], new_data_dhw['Long'][l]]

    neigh = NearestNeighbors(n_neighbors=nyears)
    neigh.fit(samples)

    for k in range(len(synth_lats)):
        site_data_vals[k][:] = [-synth_lats[k], synth_longs[k]]

    nearest_sites = neigh.kneighbors(site_data_vals, return_distance=False)
    selected_dhws = np.zeros((len(nearest_sites), nyears))
    for nn in range(len(nearest_sites)):

        nearest_sites[nn].sort()
        selected_dhws[nn, :] = new_data_dhw['Dhw'][nearest_sites[nn]]

    return selected_dhws

def find_NN_conn_data(site_data_synth,conn_samples,conn_orig):
    """
    Find closest neighbours in synthetic connectivity data to synthetic site_data.

    :param dataframe site_data_synth: synthetic site data.
    :param dataframe conn_samples: synthetic connectivity data.
    :param dataframe conn_orig: original connectivity data.
    """

    breakpoint()
    synth_lats = site_data_synth['lat']
    synth_longs = site_data_synth['long']
    samples = np.zeros((len(conn_samples.lats), 2))
    site_data_vals = np.zeros((len(synth_lats), 2))
    
    for l in range(len(conn_samples.lats)):
        samples[l][:] = [conn_samples.longs[l], conn_samples.lats[l]]

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)

    for k in range(len(synth_lats)):
        site_data_vals[k][:] = [-synth_lats[k], synth_longs[k]]

    nearest_sites = neigh.kneighbors(site_data_vals, return_distance=False)
    nearest_site_inds = [nearest_sites[kk][0] for kk in range(len(nearest_sites))]
    selected_conn_data = np.zeros([len(nearest_site_inds),len(nearest_site_inds)],dtype=float)
    conn_samples_array = conn_samples[conn_orig.columns[nearest_site_inds]].to_numpy()
    selected_conn_data = conn_samples_array[nearest_site_inds,:]

    return selected_conn_data

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

def sample_dhw_ensemble(model,context,nsamples,nsites,nyears):
    """
    Sample synthetic dhw data using conditional model to create nsamples*nsites*ntimesteps array,
    which can be saved as a netCDF file.

    :param SDV model model: Synthetic data model for dhw.
    :param dict context: Lats and longs, synthesized for synthetic site data, to conditionalise the dhw model on.
    :param int nsamples: number of samples to take.
    :param int nsites: number of sites in the model.
    :param nyears: number of years simulated by model.

    """
    store_dhws = np.zeros(nsamples,nsites,nyears)
    for ss in range(nsamples):
        sample_temp = model.sample(context=context)
        for si in range(nsites):
            store_dhws[ss,si,:] = sample_temp['Dhw'][sample_temp['Site']==si]

    return store_dhws

def create_dhw_nc(store_dhws,lats,longs,site_ids,fn):
    """
    Save dhw data as a net cdf file for site data packaging.

    :param numpy array store_dhws: Synthetic data model for dhw.
    :param dict context: Lats and longs, synthesized for synthetic site data, to conditionalise the dhw model on.
    :param int nsamples: number of samples to take.
    :param int nsites: number of sites in the model.
    :param nyears: number of years simulated by model.

    """
    ds = nc.Dataset(fn, 'w', format='NETCDF4')
    ds.createDimension('sites', len(site_ids))
    ds.createDimension('member',store_dhws.shape[0])
    ds.createDimension('timesteps', store_dhws.shape[2])

    longitude = ds.createVariable('longitude', 'f4', ('sites',))
    latitude = ds.createVariable('latitude', 'f4', ('sites',))
    reef_siteid = ds.createVariable('reef_siteid', 'f4', ('sites',))
    UNIQUE_ID = ds.createVariable('UNIQUE_ID', 'f4', ('sites',))
    dhw = ds.createVariable('dhw', 'f4', ('member', 'sites', 'timesteps',))

    longitude[:] = lats
    latitude[:] = longs
    reef_siteid[:] = site_ids
    UNIQUE_ID[:] = site_ids
    dhw[:,:,:] = store_dhws
    ds.close()