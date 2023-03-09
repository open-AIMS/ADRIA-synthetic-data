import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors

def sample_rand_radii(new_site_data, nrand_sites, n_gen):
    """
   Calculate new site lats and longs within randomised 
   radius, within the same domain as old sites.

   :param dataframe new_site_data: synthetic site data.
   :param int nrand_sites: number of site positions to generate.
   :param int n_gen: number of sites to generate around site positions.
   """
    rand_sites = np.unique(np.random.randint(0, len(new_site_data.lat), size=(1, nrand_sites))[0])
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
        
        site = np.unique(np.random.randint(0, len(rand_sites)-1, size=(1, 1))[0])
        rlat_temp = lats[rand_sites[site]] + rand_radii[rr] * np.cos(rand_theta[rr])

        if rlat_temp[rand_sites[site][0]] < max_lat and rlat_temp[rand_sites[site][0]] > min_lat:
            rand_lats.append(rlat_temp[rand_sites[site][0]])
        rlong_temp = longs[rand_sites[site]] + rand_radii[rr] * np.sin(rand_theta[rr])

        if rlong_temp[rand_sites[site][0]] < max_long and rlong_temp[rand_sites[site][0]] > min_long:
            rand_longs.append(rlong_temp[rand_sites[site][0]])

    nsites_samp = min(len(rand_lats), len(rand_longs))
    conditions = pd.DataFrame(
        {'lat': rand_lats[0:nsites_samp], 'long': rand_longs[0:nsites_samp]})

    return conditions

def create_cover_conditional_struct(site_data_synth, nspecies):
    size = nspecies*site_data_synth.shape[0]
    conditions = pd.DataFrame({"species": [0]*size,"lat": [0.0]*size,"long": [0.0]*size})
    count = 0
    for sp in range(nspecies):
        for si in range(site_data_synth.shape[0]):
            conditions['species'][count] = sp+1
            conditions['lat'][count] = -1*site_data_synth['lat'][si]
            conditions['long'][count] = site_data_synth['long'][si]
            count+=1
    return conditions

def sample_env_ensemble(model,context,nsamples,nsites,nyears,layer):
    """
    Sample synthetic environmental data using conditional model to create nsamples*nsites*ntimesteps array,
    which can be saved as a netCDF file.

    :param SDV model model: Synthetic data model for dhw or wave data.
    :param dict context: Lats and longs, synthesized for synthetic site data, to conditionalise the model on.
    :param int nsamples: number of samples to take.
    :param int nsites: number of sites in the model.
    :param nyears: number of years simulated by model.
    :param str: indicates the data layer ('dhw' or 'wave')

    """
    store_env = np.zeros([nsamples,nsites,nyears])
    for ss in range(nsamples):
        sample_temp = model.sample(context=context)
        for si in range(nsites):
            store_env[ss,si,:] = sample_temp[layer][sample_temp['site']==si]

    return store_env

    
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

    synth_lats = site_data_synth['lat']
    synth_longs = site_data_synth['long']
    samples = np.zeros((len(conn_samples.lat), 2))
    site_data_vals = np.zeros((len(synth_lats), 2))
    
    for l in range(len(conn_samples.lat)):
        samples[l][:] = [conn_samples.long[l], conn_samples.lat[l]]

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
