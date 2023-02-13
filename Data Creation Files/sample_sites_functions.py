import numpy as np
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def sample_rand_radii(new_site_data,nrand_sites,n_gen):
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
