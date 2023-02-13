import os
import pandas as pd
#import geopandas as gp
import numpy as np
import matplotlib.pyplot as plt

from table_evaluator import load_data, TableEvaluator
#from sdv.evaluation import evaluate
#from sdv.metrics.tabular import LogisticDetection

import preprocess_functions
import GAN_model
from sklearn.neighbors import NearestNeighbors
### ----------------Load site data and connectivity data to synethesize------------------###
data_set_folder = "Original Data"
synth_data_set_folder = "Synthetic Data"
conn_orig = pd.read_csv(
    data_set_folder
    + "\\Moore_2022-11-17\\connectivity\\2015\\connect_matrix_2015_3.csv",
    skiprows=3,
)
site_data = pd.read_csv(
    data_set_folder + "\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.csv"
)
site_data_synth = pd.read_csv(
    synth_data_set_folder + "\\site_data_" + data_set_folder + "_numsamps_29.csv"
)

breakpoint()
conn_data, conn_orig, scaler =  preprocess_functions.reprocess_conn_data(site_data,conn_orig)

### ----------------Train GAN model------------------###
# define the training parameters for the GAN network
data_cols = conn_data.columns
breakpoint()
# Define the GAN and training parameters
noise_dim = 32
dim = 128
batch_size = 32

log_step = 100
epochs = 5000 + 1
learning_rate = 5e-4
models_dir = "model"

# conn_data[data_cols] = conn_data[data_cols]

print(conn_data.shape[1])

gan_args = [batch_size, learning_rate, noise_dim, conn_data.shape[1], dim]
train_args = ["", epochs, log_step]

# run training to learn from data
model = GAN_model.GAN
breakpoint()
# Training the GAN model
synthesizer = model(gan_args)
breakpoint()
synthesizer.train(conn_data, train_args)

breakpoint()
# synthesizer.save('generator_connectivity_data_Moore')

# look at generator and discriminator summary
synthesizer.generator.summary()
synthesizer.discriminator.summary()

models = {"GAN": ["GAN", False, synthesizer.generator]}

breakpoint()
# Setup parameters visualization parameters
seed = 17
test_size = 200  # number of sites
noise_dim = 32
### ----------------Sample data and transform to original data space------------------###
np.random.seed(seed)
real = synthesizer.get_data_batch(train=conn_data, batch_size=test_size, seed=seed)
real_samples = pd.DataFrame(real, columns=data_cols)
conn_samples = pd.DataFrame(scaler.inverse_transform(real_samples[data_cols[1:]]), columns=data_cols[1:])
conn_data_full = pd.DataFrame(scaler.inverse_transform(conn_data[data_cols[1:]]), columns=data_cols[1:])
# ax = plt.imshow(conn_orig[conn_orig.keys()[1:20]][0:19] ,interpolation = 'nearest')
# plt.show()
# ax = plt.imshow(conn_samples[conn_samples.keys()[1:20]][0:19] ,interpolation = 'nearest')
# plt.show()

#evaluate(conn_samples, conn_data_full, metrics=["KSTest"])
#LogisticDetection.compute(conn_data_full, conn_samples)
breakpoint()
#table_evaluator = TableEvaluator(conn_data_full[conn_data_full.keys()[1:200]],conn_samples[conn_samples.keys()[1:200]],)
#table_evaluator.visual_evaluation()

breakpoint()

synth_lats = site_data_synth['lat']
synth_longs = site_data_synth['long']
samples = np.zeros((len(conn_samples.lats), 2))
site_data_vals = np.zeros((len(synth_lats), 2))

for l in range(len(conn_samples.lats)):
    samples[l][:] = [conn_samples.longs[l], conn_samples.lats[l]]

breakpoint()
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(samples)
breakpoint()

for k in range(len(synth_lats)):
    site_data_vals[k][:] = [-synth_lats[k], synth_longs[k]]
breakpoint()
nearest_sites = neigh.kneighbors(site_data_vals, return_distance=False)
nearest_site_inds = [nearest_sites[kk][0] for kk in range(len(nearest_sites))]
selected_conn_data = np.zeros(len(nearest_site_inds),len(nearest_site_inds))
conn_samples_array = conn_samples[conn_orig.columns[nearest_site_inds]].to_numpy()
selected_conn_data = conn_samples_array[nearest_site_inds,:]

