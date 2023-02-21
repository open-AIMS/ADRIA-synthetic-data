import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from table_evaluator import load_data, TableEvaluator
from sdmetrics.reports.single_table import QualityReport
#from sdv.evaluation import evaluate
#from sdv.metrics.tabular import LogisticDetection

import preprocess_functions
import GAN_model
from postprocess_functions import find_NN_conn_data, anonymize_conn
### -----------------------------Load site data and connectivity data to synethesize----------------------------###
original_conn_fn = "Original Data\\Moore_2022-11-17\\connectivity\\2015\\connect_matrix_2015_3.csv"
orginal_site_data_fn= "Original Data\\Moore_2022-11-17\\site_data\\Moore_2022-11-17.csv"
synth_data_fn = "Synthetic Data\\site_data_Original Data_numsamps_29.csv"

conn_orig = pd.read_csv(original_conn_fn,skiprows=3)
site_data = pd.read_csv(orginal_site_data_fn)
site_data_synth = pd.read_csv(synth_data_fn)

conn_data, conn_orig, scaler, metadata_conn =  preprocess_functions.preprocess_conn_data(site_data,conn_orig)
breakpoint()
### ---------------------------------------Train GAN model-------------------------------------------------------###
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
# Training the GAN model
synthesizer = model(gan_args)
synthesizer.train(conn_data, train_args)
# synthesizer.save('generator_connectivity')

# look at generator and discriminator summary
synthesizer.generator.summary()
synthesizer.discriminator.summary()

models = {"GAN": ["GAN", False, synthesizer.generator]}

breakpoint()
# Setup parameters visualization parameters
seed = 17
test_size = 200  # number of sites
noise_dim = 32

### -----------------------------Sample data and transform to original data space--------------------------------###
np.random.seed(seed)
real = synthesizer.get_data_batch(train=conn_data, batch_size=test_size, seed=seed)
real_samples = pd.DataFrame(real, columns=data_cols)
conn_samples = pd.DataFrame(scaler.inverse_transform(real_samples[data_cols[1:]]), columns=data_cols[1:])
conn_data_full = pd.DataFrame(scaler.inverse_transform(conn_data[data_cols[1:]]), columns=data_cols[1:])

### ------------------------------------Test synthetic data utility--------------------------------------------###
report = QualityReport()
report.generate(conn_orig, conn_samples[conn_orig.columns], metadata_conn)
report.get_details(property_name='Column Shapes')
report.get_details(property_name='Pair Trends')
#evaluate(conn_samples, conn_data_full)
#LogisticDetection.compute(conn_data_full, conn_samples)
breakpoint()
### -------------------------------Select conn data closest to site data spatially-------------------------------###

selected_conn_data = find_NN_conn_data(site_data_synth,conn_samples,conn_orig)
selected_conn_data = anonymize_conn(site_data_synth,selected_conn_data)
synth_conn_fn = "Synthetic Data\\Synthetic Data Packages\\"+synth_data_fn.split("\\")[1][9:-3]
+"connectivity\\2000\\conn_data"+synth_data_fn.split("\\")[1][9:-3]+"csv"

selected_conn_data.to_csv(synth_conn_fn)
