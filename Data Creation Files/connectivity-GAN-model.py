import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras import Model
from keras.optimizers import Adam

from table_evaluator import load_data, TableEvaluator
from sdv.evaluation import evaluate
from sdv.metrics.tabular import LogisticDetection

from scipy import interpolate

### ----------------Load site data and connectivity data to synethesize------------------###
conn_orig = pd.read_csv(
    'C:/Users/rcrocker/Documents/datasets/data_packages/Moore/connectivity/2016/moore_d3_2016_transfer_probability_matrix_wide.csv')
site_data = pd.read_csv(
    "C:/Users/rcrocker/Documents/datasets/data_packages/Moore/site_data/MooreReefCluster_Spatial_w4.5covers.csv")
# conn_orig = pd.read_csv("example_connectivity_Moore.csv")
# site_data = pd.read_csv("example_site_data_Moore.csv")
site_data_synth = pd.read_csv("site_data_Moore_numsamps_28.csv")

breakpoint()
lats = site_data['lat']
longs = site_data['long']


# tidal distance function
def tide_dist(latlong_site1, latlong_site2):

    a2 = latlong_site1 * np.pi/180  # convert to radians
    b2 = latlong_site2 * np.pi/180
    a = (np.sin((b2-a2)/2))**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6378.145  # radius of the earth
    d = R * c
    return d


### ----------------Preprocessing data and appending tidal distance matrices------------------###
conn_data = conn_orig
rec_sites = conn_orig['receiving_site']
conn_data.drop(conn_data.columns[0], axis=1, inplace=True)
# scaler_conn = MinMaxScaler().fit(conn_data)
# conn_data = scaler_conn.transform(conn_data)
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
east_west_cols = [n for n in conn_orig.columns + '_EW']
east_west = pd.DataFrame(east_west, columns=east_west_cols)
breakpoint()

north_south_cols = [n for n in conn_orig.columns + '_NS']
north_south = pd.DataFrame(north_south, columns=north_south_cols)

rec_sites = rec_sites.astype('category').cat.codes
breakpoint()

conn_data = pd.concat([conn_data, lats, longs, east_west, north_south], axis=1)
cols = conn_data.columns
scaler = MinMaxScaler().fit(conn_data)
conn_data = scaler.transform(conn_data)
conn_data = pd.DataFrame(conn_data, columns=cols)
conn_data = pd.concat([rec_sites, conn_data], axis=1)
conn_data.rename(columns={0: 'recieving_site'}, inplace=True)
breakpoint()

### ----------------GAN model definitions------------------###
# define the GAN model (taken from GitHub repo :https://github.com/ydataai/ydata-synthetic )


class GAN():

    def __init__(self, gan_args):
        [self.batch_size, lr, self.noise_dim,
         self.data_dim, layers_dim] = gan_args

        self.generator = Generator(self.batch_size).\
            build_model(input_shape=(self.noise_dim,),
                        dim=layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size).\
            build_model(input_shape=(self.data_dim,), dim=layers_dim)

        optimizer = Adam(lr, 0.5)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(record)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - some samples will have excessively low or high sampling, but easy to implement
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly

        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        # wasteful to shuffle every time
        train_ix = np.random.choice(
            list(train.index), replace=False, size=len(train))
        # duplicate to cover ranges past the end of the set
        train_ix = list(train_ix) + list(train_ix)
        x = train.loc[train_ix[start_i: stop_i]].values
        return np.reshape(x, (batch_size, -1))

    def train(self, data, train_arguments):
        [cache_prefix, epochs, sample_interval] = train_arguments

        data_cols = data.columns

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            batch_data = self.get_data_batch(data, self.batch_size)
            noise = tf.random.normal((self.batch_size, self.noise_dim))

            # Generate a batch of new images
            gen_data = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(batch_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = tf.random.normal((self.batch_size, self.noise_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated events
            if epoch % sample_interval == 0:
                # Test here data generation step
                # save model checkpoints
                model_checkpoint_base_name = 'model/' + \
                    cache_prefix + '_{}_model_weights_step_{}.h5'
                self.generator.save_weights(
                    model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(
                    model_checkpoint_base_name.format('discriminator', epoch))

                # Here is generating the data
                z = tf.random.normal((432, self.noise_dim))
                gen_data = self.generator(z)
                print('generated_data')

    def save(self, path, name):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return

    def load(self, path):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator


class Generator():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim, activation='relu')(input)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dense(dim * 4, activation='relu')(x)
        x = Dense(data_dim)(x)
        return Model(inputs=input, outputs=x)


class Discriminator():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4, activation='relu')(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=x)


### ----------------Train GAN model------------------###
# define the training parameters for the GAN network
data_cols = conn_data.columns
breakpoint()
# Define the GAN and training parameters
noise_dim = 32
dim = 128
batch_size = 32

log_step = 100
epochs = 5000+1
learning_rate = 5e-4
models_dir = 'model'

conn_data[data_cols] = conn_data[data_cols]

print(conn_data.shape[1])

gan_args = [batch_size, learning_rate, noise_dim, conn_data.shape[1], dim]
train_args = ['', epochs, log_step]

# run training to learn from data
model = GAN
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

models = {'GAN': ['GAN', False, synthesizer.generator]}

breakpoint()
# Setup parameters visualization parameters
seed = 17
test_size = 200  # number of sites
noise_dim = 32
### ----------------Sample data and transform to original data space------------------###
np.random.seed(seed)
real = synthesizer.get_data_batch(
    train=conn_data, batch_size=test_size, seed=seed)
real_samples = pd.DataFrame(real, columns=data_cols)
conn_samples = pd.DataFrame(scaler.inverse_transform(
    real_samples[data_cols[1:]]), columns=data_cols[1:])
conn_data_full = pd.DataFrame(scaler.inverse_transform(
    conn_data[data_cols[1:]]), columns=data_cols[1:])
# ax = plt.imshow(conn_orig[conn_orig.keys()[1:20]][0:19] ,interpolation = 'nearest')
# plt.show()
# ax = plt.imshow(conn_samples[conn_samples.keys()[1:20]][0:19] ,interpolation = 'nearest')
# plt.show()

evaluate(conn_samples, conn_data_full, metrics=['KSTest'])
LogisticDetection.compute(conn_data_full, conn_samples)
breakpoint()
table_evaluator = TableEvaluator(conn_data_full[conn_data_full.keys()[
                                 1:200]], conn_samples[conn_samples.keys()[1:200]])
table_evaluator.visual_evaluation()

breakpoint()
# interpolate connectivity onto sythetic site data lats and longs
synth_lats = site_data_synth['lat']
synth_longs = site_data_synth['long']
n_sites_synth = len(synth_lats)
synth_east_west = np.zeros((n_sites_synth, n_sites_synth))
synth_north_south = np.zeros((n_sites_synth, n_sites_synth))

for i in range(n_sites_synth):
    for j in range(n_sites_synth):

        synth_east_west[i, j] = tide_dist(synth_longs[i], synth_longs[j])
        synth_north_south[i, j] = tide_dist(synth_lats[i], synth_lats[j])

breakpoint()
east_west_samps = conn_samples[east_west_cols]
north_south_samps = conn_samples[north_south_cols]
conn_samps = conn_samples[conn_orig.columns]
fun = interpolate.interp2d(
    east_west_samps.values, north_south_samps.values, conn_samps.values, kind='linear')
