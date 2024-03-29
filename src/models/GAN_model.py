import os
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout
from keras import Model
from keras.optimizers.legacy import Adam
import numpy as np

### ----------------GAN model definitions------------------###
# define the GAN model (taken from GitHub repo :https://github.com/ydataai/ydata-synthetic )

file_dir = os.path.dirname(os.path.abspath(__file__))
GAN_MOD_DIR = file_dir[:-11] + "\\src\\models\\"


class GAN:
    def __init__(self, gan_args):
        [self.batch_size, lr, self.noise_dim, self.data_dim, layers_dim] = gan_args

        self.generator = Generator(self.batch_size).build_model(
            input_shape=(self.noise_dim,), dim=layers_dim, data_dim=self.data_dim
        )

        self.discriminator = Discriminator(self.batch_size).build_model(
            input_shape=(self.data_dim,), dim=layers_dim
        )

        optimizer = Adam(lr, 0.5)

        # Build and compile the discriminator
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

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
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

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
        train_ix = np.random.choice(list(train.index), replace=False, size=len(train))
        # duplicate to cover ranges past the end of the set
        train_ix = list(train_ix) + list(train_ix)
        x = train.loc[train_ix[start_i:stop_i]].values
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
            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (epoch, d_loss[0], 100 * d_loss[1], g_loss)
            )

            # If at save interval => save generated events
            if epoch % sample_interval == 0:
                # Test here data generation step
                # save model checkpoints

                model_checkpoint_base_name = (
                    GAN_MOD_DIR
                    + "model\\"
                    + cache_prefix
                    + "_{}_model_weights_step_{}.h5"
                )
                self.generator.save_weights(
                    model_checkpoint_base_name.format("generator", epoch)
                )
                self.discriminator.save_weights(
                    model_checkpoint_base_name.format("discriminator", epoch)
                )

                # Here is generating the data
                z = tf.random.normal((432, self.noise_dim))
                gen_data = self.generator(z)
                print("generated_data")

    def save(self, path, name):
        assert (
            os.path.isdir(path) == True
        ), "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return

    def load(self, path):
        assert (
            os.path.isdir(path) == True
        ), "Please provide a valid path. Path must be a directory."
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator


class Generator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim, data_dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim, activation="relu")(input)
        x = Dense(dim * 2, activation="relu")(x)
        x = Dense(dim * 4, activation="relu")(x)
        x = Dense(data_dim)(x)
        return Model(inputs=input, outputs=x)


class Discriminator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        x = Dense(dim * 4, activation="relu")(input)
        x = Dropout(0.1)(x)
        x = Dense(dim * 2, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(dim, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)
        return Model(inputs=input, outputs=x)
