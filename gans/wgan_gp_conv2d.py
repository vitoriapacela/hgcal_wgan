from __future__ import print_function, division
import setGPU
import h5py
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import time

BATCH_SIZE=128

def saveModel(model, name="regression"):
    '''
    Saves model as json file..
    :parameter model: model to be saved.
    :parameter name: name of the model to be saved.
    :type name: str
    :return: saved model.
    '''
    model_name = name
    model.summary()
    #model.save_weights('%s.h5' % model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s.json" % model_name, "w") as json_file:
        json_file.write(model_json)
        
def plotPred(real_sum, generated_sum, epoch):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(real_sum, alpha=0.3, bins=7, label='real sum', color='blue')
    ax.hist(generated_sum, alpha=0.3, bins=7, label='generated sum', color='red')
    plt.legend(prop={'size': 15})
    ax.set_xlabel('Energy (GeV)', size=16)
    ax.set_ylabel('n samples', size=16)
    #plt.show()
    plt.savefig('images/pred_%s.png'%epoch)
    
def saveLosses(name, discr_real, discr_fake, discr, gen):
    '''
    Saves true energy and prediction energy arrays into an HDF5 file.
    :parameter name: name of the file to be saved.
    :type name: str.
    '''
    new_file = h5py.File(name + "_losses.h5", 'a')
    new_file.create_dataset("discriminator_real", data=discr_real)
    new_file.create_dataset("discriminator_fake", data=discr_fake)
    new_file.create_dataset("discriminator", data=discr)
    new_file.create_dataset("generator", data=gen)
    new_file.close()

    
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
    
    
class WGANGP():
    def __init__(self):
        self.img_rows = 16
        self.img_cols = 16
        self.channels = 55
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.tag = "gp"
        #self.img_shape = (16, 16, 55)
        self.latent_dim = 100
        
        # Whether you want to use a validation set
        self.validate = False

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()

        # Build the generator
        self.generator = self.build_generator()
        
        
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------
        
        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        
        # Noise input
        z_disc = Input(shape=(100,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)
    
        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)
        
        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        saveModel(self.critic, "weights/discriminator_model_" + self.tag)
        
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        saveModel(self.generator, "weights/generator_model_" + self.tag)

        
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))

        model.add(Reshape((8, 8, 128)))

        model.add(Conv2D(filters=128, kernel_size=(6, 6), padding='same'))

        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D(size=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(1, 1), padding='valid'))

        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D(size=(1, 1)))

        model.add(ZeroPadding2D(padding=(1, 1)))

        model.add(Conv2D(filters=self.channels, kernel_size=(3, 3), padding='valid'))  

        model.add(Activation('relu'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', input_shape=self.img_shape))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.25))

        model.add(ZeroPadding2D(padding=(2, 2)))

        model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='valid'))

        model.add(LeakyReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(rate=0.25))

        model.add(ZeroPadding2D(padding=(2, 2)))

        model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='valid'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(rate=0.25))

        model.add(ZeroPadding2D(padding=(1, 1)))

        model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='valid'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(rate=0.25))

        model.add(AveragePooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the training dataset
        #f = h5py.File('/bigdata/shared/HGCAL_data/new/all_noPU.h5', 'r')
        f = h5py.File('/mnt/ceph/users/vbarinpa/single_particle/all_noPU.h5', 'r')
        X_train = np.asarray(f['image'])

        # Shuffle
        np.random.shuffle(X_train)
        f.close()

        # Load validation data
        if (self.validate):
            #g = h5py.File('/bigdata/shared/HGCAL_data/new_multi_small/no_pu/ntuple_merged_159_no_pu.h5', 'r')
            g = h5py.File('/mnt/ceph/users/vbarinpa/single_particle/no_pu/ntuple_merged_998_no_pu.h5', 'r')
            X_val = np.asarray(g['image'])

            # Shuffle
            np.random.shuffle(X_val)
            g.close()
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        d_loss_reals = []
        d_loss_fakes = []
        d_losses = []
        
        val_d_loss_reals = []
        val_d_loss_fakes = []
        val_d_losses = []
        
        g_losses = []
        
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images to TRAIN
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                # Select a random batch of images to VALIDATE
                if (self.validate):
                    idx_val = np.random.randint(0, X_val.shape[0], batch_size)
                    imgs_val = X_val[idx_val]
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])
                
                # Validation critic
                if (self.validate):
                    val_d_loss_real = self.critic_model.train_on_batch([imgs_val, noise], [valid, fake, dummy])

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

                d_losses.append(d_loss)
                
                if (self.validate):
                    val_d_losses.append(val_d_loss)
            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)
            
            g_losses.append(g_loss)
            
            # Print the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples and weights
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            
                # Save model weights after every sample interval
                self.generator.save_weights("weights/generator_weights_epoch_%d_%s.h5" % (epoch, self.tag))
                self.critic.save_weights("weights/discriminator_weights_epoch_%d_%s.h5" % (epoch, self.tag))
            
            
        # Save losses in an HDF5 file:
        saveLosses(self.tag, d_loss_reals, d_loss_fakes, d_losses, g_losses)
        
        if (self.validate):
            saveLosses("val_"+self.tag, val_d_loss_reals, val_d_loss_fakes, val_d_losses, g_losses)
        
                
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 25], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/hgcal_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    start = time.mktime(time.gmtime())

    if not os.path.exists(os.getcwd()+"/weights/"):
        os.makedirs(os.getcwd()+"/weights/")
    
    if not os.path.exists(os.getcwd()+"/images/"):
        os.makedirs(os.getcwd()+"/images/")
        
    wgan_gp = WGANGP()
    wgan_gp.train(epochs=2000, batch_size=BATCH_SIZE, sample_interval=25)

    stop = time.mktime(time.gmtime())
    print("train_time " + str(stop - start))
