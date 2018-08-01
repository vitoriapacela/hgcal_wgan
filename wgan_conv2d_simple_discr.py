from __future__ import print_function, division
import setGPU
import h5py

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import time

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

class WGAN():
    def __init__(self):
        self.img_rows = 16
        self.img_cols = 16
        self.channels = 55
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.tag = "change_discriminator"
        #self.img_shape = (16, 16, 55)
        self.latent_dim = 100
        
        # Whether you want to use a validation set
        self.validate = True

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        #self.critic.to_json()
        saveModel(self.critic, "weights/discriminator_model_" + self.tag)

        # Build the generator
        self.generator = self.build_generator()
        #self.generator.to_json()
        saveModel(self.generator, "weights/generator_model_" + self.tag)
        
        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        #self.combined.to_json()
        saveModel(self.combined, "weights/combined_model_" + self.tag)

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

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))        

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the training dataset
        #f = h5py.File('/bigdata/shared/HGCAL_data/new/all_noPU.h5', 'r')
        f = h5py.File('/mnt/ceph/users/vbarinpa/all_noPU.h5', 'r')
        X_train = np.asarray(f['X'])
        X_train = X_train[:, :, :, :, 0]

        # Rescale train -1 to 1
        #X_train = (X - np.mean(X))/np.mean(X)

        # Shuffle
        np.random.shuffle(X_train)
        f.close()

        # Load validation data
        if (self.validate):
            #g = h5py.File('/bigdata/shared/HGCAL_data/new_multi_small/no_pu/ntuple_merged_159_no_pu.h5', 'r')
            g = h5py.File('/mnt/ceph/users/vbarinpa/multi_3d/no_pu/ntuple_merged_998_no_pu.h5', 'r')
            X_val = np.asarray(g['X'])
            X_val = X_val[:, :, :, :, 0]
            # Rescale val -1 to 1
            #X_val = (val - np.mean(val))/np.mean(val)

            # Shuffle
            np.random.shuffle(X_val)
            g.close()
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

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
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                
                # Validation critic
                if (self.validate):
                    val_d_loss_real = self.critic.test_on_batch(imgs_val, valid)
                    val_d_loss_fake = self.critic.test_on_batch(gen_imgs, fake)
                    val_d_loss = 0.5 * np.add(val_d_loss_fake, val_d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

                d_loss_reals.append(d_loss_real)
                d_loss_fakes.append(d_loss_fake)
                d_losses.append(d_loss)
                
                if (self.validate):
                    val_d_loss_reals.append(val_d_loss_real)
                    val_d_loss_fakes.append(val_d_loss_fake)
                    val_d_losses.append(val_d_loss)
            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)
            
            #val_g_loss = self.combined.test_on_batch(noise, valid)
            
            g_losses.append(g_loss)
            
            # Print the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples and weights
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            
                # Save model weights after every sample interval
                self.generator.save_weights("weights/generator_weights_epoch_%d_%s.h5" % (epoch, self.tag))
                self.critic.save_weights("weights/discriminator_weights_epoch_%d_%s.h5" % (epoch, self.tag))
                self.combined.save_weights("weights/combined_weights_epoch_%d_%s.h5" % (epoch, self.tag))
            
                # Plot histograms
                #inp_sum = np.sum(X[0:batch_size], axis = (1, 2, 3))
                #inp_sum = np.sum(val[0:batch_size], axis = (1, 2, 3))
                #gen_sum = np.sum(gen_imgs, axis = (1, 2, 3))
                #print(inp_sum[:,0].shape)
                #print(gen_sum[:,0].shape)
                #plotPred(inp_sum[:,0], gen_sum[:,0], epoch)

            #self.sums(epoch)
            
        # Save losses in an HDF5 file:
        saveLosses(self.tag, d_loss_reals, d_loss_fakes, d_losses, g_losses) # TO EDIT, ADD VALIDATION
        
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
        
    wgan = WGAN()
    wgan.train(epochs=2000, batch_size=128, sample_interval=25)

    stop = time.mktime(time.gmtime())
    print("train_time " + str(stop - start))
