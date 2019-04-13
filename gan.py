import os
import numpy as np

from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Activation, Input, Dense, Reshape, Flatten, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.advanced_activations import LeakyReLU

import pymrt as mrt
import pymrt.geometry

import matplotlib.pyplot as plt
from skimage import measure

class GAN(object):


    def __init__(self, width=128, height=128, depth=128):
        self.width = width
        self.height = height
        self.depth = depth
        self.size = width * height * depth
        self.latent = 128
        self.shape = (self.width, self.height, self.depth)

        self.OPTIMIZERD = Adam(lr=0.0005, decay=0.01)
        self.OPTIMIZER = Adam(lr=0.0002, decay=0.005)

        self.G = self.generator()
        self.D = self.discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZERD, metrics=['accuracy'])

        z = Input(shape=(self.latent,))
        self.stacked_G_D = Model(z, self.D(self.G(z)))
        self.D.trainable = False
        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER)

    def generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.latent,)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dense(self.size, activation='tanh'))
        model.add(Reshape(self.shape))
       
        

        noise = Input(shape=(self.latent,))
        img = model(noise)

        return Model(noise, img)

    def discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(128, input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.shape)
        validity = model(img)
        
        return Model(img, validity)



    def train(self, X_train, epochs=10000, batch=2, save_interval=20):
        valid = np.ones((batch, 1))
        fake = np.zeros((batch, 1))
        for cnt in range(epochs):
            noise = np.random.normal(0, 1, (batch, self.latent))
            ## train discriminator
            random_index = np.random.randint(0, len(X_train), batch)
            legit_images = X_train[random_index].reshape(batch, self.width, self.height, self.depth)
            
            # generate some noise
            # generate a batch of new images
            syntetic_images = self.G.predict(noise)
            
      
            d_loss = 0.5 * np.add(self.D.train_on_batch(legit_images, valid),self.D.train_on_batch(syntetic_images, fake))
                                    
            
            # train generator
            g_loss = self.stacked_G_D.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (cnt, d_loss[0], 100*d_loss[1], g_loss))

            if cnt % save_interval == 0 and cnt > 0:
                plot(syntetic_images[0])


