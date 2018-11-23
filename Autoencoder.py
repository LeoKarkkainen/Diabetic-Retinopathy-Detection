import pandas as pd
import csv
import numpy as np
import os
import sys
import random

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import metrics
from keras.utils import to_categorical
from keras import backend as K
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

import sparsenet


class read_data:
    level_number = []

    def __init__(self, HEIGHT=512, WIDTH=512):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

    def readtrainCSV(self, csvDir):
        print("reading trainLabels.csv ...")
        raw_df = pd.read_csv(csvDir, sep=',')
        total_num = len(raw_df)
        level_list = [0, 1, 2, 3, 4]
        level_count = []
        for level in level_list:
            targeted_label = raw_df[raw_df['level'] == level]
            level_count.append(len(targeted_label))
            self.level_number.append(level_count[level])
            print("the number of label " + str(level) + " is: " + str(level_count[level])
                  + " as" + "{0: .0%}".format(level_count[level] / total_num))

        raw_df['PatientID'] = ''

        for index, row in raw_df.iterrows():
            patientID = row[0]
            patientID = patientID.replace('_right', '')
            patientID = patientID.replace('_left', '')
            raw_df.at[index, 'PatientID'] = patientID

        print('number of patients: ', len(raw_df['PatientID'].unique()))

        return raw_df

    def readtrainData(self, trainDir, isBGR=True):
        ImageNameDataHash = {}
        # loop over the input images
        images = os.listdir(trainDir)
        print("Number of files in " + trainDir + " is " + str(len(images)))
        for imageFileName in images:
            if (imageFileName == "trainLabels.csv"):
                continue
            if ("229" in imageFileName):
                break;
            # load the image, pre-process it, and store it in the data list
            imageFullPath = os.path.join(trainDir, imageFileName)
            img = cv2.imread(imageFullPath)
            if isBGR == False:
                img = np.stack((img[:, :, 2], img[:, :, 1], img[:, :, 0]), axis=-1)
            try:
                resized_img = self.local_normalization(img)
                resized_img = cv2.resize(resized_img,(self.HEIGHT, self.WIDTH))  # Numpy array with shape (HEIGHT, WIDTH,3)
                imageFileName = imageFileName.replace('.jpeg', '')
                ImageNameDataHash[str(imageFileName)] = resized_img
            except:
                print("discard image " + imageFileName)
        return ImageNameDataHash

    def outputPD(self, raw_df, ImageNameDataHash):
        '''
        adjust the label dataset to the imported training dataset
        '''
        keepImages = list(ImageNameDataHash.keys())
        raw_df = raw_df[raw_df['image'].isin(keepImages)]
        print("the new length of label data is: ", len(raw_df))

        # convert hash to dataframe
        image_name = []
        image_data = []
        for index, row in raw_df.iterrows():
            key = str(row[0])
            if key in ImageNameDataHash:
                image_name.append(key)
                image_data.append(ImageNameDataHash[key])

        total_data = pd.DataFrame({'image': image_name, 'data': image_data})
        print(list(total_data.columns))
        print("the length of the new total data is ", len(total_data))
        total_data = pd.merge(total_data, raw_df, left_on='image', right_on='image', how='outer')
        print("after merging both labels and image data...")
        print(list(total_data.columns))
        return total_data

    def scale_radius(self, img, scale):
        x = img[int(img.shape[0] / 2), :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        return cv2.resize(img, (0, 0), fx=s, fy=s)

    def local_normalization(self, img, scale=300):
        a = self.scale_radius(img, scale)
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
        b = np.zeros(a.shape)
        cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        a = a * b + 128 * (1 - b)
        #print("a shape: ", a.shape)
        return a


class process_data:

    def enhanceImage(self, img_rgb, gray=False):
        if gray == False:
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L = img_lab[:, :, 0]
            L_clahe = clahe.apply(L)
            img_lab[:, :, 0] = L_clahe
            new_img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            img_g = img_rgb[:, :, 1]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            new_img = clahe.apply(img_g)
            if len(new_img) == 2:
                new_img = np.stack((new_img,), axis=-1)

        return new_img

    def multi_class_split(self, data, HEIGHT=512, WIDTH=512, DEPTH=3, green=False):
        image_data = np.array(data['data'])
        image_label = np.array(data['level']).reshape(len(data['level']), 1)
        image_data_new = np.zeros((image_data.shape[0], HEIGHT, WIDTH, DEPTH))
        if green == True:
            for i in range(image_data.shape[0]):
                img_g = image_data[i]
                img_g = img_g[:, :, 1]
                img_g = np.stack((img_g,),axis=-1)
                image_data_new[i, :, :, :] = img_g
        else:
            for i in range(image_data.shape[0]):
                image_data_new[i, :, :, :] = image_data[i]

        return image_data_new, image_label

    def binaryCluster(self, trainDF, aug_sick=True, aug_healthy=True, HEIGHT=512, WIDTH=512, DEPTH=3):
        train_sick = trainDF[trainDF['level'] != 0]
        print("data_sick shape: ", train_sick.shape)
        train_healthy = trainDF[trainDF['level'] == 0]
        print("data_healthy shape:", train_healthy.shape)

        # deal with data labeled with 1,2,3,4
        train_sick_new = np.zeros((train_sick['data'].shape[0], HEIGHT, WIDTH, DEPTH))
        train_sick = np.array(train_sick['data'])
        for i in range(train_sick.shape[0]):
            # adjusting data format for ImageDataGenerator
            train_sick_new[i, :, :, :] = train_sick[i]
        print("data_sick_new shape: ", train_sick_new.shape)

        if aug_sick == True:
            train_sick = self.augmentData(train_sick_new, train_sick_new.shape[0])
            print("data_sick final shape: ", train_sick.shape)
        else:
            train_sick = train_sick_new
            print("data_sick final shape:", train_sick.shape)

        train_sick_label = np.ones((train_sick.shape[0], 1))  # 1,2,3,4 converted to class 1

        # deal with data labeled with 0
        train_healthy_new = np.zeros((train_healthy.shape[0], HEIGHT, WIDTH, DEPTH))
        train_healthy = np.array(train_healthy['data'])
        for i in range(train_healthy.shape[0]):
            train_healthy_new[i, :, :, :] = train_healthy[i]

        if aug_healthy == True:
            train_healthy = self.augmentData(train_healthy_new, train_healthy_new.shape[0] // 3, False)
            print("data_healthy final shape:", train_healthy.shape)
        else:
            train_healthy = train_healthy_new
            print("data_healthy final shape:", train_healthy.shape)

        train_healthy_label = np.zeros((train_healthy.shape[0], 1))

        train_x = np.concatenate((train_sick, train_healthy))
        train_y = np.concatenate((train_sick_label, train_healthy_label))
        # train_x = np.random.shuffle(train_x)

        return train_x, train_y

    def augmentData(self, data, batch_size, big=True):

        print("augmenting data")
        sys.stdout.flush()
        if big == True:
            # augment data scheme 1
            aug = ImageDataGenerator(rotation_range=90, fill_mode="nearest")
            generated_data_1 = aug.flow(data, batch_size=batch_size)
            generated_data_1 = generated_data_1[0]
            # augment data scheme 2
            aug = ImageDataGenerator(rotation_range=90, vertical_flip=True, fill_mode="nearest")
            generated_data_2 = aug.flow(data, batch_size=batch_size)
            generated_data_2 = generated_data_2[0]
            augmented_data = np.concatenate((data, generated_data_1, generated_data_2))
            print("augmented_data size: ", augmented_data.shape)
        else:
            # augment data scheme 1
            aug = ImageDataGenerator(rotation_range=90, fill_mode="nearest")
            generated_data_1 = aug.flow(data, batch_size=batch_size)
            generated_data_1 = generated_data_1[0]
            augmented_data = np.concatenate((data, generated_data_1))
            print("augmented_data size: ", augmented_data.shape)

        return augmented_data

    def balancing_data(total_data, level_number):
        targeted_class = np.argmin(level_number)
        balanced_num = np.min(level_number)
        pd_data = total_data[total_data['level'] == targeted_class]
        for i in range(5):
            if i == targeted_class:
                continue
            total_num = level_number[i]
            random_index = random.sample(range(total_num), balanced_num)
            ori_data = total_data[total_data['level'] == i]
            ori_data = ori_data.reset_index(drop=True)
            new_data = ori_data[ori_data.index.isin(random_index)].reset_index(drop=True)
            pd_data = pd_data.append(new_data)

        return pd_data.reset_index(drop=True)


def createModel(input_shape, NUM_CLASSES, INIT_LR=1e-3, EPOCHS=10, metrics="accuracy", loss="binary_crossentropy"):
    model = Sequential()
    # first set of CONV => RELU => MAX POOL layers
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(activation='softmax', units=NUM_CLASSES))
    # returns our fully constructed deep learning + Keras image classifier
    opt = Adam(lr=INIT_LR, decay=1e-6)
    # use binary_crossentropy if there are two classes
    model.compile(loss=loss, optimizer=opt, metrics=[metrics])
    return model

class VAutoEncoder:

    # sampling function
    def sampling(self, args):
        latent_dim = 2
        z_mu, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0., stddev=1.)
        return z_mu + K.exp(z_log_sigma) * epsilon

    def creatAutoEncoder(self):
        # ---- Model Construction ------------ #
        # -------------Encoder Network
        img_shape = (512, 512, 1)
        batch_size = 16
        latent_dim = 2  # Number of latent dimension parameters
        # Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
        input_img = keras.Input(shape=img_shape)

        x = layers.Conv2D(32, 3,
                          padding='same',
                          activation='relu')(input_img)
        x = layers.Conv2D(64, 3,
                          padding='same',
                          activation='relu',
                          strides=(2, 2))(x)
        x = layers.Conv2D(64, 3,
                          padding='same',
                          activation='relu')(x)
        x = layers.Conv2D(64, 3,
                          padding='same',
                          activation='relu')(x)
        # need to know the shape of the network here for the decoder
        shape_before_flattening = K.int_shape(x)

        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)

        # Two outputs, latent mean and (log)variance
        z_mu = layers.Dense(latent_dim)(x)
        z_log_sigma = layers.Dense(latent_dim)(x)

        encoder = Model(input_img, z_mu)
        # sample vector from the latent distribution
        z = layers.Lambda(self.sampling)([z_mu, z_log_sigma])

        # ---------------Decoder Network
        # decoder takes the latent distribution sample as input
        decoder_input = layers.Input(K.int_shape(z)[1:])

        # Expand to 784 total pixels
        x = layers.Dense(np.prod(shape_before_flattening[1:]),  # 14 * 14 * 64
                         activation='relu')(decoder_input)

        # reshape
        x = layers.Reshape(shape_before_flattening[1:])(x)

        # use Conv2DTranspose to reverse the conv layers from the encoder
        x = layers.Conv2DTranspose(32, 3,
                                   padding='same',
                                   activation='relu',
                                   strides=(2, 2))(x)
        x = layers.Conv2D(1, 3,
                          padding='same',
                          activation='sigmoid')(x)  # decoder output with a sigmoid function

        # decoder model statement
        decoder = Model(decoder_input, x)

        # apply the decoder to the sample from the latent distribution
        z_decoded = decoder(z)
        # apply the custom loss to the input images and the decoded latent distribution sample
        y = CustomVariationalLayer()([input_img, z_decoded, z_mu, z_log_sigma])

        # VAE model statement
        vae = Model(input_img, y)
        vae.compile(optimizer='rmsprop', loss=None)
        return vae, encoder

# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mu = inputs[2]
        z_log_sigma = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
        self.add_loss(loss, inputs=inputs) #Layer.add_loss()
        return z_decoded

if __name__ == "__main__":
    readData = read_data()
    raw_df = readData.readtrainCSV('trainLabels.csv')
    total_NameDataHash = readData.readtrainData("train/")
    pData = process_data()
    # output total_data in pandas dataframe for traininig and validation
    total_data = readData.outputPD(raw_df, total_NameDataHash)
    # total_data.to_pickle("total_data.pkl")
    # total_data =pd.read_pickle('total_data.pkl')
    # level_number = read_data.level_number

    print("partition data into 75:25...")
    sys.stdout.flush()
    sample_num = total_data.shape[0]
    index_list = np.array(range(sample_num))

    train_ids, valid_ids = train_test_split(index_list, test_size=0.25, random_state=10)
    trainID_list = train_ids.tolist()
    print("trainID_list shape: ", len(trainID_list))

    trainDF = total_data[total_data.index.isin(trainID_list)]
    valDF = total_data[~total_data.index.isin(trainID_list)]
    del total_data
    trainDF = trainDF.reset_index(drop=True)
    valDF = valDF.reset_index(drop=True)

    # balancing the training data
    # trainDF = balancing_data(trainDF, level_number)
    train_x, train_y = pData.multi_class_split(trainDF, DEPTH=1, green=True)
    print('train_x shape: ', train_x.shape)
    val_x, val_y = pData.multi_class_split(valDF, DEPTH=1, green=True)
    print('val_x shape', val_x.shape)

    vautoencoder = VAutoEncoder()
    vae, encoder = vautoencoder.creatAutoEncoder()
    vae.summary()
    batch_size = 16
    # --------Train the VAE
    vae.fit(x=train_x / 255.0, y=None,
            shuffle=True,
            epochs=7,
            batch_size=batch_size,
            validation_data=(val_x / 255.0, None),
            verbose=1)

    # Translate into the latent space
    #encoder = Model(input_img, z_mu)
    x_valid_encoded = encoder.predict(val_x / 255.0, batch_size=batch_size)
    plt.figure(figsize=(10, 10))
    plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1], c=val_y, cmap='brg')
    plt.colorbar()
    plt.savefig('clustered_results.png')

