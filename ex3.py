
import pandas as pd
import csv
import numpy as np
import os
import sys
import random

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras import backend as K

import cv2
import matplotlib
# matplotlib.use('Agg')
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
            targeted_label = raw_df [ raw_df['level'] == level ]
            level_count.append(len(targeted_label))
            self.level_number.append(level_count[level])
            print("the number of label " + str(level) +" is: " + str(level_count[level])
                  + " as" + "{0: .0%}".format(level_count[level] / total_num))

        raw_df['PatientID'] = ''

        for index, row in raw_df.iterrows():
            patientID = row[0]
            patientID = patientID.replace('_right', '')
            patientID = patientID.replace('_left', '')
            raw_df.at[index, 'PatientID'] = patientID

        print('number of patients: ', len(raw_df['PatientID'].unique()))

        return raw_df

    def readtrainData(self, trainDir):
        ImageNameDataHash = {}
        # loop over the input images
        images = os.listdir(trainDir)
        print("Number of files in " + trainDir + " is " + str(len(images)))
        for imageFileName in images:
            if (imageFileName == "trainLabels.csv"):
                continue
            if ("15001" in imageFileName):
                break;
            # load the image, pre-process it, and store it in the data list
            imageFullPath = os.path.join(trainDir, imageFileName)
            # print(imageFullPath)
            # img = load_img(imageFullPath)
            # arr = img_to_array(img)
            try:
                img_bgr = cv2.imread(imageFullPath)
                img_rgb = np.stack((img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0]), axis=-1)
                resized_img = cv2.resize(img_rgb, (self.HEIGHT, self.WIDTH))  # Numpy array with shape (HEIGHT, WIDTH,3)
                # print(imageFullPath+"is empty")

                imageFileName = imageFileName.replace('.jpeg', '')
                ImageNameDataHash[str(imageFileName)] = resized_img
            except:
                print("discard image" + imageFileNmae)
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
            new_img = clache.apply(img_g)

        return new_img

    def multi_class_split(self, data, HEIGHT=512, WIDTH=512, DEPTH=3):
        image_data = np.array(data['data'])
        image_label = np.array(data['level']).reshape(len(data['level'], 1))
        image_data_new = np.zeros((image_data.shape[0], HEIGHT, WIDTH, DEPTH))
        for i in range(image_data.shape[0]):
            image_data_new[i,:,:,:] = image_data[i]

        return image_data_new, image_label

    def binaryCluster(self, trainDF, aug_sick=True, aug_healthy=True, DEPTH=3):
        train_sick = trainDF[trainDF['level'] != 0]
        print("data_sick shape: ", train_sick.shape)
        train_healthy = trainDF[trainDF['level'] == 0]
        print("data_healthy shape:", train_healthy.shape)

        # deal with data labeled with 1,2,3,4
        train_sick_new = np.zeros((train_sick['data'].shape[0], 512, 512, DEPTH))
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
        train_healthy_new = np.zeros((train_healthy.shape[0], 512, 512, DEPTH))
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


if __name__ == "__main__":
    readData = read_data()
    raw_df = readData.readtrainCSV('trainLabels.csv')
    total_NameDataHash = readData.readtrainData("train_smaller/")
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

    batch_size = 100
    nb_classes = 5
    nb_epoch = 5

    img_rows, img_cols = 512, 512
    img_channels = 3

    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
    depth = 40
    nb_dense_block = 3
    growth_rate = 24
    nb_filter = -1
    dropout_rate = 0.0  # 0.0 for data augmentation

    model = sparsenet.SparseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                                growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-3, amsgrad=True)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    #(trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX, trainY = pData.multi_class_split(trainDF)
    testX, testY = pData.multi_class_split(valDF)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    # trainX = sparsenet.preprocess_input(trainX)
    # testX = sparsenet.preprocess_input(testX)

    cifar_mean = trainX.mean(axis=(0, 1, 2), keepdims=True)
    cifar_std = trainX.std(axis=(0, 1, 2), keepdims=True)

    trainX = (trainX - cifar_mean) / (cifar_std + 1e-8)
    testX = (testX - cifar_mean) / (cifar_std + 1e-8)

    Y_train = np_utils.to_categorical(trainY, nb_classes)
    Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)

    generator.fit(trainX, seed=0)

    # Load model
    weights_file = "weights/SparseNet-40-24-CIFAR10.h5"
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        print("Model loaded.")

    out_dir = "weights/"

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    callbacks = [lr_reducer, model_checkpoint]

    model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                        steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, Y_test),
                        validation_steps=testX.shape[0] // batch_size, verbose=1)

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)


