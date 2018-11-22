import pandas as pd
import csv
import numpy as np
import os
import sys
import random

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras import regularizers
from keras import backend as K


import cv2
import matplotlib
import matplotlib.pyplot as plt


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
            patientID = patientID.replace('_right','')
            patientID = patientID.replace('_left','')
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
            if ("10636" in imageFileName):
                break
            # load the image, pre-process it, and store it in the data list
            imageFullPath = os.path.join(trainDir, imageFileName)
            #print(imageFullPath)
            #img = load_img(imageFullPath)
            #arr = img_to_array(img)
            img_bgr = cv2.imread(imageFullPath)
            img_rgb = np.stack((img_bgr[:,:,2],img_bgr[:,:,1],img_bgr[:,:,0]), axis=-1)
            resized_img = cv2.resize(img_rgb, (self.HEIGHT,self.WIDTH)) #Numpy array with shape (HEIGHT, WIDTH,3)
            #print(imageFullPath+"is empty")

            imageFileName = imageFileName.replace('.jpeg','')
            ImageNameDataHash[str(imageFileName)] = resized_img
        return ImageNameDataHash

    def outputPD(self, raw_df, ImageNameDataHash):
        keepImages = list(ImageNameDataHash.keys())
        raw_df = raw_df[raw_df['image'].isin(keepImages)]
        print("the new length of label data is: ",len(raw_df))

        #convert hash to dataframe
        image_name = []
        image_data = []
        for index, row in raw_df.iterrows():
            key = str(row[0])
            if key in ImageNameDataHash:
                image_name.append(key)
                image_data.append(ImageNameDataHash[key])

        total_data = pd.DataFrame({'image': image_name, 'data': image_data})
        print(list(total_data.columns))
        print("the length of the new total data is ",len(total_data))
        total_data = pd.merge(total_data, raw_df, left_on='image', right_on='image', how='outer')
        print("after merging both labels and image data...")
        print(list(total_data.columns))
        return total_data
        
        

class process_data:

    def enhanceImage(self, img_rgb):
        #img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_g = img_rgb[:,:,1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #L = img_lab[:,:,0]
        img_g = clahe.apply(img_g)
        #convert to 1-channel
        img_g = np.stack((img_g,), axis=-1)
        #img_lab[:,:,0] = L_clahe
        #img_rgb_new = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return img_g

def createModel(input_shape, INIT_LR = 1e-3, EPOCHS=10, metrics="accuracy", loss="binary_crossentropy"):
        model = Sequential()
        # first set of CONV => RELU => MAX POOL layers
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(activation='sigmoid', units = 1))
        # returns our fully constructed deep learning + Keras image classifier 
        opt = SGD(lr=INIT_LR, decay=0.0015, momentum=0.9)
        # use binary_crossentropy if there are two classes
        model.compile(loss=loss, optimizer=opt, metrics=[metrics])
        return model

def balancing_data(total_data, level_number):
    targeted_class = np.argmin(level_number)
    balanced_num = np.min(level_number)
    pd_data = total_data [ total_data['level'] == targeted_class ]
    for i in range(5):
        if i == targeted_class:
            continue
        total_num = level_number[i]
        random_index = random.sample(range(total_num), balanced_num)
        ori_data = total_data[ total_data['level'] == i ]
        ori_data = ori_data.reset_index(drop=True)
        new_data = ori_data[ori_data.index.isin(random_index)].reset_index(drop=True)
        pd_data = pd_data.append(new_data)

    return pd_data.reset_index(drop=True)

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = .2
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    

if __name__ == "__main__":
    readData = read_data()
    raw_df = readData.readtrainCSV('trainLabels.csv')
    total_NameDataHash = readData.readtrainData("train_small/")
    pData = process_data()
    # output total_data in pandas dataframe for traininig and validation
    total_data = readData.outputPD(raw_df, total_NameDataHash)
    # total_data.to_pickle("total_data.pkl")
    # total_data = pd.read_pickle('total_data.pkl')
    #level_number = read_data.level_number

    WIDTH = 512
    HEIGHT = 512
    DEPTH = 1

    print("partition data into 75:25...")
    sys.stdout.flush()
    sample_num = total_data.shape[0]
    index_list = np.array(range(sample_num))

    train_ids, valid_ids = train_test_split(index_list, test_size=0.05, random_state=10)
    trainID_list = train_ids.tolist()
    print("trainID_list shape: ", len(trainID_list))

    trainDF = total_data[total_data.index.isin(trainID_list)]
    valDF = total_data[~total_data.index.isin(trainID_list)]
    del total_data
    trainDF = trainDF.reset_index(drop=True)
    valDF = valDF.reset_index(drop=True)

    # balancing the training data
    # trainDF = balancing_data(trainDF, level_number)
    train_sick = trainDF[trainDF['level'] != 0]
    train_healthy = trainDF[trainDF['level'] == 0]
    # del train_DF
    # data augmentation
    print("augmenting data")
    sys.stdout.flush()
    aug = ImageDataGenerator(rotation_range=180, fill_mode="nearest")
    train_sick_new = np.zeros((train_sick['data'].shape[0], 512, 512, 1))
    train_sick = np.array(train_sick['data'])
    for i in range(train_sick.shape[0]):
        train_sick_new[i, :, :, :] = pData.enhanceImage(train_sick[i])
    generated_data = aug.flow(train_sick_new, batch_size=train_sick_new.shape[0])
    generated_data = generated_data[0]
    aug = ImageDataGenerator(vertical_flip=True, fill_mode="nearest")
    sec_generated_data = aug.flow(train_sick_new, batch_size=train_sick_new.shape[0])
    sec_generated_data = sec_generated_data[0]

    train_sick = np.concatenate((train_sick_new, generated_data, sec_generated_data))
    del train_sick_new
    train_sick_label = np.ones((train_sick.shape[0], 1))

    print("sample image from train_sick: ")
    #plt.imshow(train_sick[10,:,:,:].reshape(512, 512) / 255.0)
    #plt.show()
    print("label for the sample image is: ", train_sick_label[10])

    train_healthy_label = np.array(train_healthy['level']).reshape(train_healthy.shape[0], 1)
    train_healthy = np.array(train_healthy['data'])
    train_healthy_new = np.zeros((train_healthy.shape[0], 512, 512, 1))
    for i in range(train_healthy.shape[0]):
        train_healthy_new[i, :, :, :] = pData.enhanceImage(train_healthy[i])
    train_healthy = train_healthy_new
    print("sample image from train_healthy: ")
    #plt.imshow(train_healthy_new[10,:,:,:].reshape(512, 512) / 255.0)
    #plt.show()
    print("label for the sample image is: ", train_healthy_label[10])
    del train_healthy_new

    train_x = np.concatenate((train_healthy, train_sick))
    train_y = np.concatenate((train_healthy_label, train_sick_label))
    train_healthy = []
    train_sick = []
    train_healthy_label = []
    train_sick_label = []
    del train_healthy , train_sick, train_healthy_label, train_sick_label

    val_y = np.array(valDF['level']).reshape(len(valDF['level']), 1)
    val_y_healthy = val_y[ val_y == 0 ]
    val_y_healthy = val_y_healthy.reshape((len(val_y_healthy),1))
    val_y_sick = val_y[ val_y != 0 ]
    val_y_sick = np.ones((len(val_y_sick), 1))
    val_y = np.concatenate((val_y_healthy, val_y_sick))

    val_x_healthy = valDF[ valDF['level'] == 0]
    val_x_sick = valDF[ valDF['level'] != 0]
    val_x_healthy = np.array(val_x_healthy['data'])
    val_x_sick = np.array(val_x_sick['data'])
    val_x = np.concatenate((val_x_healthy, val_x_sick))

    val_x_new = np.zeros((val_x.shape[0], 512, 512, 1))
    for i in range(val_x.shape[0]):
        val_x_new[i,:,:,:] = pData.enhanceImage(val_x[i])
    val_x = val_x_new

    print("sample image from val_x: ")
    #plt.imshow(val_x[10,:,:,:].reshape(512,512) / 255.0)
    #plt.show()
    #print("label for the sample image is: ", val_y[10])
    del val_x_new, valDF, val_x_healthy, val_x_sick

    print("train_x shape: ", train_x.shape, "val_x shape: ", val_x.shape)
    print("train_y shape: ", train_y.shape, "val_y shape: ", val_y.shape)
    print("classes contained in train_y: ", np.unique(train_y))
    print("classes contained in val_y: ", np.unique(val_y))

    #NUM_CLASSES = 5
    #rain_y = to_categorical(train_y, num_classes=NUM_CLASSES)
    #val_y = to_categorical(val_y, num_classes=NUM_CLASSES)

    #data augmentation

    BATCH_SIZE = 32
    EPOCHS = 8

    #level_number = readData.level_number
    #class_weight = {0: 2., 1: 8., 2: 8., 3: 8., 4: 8.}
    #print("class_weight: ", class_weight)

    #print("Reshaping train_x and val_x")
    #x_train = np.zeros([train_x.shape[0], HEIGHT, WIDTH, DEPTH], dtype=int)
    #x_val = np.zeros([val_x.shape[0], HEIGHT, WIDTH, DEPTH], dtype=int)
    #for i in range(train_x.shape[0]):
       #x_train[i] = train_x[i]
    #for i in range(val_x.shape[0]):
        #x_val[i] = val_x[i]
    #print("reshaped train_x as x_train: ", x_train.shape)
    #print("reshaped val_x as x_val: ", x_val.shape)
    print('compiling model...')
    
    #global variable
    INTERESTING_CLASS_ID = 0  # Choose the class of interest
    
    input_shape = (HEIGHT, WIDTH, DEPTH)
    model = createModel(input_shape, loss=focal_loss)
    model.summary()

    #del total_data, total_NameDataHash, train_x, val_x
    print('training network')
    sys.stdout.flush()
    H = model.fit(train_x / 255.0, train_y, batch_size=BATCH_SIZE, validation_data=(val_x / 255.0, val_y), epochs=EPOCHS, verbose=1, shuffle=True)

    #print("saving model")
    #sys.stdout.flush()
    #model.save('trained_model')

    #print("Generating plots...")
    predictions = model.predict(val_x / 255.0)
    print("one-hot coded prediction results: ",predictions)
    y_test = np.argmax(val_y, axis=-1)
    print("Ground truth: ",y_test)
    predictions = np.argmax(predictions, axis=-1)
    print("Predictions as one single variable list: ",predictions)
    c = confusion_matrix(y_test, predictions)
    print("The shape of confusion matrix: ",c.shape)
    print(c)
