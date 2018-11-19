import pandas as pd
import csv
import numpy as np
import os

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import cv2
from matplotlib import pyplot as plt

class read_data:

    def __init__(self, HEIGHT=512, WIDTH=512):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
    
    
    def readtrainCSV(self, csvDir):
        raw_df = pd.read_csv(csvDir, sep=',')
        total_num = len(raw_df)
        level_list = [0, 1, 2, 3, 4]
        level_count = []
        for level in level_list:
            targeted_label = raw_df [ raw_df['level'] == level ]
            level_count.append(len(targeted_label))
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
            # load the image, pre-process it, and store it in the data list
            imageFullPath = os.path.join(trainDir, imageFileName)
            #print(imageFullPath)
            #img = load_img(imageFullPath)
            #arr = img_to_array(img)
            img_bgr = cv2.imread('sample/16_left.jpeg')
            img_rgb = np.stack((img_bgr[:,:,2],img_bgr[:,:,1],img_bgr[:,:,0]), axis=-1)
            resized_img = cv2.resize(img_rgb, (self.HEIGHT,self.WIDTH)) #Numpy array with shape (HEIGHT, WIDTH,3)
            
            imageFileName = imageFileName.replace('.jpeg','')
            ImageNameDataHash[str(imageFileName)] = resized_img
        return ImageNameDataHash

class process_data:

    def enhanceImage(self, img_rgb):
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = img_lab[:,:,0]
        L_clahe = clahe.apply(L)
        img_lab[:,:,0] = L_clahe
        img_rgb_new = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        return img_rgb_new

    def colorNormalisation(self, img_rgb, window_size=16, stride=2):
        r = img_rgb[:,:,0]
        g = img_rgb[:,:,1]
        b = img_rgb[:,:,2]
        new_r = np.zeros((HEIGHT, WIDTH))
        new_g = np.zeros((HEIGHT, WIDTH))
        new_b = np.zeros((HEIGHT, WIDTH))
        num_steps = int((WIDTH - window_size + stride )/stride)

        for i in range(num_steps):
            for j in range(num_steps):
                new_r[i*stride:i*stride+stride, j*stride:j*stride+stride] = r[i*stride:i*stride+stride, j*stride:j*stride+stride] - np.mean(r[i*stride:i*stride+stride, j*stride:j*stride+stride])
                new_g[i*stride:i*stride+stride, j*stride:j*stride+stride] = g[i*stride:i*stride+stride, j*stride:j*stride+stride] - np.mean(g[i*stride:i*stride+stride, j*stride:j*stride+stride])
                new_b[i*stride:i*stride+stride, j*stride:j*stride+stride] = b[i*stride:i*stride+stride, j*stride:j*stride+stride] - np.mean(b[i*stride:i*stride+stride, j*stride:j*stride+stride])

        new_r = (new_r - np.min(new_r)) / (np.max(new_r) - np.min(new_r))
        new_g = (new_g - np.min(new_g)) / (np.max(new_g) - np.min(new_g))
        new_b = (new_b - np.min(new_b)) / (np.max(new_b) - np.min(new_b))
        new_rgb = np.stack((new_r, new_g, new_b), axis=-1)

        return new_rgb

def createModel(input_shape, NUM_CLASSES, INIT_LR = 1e-3, EPOCHS=10):
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
        model.add(Dense(output_dim=NUM_CLASSES, activation='softmax'))
        # returns our fully constructed deep learning + Keras image classifier 
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        # use binary_crossentropy if there are two classes
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model


if __name__ == "__main__":
    readData = read_data()
    raw_df = readData.readtrainCSV('trainLabels.csv')
    total_data = readData.readtrainData("sample/")
    pData = process_data()


    WIDTH = 512
    HEIGHT = 512
    DEPTH = 3

    sample_img = total_data['16_left']
    img_rgb_new = pData.enhanceImage(sample_img)

    plt.imshow(sample_img)
    plt.show()

    plt.imshow(img_rgb_new)
    plt.show()

    new_rgb = pData.colorNormalisation(img_rgb_new)

    plt.imshow(new_rgb)
    plt.show()

    class_weight = {0: 2.,
                    1: 15.,
                    2: 7.,
                    3: 73.,
                    4: 73.}
    
    input_shape = (HEIGHT, WIDTH, DEPTH)
    model = createModel(input_shape, 5)
