########################## IMPORT LIBRARIES #############################
import glob, os, sys, random
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


################################### LIBRARIES IMPORTED #########################
class ProcessData():
	def __init__(self,train_image_dir=None, labels_dir=None, num_samples=None):
		self.train_images_dir = train_image_dir
		self.labels_dir = labels_dir
		self.num_samples = num_samples
		self.scale = 300

	def read_directory_images(self):
		# Returns the complete list of X and y inside directory.
		labels_df = pd.read_csv(self.labels_dir) #image, level
		image_col_labels_csv = labels_df['image'] 

		train_images_names = os.listdir(self.train_images_dir) # returns a list of names of images inside dir.

		if (self.num_samples != None):
			if (self.num_samples > len(train_images_names)):
				raise ValueError('Number of samples exceed the length of original data')
			else:
				train_images_names = random.sample(train_images_names, self.num_samples)
		
		#train_images_names = train_images_names.replace({'.jpeg':''}, regex=True)

		return train_images_names, labels_df


	def get_labels(self, train_images_names, labels_df):

		# Gets a train_X (which has names) and outputs a Y vector.

		train_images_df = pd.DataFrame()
		train_images_df["image"] = train_images_names # From Dir. don't contain
		train_images_df = train_images_df.replace({'.jpeg':''}, regex=True)
		# Shuffle the data.
		#train_images_df = train_images_df.sample(frac=1)

		train_x_names = train_images_df['image'].values

		matches_indices = []
		for i, row_data in enumerate(train_images_df['image']):
			matches_indices.append(labels_df.index[labels_df['image']==row_data].tolist()[0])
		

		# Get the Corresonding Labels on Labelscsv
		y = labels_df['level'].loc[matches_indices].values
		y = np.reshape(y, (len(y),1))
		
		### BELOW CODE IS JUST FOR CONFIRMATION THAT TRAIN_X and Y have been formed correctly.
		
		
		#temp_indices = np.argwhere(y==4)
		#temp_indices = temp_indices[:,0]
		#print ('temp indices are: ', temp_indices)
		#print (train_x_names[temp_indices])

		return train_x_names, y

	def print_class_weights(self,y):
		# y --> Output Labels
		size_y = len(y)
		unique, counts = np.unique(y, return_counts=True) 
		
		for class_num, class_weights in zip(unique, counts):
			print ('Class ', class_num, ' representation is: ', (class_weights/size_y)*100, '%')


	def load_images(self, train_X_names):

		train_stacked = np.zeros((train_X_names.shape[0], 512, 512, 3))
		i = 0
		for image_name in train_X_names:
			cur_path = train_images_dir+str(image_name)+'.jpeg'
			img_bgr = cv.imread(train_images_dir+str(image_name)+'.jpeg')
			processed_img = self.image_processing(img_bgr)

			train_stacked[i, :, :, :] = processed_img
			i += 1

		return train_stacked

	def scale_radius(self, img):

		x = img[int(img.shape[0]/2),:,:].sum(1)
		r=(x>x.mean()/10).sum()/2
		s = self.scale*1.0/r

		return cv.resize(img,(0,0),fx=s,fy=s)

	def image_processing(self, img):

		# Scale Image to a given radius
		a = self.scale_radius(img)

		a = cv.addWeighted (a,4, cv.GaussianBlur(a,(0,0),self.scale/30),-4,128)
		b = np.zeros(a.shape)
		cv.circle(b,(int(a.shape[1]/2), int(a.shape[0]/2)),int(self.scale*0.9),(1,1,1),-1,8,0)
		dst = a*b + 128*(1-b)
		dst = cv.resize(dst, (512,512))
		return dst

class TrainModel():
	def __init__(self):
		self.input_shape = (512,512,3)
		self.lr = 1e-3
		self.epochs = 30
		self.metrics="accuracy"
		self.loss="binary_crossentropy"
		self.batch_size = 32

	def create_model(self):
		model = Sequential()
		# first set of CONV => RELU => MAX POOL layers
		model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape = self.input_shape))
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
		opt = SGD(lr=self.lr, decay=0.0015, momentum=0.9)
		# use binary_crossentropy if there are two classes
		model.compile(loss=self.loss, optimizer=opt, metrics=[self.metrics])
		return model

	def fit_model(self, input_model):
		aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 
		horizontal_flip=True, fill_mode="nearest")
		history = input_model.fit_generator(aug.flow(train_X, train_y, batch_size=self.batch_size), validation_data=(val_X, val_y),
			steps_per_epoch=len(train_X)//self.batch_size, epochs=self.epochs, verbose=1)
		
		score, acc = input_model.evaluate(val_X, val_y, batch_size = self.batch_size, verbose=1)
		print ('Model Score is:', score)
		print ('Model Accuracy is:', acc)
		return history


	def plot_val_loss(self, history):
		H = history
		N = self.epochs
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
		plt.title("Training Loss and Accuracy on diabetic retinopathy detection")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.show()
		plt.savefig("val_loss_plot.png")


############################################## MAIN ##############################################
if __name__ == "__main__":

	
	train_images_dir = 'dataset/train/'
	labels_dir = 'dataset/trainLabels.csv'
	num_samples = 900
	
	DR_data = ProcessData(train_images_dir,labels_dir, num_samples) # DR --> Diabetic Retinography
	
	X_names, labels_df = DR_data.read_directory_images() # returns the names of images of X.

	train_X_names, val_X_names = train_test_split(X_names, test_size=0.20, random_state=42)
	
	train_X_names, train_y = DR_data.get_labels(train_X_names,labels_df)
	val_X_names, val_y = DR_data.get_labels(val_X_names,labels_df)
	#print (train_X_names)
	train_X = DR_data.load_images(train_X_names)
	val_X = DR_data.load_images(val_X_names)

	#print (train_X.shape)
	#print (train_y.shape)
	### See how do the classes weights look like
	print ('Training classes weights are as follows:')
	print(DR_data.print_class_weights(train_y))
	print ('Validation classes weights are as follows:')
	print(DR_data.print_class_weights(val_y))
	

	cnn_model_class = TrainModel()
	cnn_model = cnn_model_class.create_model()
	history = cnn_model_class.fit_model(cnn_model)
	cnn_model_class.plot_val_loss(history)
	


