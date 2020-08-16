import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disables GPU

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization

# importing dataset
training_set = pd.read_csv("../dataset/train.csv")

# to ndarray
X_train = training_set.iloc[:,1:].values
y_train = training_set.iloc[:,0].values.astype('int32')
y_train = to_categorical(y_train)

# normalization
X_train = X_train/255

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10)

# to tensors
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# image augmentation object
datagen = ImageDataGenerator(
          featurewise_center=False,
          samplewise_center=False,
          featurewise_std_normalization=False,
          samplewise_std_normalization=False,
          zca_whitening=False,
          rotation_range=10,
          zoom_range = 0.1, 
          width_shift_range=0.1,
          height_shift_range=0.1,
          horizontal_flip=False,
          vertical_flip=False)

# applying augmentation
train_gen = datagen.flow(X_train, y_train, batch_size=64)
test_gen = datagen.flow(X_test, y_test, batch_size=64)

# Convolutional Neural Network
cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
cnn.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(filters=56, kernel_size = (3,3), activation="relu"))

cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(BatchNormalization())    
cnn.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    
cnn.add(MaxPooling2D(pool_size=(2,2)))
    
cnn.add(Flatten())
cnn.add(BatchNormalization())
cnn.add(Dense(512,activation="relu"))
    
cnn.add(Dense(10,activation="softmax"))

# compiling the CNN    
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fitting data
cnn.fit(train_gen, epochs = 3, validation_data = test_gen)                                

# serializing model to JSON
model_json = cnn.to_json()
with open("../models/model.json", "w") as json_file:
    json_file.write(model_json)
    
# serializing weights to HDF5
cnn.save_weights("../models/model.h5")
print("Saved model to disk.")

# printing a sample
y_pred = cnn.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
