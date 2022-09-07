#Importing Different Packages Supported to Python
import os
import numpy as np
import cv2
import random
#Stopping Randomness
random.seed(101)

#One Hot Encoder
def one_hot_encoder(y):
  Encoded_y = np.array([[0.]*np.shape(y)[0]]*len(classes))
  for i in range(np.shape(y)[0]):
    Encoded_y[int(y[i])][i] = 1.
  return np.transpose(np.array(Encoded_y))

#Load Dataset and Preprocessing
def load_data(size,root_path):
    
  X_train = []
  y_train = []
  patient_list = os.listdir(root_path)

  for patient in patient_list:
    patient_path = root_path + '/' + patient
    for study in os.listdir(patient_path):
        if study == 'study1_negative' or study == 'study2_negative' or study == 'study3_negative' or study == 'study4_negative':
            for i in os.listdir(patient_path + '/' + study):
                image_path = patient_path + '/' + study + '/' + i
                img = cv2.imread(image_path)
                img = cv2.resize(img, size)
                img = np.array(img)
                X_train.append(img / 255)
                y_train.append(0)
        elif study == 'study1_positive' or study == 'study2_positive' or study == 'study3_positive' or study == 'study4_positive':
            for i in os.listdir(patient_path + '/' + study):
                image_path = patient_path + '/' + study + '/' + i
                img = cv2.imread(image_path)
                img = cv2.resize(img, size)
                img = np.array(img)
                X_train.append(img / 255)
                y_train.append(1)
        
  X_train = np.array(X_train)
  y_train = one_hot_encoder(np.array(y_train))
  return X_train, y_train

#Different Classes
classes = {0:'Negative', 1:'Positive'}
#Loading Dataset
print("Loading Train Data...")
train_path = 'MURA-v1.1/train/XR_ELBOW'
X_train, y_train = load_data((128,128),train_path)
print("Training Data Loaded...")

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

#Importing Packages for Model Creation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Model Creation

def CNN_model(filters,filter_org,kernel_size,dropout,activation,batch_norm,dense_layer,input_shape,pool_size,hidden_layers):
    model = Sequential()
    
    #Setting Number of Filters depending on the Filter Organization
    if filter_org == 'half':
        k = 0.5
    elif filter_org == 'double':
        k = 2
    else:
        k = 1
        
    for i in range(hidden_layers):
        model.add(Conv2D(filters, kernel_size = kernel_size, input_shape=input_shape))
        filters = int(filters*k)
        model.add(Activation(activation))
        
        #if batch_norm is true, then only we will add batch_normalization
        if batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=pool_size))
        
    #Flattening the dimension
    model.add(Flatten())
    
    #adding Dense Layer
    model.add(Dense(dense_layer,activation=activation))
    
    #Adding Dropouts
    model.add(Dropout(dropout))
    if batch_norm:
        model.add(BatchNormalization())
        
    #Finally Adding the Output Layer
    model.add(Dense(2, activation = "softmax"))
    
    return model


#Taking Input in Command Line
# All commandline parameters

print("Please Give the following arguments...")

filters = int(input("Filters (i.e. 32, 64) = "))
filter_org = str(input("Filter Organization (half/same/double) = "))
k = int(input("Kernel Size k*k, k (i.e. 3, 5) = "))
kernel_size = (k,k)
dropouts = float(input("Dropouts (i.e. 0.2, 0.3, 0.5) = "))
dense_layer = int(input("Dense Layers (i.e. 64, 128, 256) = "))
batch_size = int(input("Batch Size (i.e. 64, 128, 256) = "))
hidden_layers = int(input("Size of Hidden Layers (i.e. 5, 10, 15) = "))
epochs = int(input("Number of Epochs (i.e. 5, 10, 15) = "))

#Training Model and Evaluate on Test Data
print("Calling Model...")

model = CNN_model(
        filters = filters,
        filter_org = filter_org,
        kernel_size = kernel_size,
        dropout = dropouts,
        activation = 'relu',
        batch_norm = False,
        dense_layer = dense_layer,
        input_shape = (128,128,3),
        pool_size = (2,2),
        hidden_layers = hidden_layers)
model.compile(loss = 'categorical_crossentropy', optimizer = 'nadam', metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1.0)
train_datagen.fit(X_train,seed = 101)
model.fit(
        train_datagen.flow(X_train,y_train, batch_size = batch_size,seed = 101),
        epochs = epochs,
        validation_data = (X_valid,y_valid),
    )


#Loading Test Data
print("Loading Test Data...")
test_path = 'MURA-v1.1/valid/XR_ELBOW'
X_test, y_test = load_data((128,128),test_path)
test_generator = ImageDataGenerator(rescale = 1.0)
test_generator.fit(X_test,seed=101)
print("Test Data Loaded...")

#evaluate on test data
print("Test Evaluation Started...")
model.evaluate(test_generator.flow(X_test,y_test,seed=101))
print("Test Evaluation Ended...")