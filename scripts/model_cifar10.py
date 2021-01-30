# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import random

from sklearn.utils import shuffle
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#Show random example in category
def display_category_examples(pred_val=None):
    """If no input, display one example for all category predictions. 
    If input is list, displays one example for listed categories. If input
    is integer, display one example for specified category."""
    if pred_val==None:
        ys=np.unique(y_train)
    elif type(pred_val)==list:
        ys=pred_val
    elif type(pred_val)==int:
        ys=[pred_val]
    else:
        raise "Invalid Input"
        
    for possible_y in ys:
        img_idx=random.choice(np.where(y_train==possible_y)[0])
        plt.imshow(PIL.Image.fromarray(x_train[img_idx]))
        plt.title(str(possible_y))
        plt.show()
        
#Better Method
def display_category_examples_2():
    class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website
    
    plt.figure(figsize=(10,10))
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_im[i], cmap='gray')
        plt.xlabel(class_types[train_lab[i][0]], fontsize=13)
    plt.tight_layout()    
    plt.show()
#%% Restrict categories [i.e. cats vs dogs]
# =============================================================================
# train_idx=np.append(np.where(y_train==3)[0], np.where(y_train==5)[0])
# test_idx=np.append(np.where(y_test==3)[0], np.where(y_test==5)[0])
# 
# x_train, y_train=(x_train[train_idx], y_train[train_idx])
# x_test, y_test=(x_test[test_idx], y_test[test_idx])
# 
# #Convert y values: 3 (Cats) and 5 (Dogs) to 0 and 1
# y_train=np.array([0 if i==3 else 1 for i in y_train])
# y_test=np.array([0 if i==3 else 1 for i in y_test])
# =============================================================================

#%% Pre-processing model inputs
#Normalize Image Arrays to [0-1]
# =============================================================================
# x_train, x_test = x_train/255.0,  x_test/255.0
# =============================================================================
    
x_train=tensorflow.keras.applications.resnet.preprocess_input(x_train)
x_test=tensorflow.keras.applications.resnet.preprocess_input(x_test)

#Reshape arrays
y_train=y_train.flatten().reshape((y_train.shape[0],1))
y_test=y_test.flatten().reshape((y_test.shape[0],1))

#Shuffle arrays
# =============================================================================
# x_train,y_train=shuffle(x_train,y_train)
# x_test,y_test=shuffle(x_test,y_test)
# x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.7, test_size=0.3)
# =============================================================================
#%% MODEL
# =============================================================================
# base_model_vgg16 = VGG16(include_top = False, weights='imagenet', input_shape = (32,32,3), classes = 2)
# base_model_vgg16.trainable = False
# =============================================================================
base_model_resnet50=ResNet50(include_top = False, weights='imagenet', input_shape = (224,224,3), classes = 10)
base_model_resnet50.trainable=False

# =============================================================================
# for layer in base_model_resnet50.layers[:]:
#     layer.trainable=False
# =============================================================================

# =============================================================================
# model.add(Conv2D(16, kernel_size=(5, 5), 
#                  strides=(1, 1),
#                  activation='relu',
#                  kernel_initializer="normal",
#                  input_shape=(32,32,3),
#                  name="conv_1"
#                  ))
# model.add(MaxPool2D(pool_size=(2, 2), 
#                     strides=(1, 1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(8, (3, 3), 
#                  activation='relu', 
#                  kernel_initializer="normal",
#                  name="conv_2"))
# model.add(MaxPool2D(pool_size=(2, 2),
#                       name="maxpool_1"))
# model.add(Dropout(0.2))
# model.add(Flatten(name="flatten"))
# model.add(Dense(4, 
#                 kernel_initializer="normal",
#                 activation="relu"))
# =============================================================================
# =============================================================================
# model.add(base_model_resnet50)
# model.add(Dense(512,
#                 activation="relu"))
# model.add(Dense(32,
#                 activation="relu"))
# model.add(Dense(10,
#                 activation="softmax"))
# model.summary()
# 
# =============================================================================

input=tf.keras.Input(shape=(32,32,3))
reshaped_input=tf.keras.layers.Lambda(lambda x: tf.image.resize_with_pad(x, 224, 224))(input)
base_model=base_model_resnet50(reshaped_input)
flatten=Flatten()(base_model)
output=Dense(10, activation="softmax")(flatten)

model=tf.keras.Model(input, output)
#%%Hyper-parameters
n_batch_size=10
num_epoch=50
learn_rate=5e-5

#%%Compiling VGG16
sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)
model.compile(optimizer = sgd, loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

# =============================================================================
# callbacks = [
#     EarlyStopping(patience=20, verbose=1),
#     ReduceLROnPlateau(factor=0.01, patience=10, min_lr=1e-7, verbose=1)]
# =============================================================================

history = model.fit(x_train, y_train, 
                    batch_size=n_batch_size,
                    epochs=num_epoch,
                    validation_data=(x_val, y_val),
                    verbose=1, 
                    shuffle=True)


#%%
# =============================================================================
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# datagen.fit(x_train)
# history = model.fit(datagen.flow(x_train, y_train, batch_size=n_batch_size), epochs=num_epoch)
# =============================================================================
#%% Evaluate on Test
y_pred=np.argmax(model.predict(x_test), axis=1)
cm=confusion_matrix(y_test, y_pred)

# Sensitivity, Specificity, Precision
true_negative, false_positive, false_negative, true_positive  = cm.ravel()
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)

print('Precison:{:.2f}'.format(precision))
print('Sensitivity:{:.2f}'.format(recall))
print('Specificity:{:.2f}'.format(specificity))

#AUC
# =============================================================================
# auc = roc_auc_score(y_test, y_pred)
# print(auc)
# =============================================================================
