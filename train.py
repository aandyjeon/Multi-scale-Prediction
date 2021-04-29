# coding: utf-8
import cv2
import glob
import numpy as np
import os.path as path
import os
from scipy import misc
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.models import Model, Input
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import math
import keras
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import pandas as pd
import sklearn.metrics as sm

from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Add
from cosine_annealing import CosineAnnealingScheduler

# parameter setting======================================================================

os.environ['CUDA_VISIBLE_DEVICES']='1'

img_rows, img_cols = 224, 224
channel = 3
num_classes = 2 
batch_size = 64
nb_epoch = 200

initial_lrate = 0.001 #lr_schedule below
opt_adam = keras.optimizers.Adam(lr=initial_lrate)
optimizer = opt_adam
MODEL_SAVE_FOLDER_PATH = './model_weight/multi/bow_sharma/lr_original_case1/'

#========================================================================================


# load dataset
with open ('./datafiles_aug/x_train_fismo', 'rb') as fp:
    x_train_fismo = pickle.load(fp)
with open ('./datafiles_aug/x_train_bow', 'rb') as fp:
    x_train_bow = pickle.load(fp)
with open ('./datafiles_aug/x_train_sharma', 'rb') as fp:
    x_train_sharma = pickle.load(fp)
with open ('./datafiles_aug/x_test_fismo', 'rb') as fp:
    x_test_fismo = pickle.load(fp)
with open ('./datafiles_aug/x_test_bow', 'rb') as fp:
    x_test_bow = pickle.load(fp)
with open ('./datafiles_aug/x_test_sharma', 'rb') as fp:
    x_test_sharma = pickle.load(fp)

print('load input data clear~!! =========================================')

with open ('./datafiles_aug/y_train_fismo', 'rb') as fp:
    y_train_fismo = pickle.load(fp)
with open ('./datafiles_aug/y_test_fismo', 'rb') as fp:
    y_test_fismo = pickle.load(fp)
with open ('./datafiles_aug/y_train_bow', 'rb') as fp:
    y_train_bow = pickle.load(fp)
with open ('./datafiles_aug/y_test_bow', 'rb') as fp:
    y_test_bow = pickle.load(fp)
with open ('./datafiles_aug/y_train_sharma', 'rb') as fp:
    y_train_sharma = pickle.load(fp)
with open ('./datafiles_aug/y_test_sharma', 'rb') as fp:
    y_test_sharma = pickle.load(fp)

print('load label data clear~!! =========================================')
print('load dataset clear~!! ============================================')
    
print('x_train_fismo: ' + str(np.shape(x_train_fismo)))
print('x_test_fismo: '+ str(np.shape(x_test_fismo)))
print('y_train_fismo: ' + str(np.shape(y_train_fismo)))
print('y_test_fismo: ' + str(np.shape(y_test_fismo)))
print('==================================================================')
print('x_train_bow: ' + str(np.shape(x_train_bow)))
print('x_test_bow: '+ str(np.shape(x_test_bow)))
print('y_train_bow: ' + str(np.shape(y_train_bow)))
print('y_test_bow: ' + str(np.shape(y_test_bow)))
print('==================================================================')
print('x_train_sharma: ' + str(np.shape(x_train_sharma)))
print('x_test_sharma: '+ str(np.shape(x_test_sharma)))
print('y_train_sharma: ' + str(np.shape(y_train_sharma)))
print('y_test_sharma: ' + str(np.shape(y_test_sharma)))

x_train = np.concatenate([x_train_fismo,x_train_bow,x_train_sharma]) / 255.
x_test = np.concatenate([x_test_fismo,x_test_bow,x_test_sharma]) / 255.
#x_train = np.concatenate([x_train_bow,x_train_sharma]) / 255.
#x_test = np.concatenate([x_test_bow,x_test_sharma]) / 255.
x_train_fismo = np.array(x_train_fismo) / 255.
x_train_bow = np.array(x_train_bow) / 255.
x_train_sharma = np.array(x_train_sharma) / 255.
x_test_fismo = np.array(x_test_fismo) /255.
x_test_bow = np.array(x_test_bow) /255.
x_test_sharma = np.array(x_test_sharma) /255.
y_train_fismo = np.array(y_train_fismo)
y_train_bow = np.array(y_train_bow)
y_train_sharma = np.array(y_train_sharma)
y_test_fismo = np.array(y_test_fismo)
y_test_bow = np.array(y_test_bow)
y_test_sharma = np.array(y_test_sharma)
y_train = np.concatenate([y_train_fismo,y_train_bow,y_train_sharma])
y_test = np.concatenate([y_test_fismo,y_test_bow,y_test_sharma])
#y_train = np.concatenate([y_train_bow,y_train_sharma])
#y_test = np.concatenate([y_test_bow,y_test_sharma])

print('x_train: ' + str(np.shape(x_train)))
print('x_test: '+ str(np.shape(x_test)))
print('y_train: ' + str(np.shape(y_train)))
print('y_test: ' + str(np.shape(y_test)))

print('dataset shape check clear~!! =====================================')


#====================================================================================

def conv_pool_softmax(x, input_size):
    """
	CMFC block
	x: input
	input_size: w, h of input
	"""
    conv_output = Conv2D(filters = 2,
        kernel_size = (1,1), 
        padding = 'same')(x)
    pool_output = MaxPooling2D(pool_size = (input_size,input_size), strides = 1)(conv_output)
    flatten = Flatten()(pool_output)
    output = Dense(2, activation = 'softmax')(flatten)
    
    return output


def __dcr_block__(x, nb_filters):
    x_conv_1 = Conv2D(filters = int((0.5)*nb_filters),
        kernel_size = (3,3), 
        padding = 'same')(x)
    x_batch_1 = BatchNormalization()(x_conv_1)
    x_relu_1 = Activation('relu')(x_batch_1)
    x_concat_1 = Concatenate()([x, x_relu_1])
    
    num_filters = int((0.5) * nb_filters)
    x_conv_2 = Conv2D(filters = num_filters,
        kernel_size = (3,3), 
        padding = 'same')(x_concat_1)
    x_batch_2 = BatchNormalization()(x_conv_2)
    x_relu_2 = Activation('relu')(x_batch_2)
    x_concat_2 = Concatenate()([x, x_relu_1, x_relu_2])
    
    x_conv_3 = Conv2D(filters = nb_filters,
        kernel_size = (3,3), 
        padding = 'same')(x_concat_2)
    x_batch_3 = BatchNormalization()(x_conv_3)
    x_concat_3 = Add()([x, x_batch_3])
    
    output = Activation('relu')(x_concat_3)
    
    return output


def DCR_block_sizeup(x, nb_filters):
    """
	DCR_block (the number of channels increase to double)
	x: input
	nb_filters: the number of input channels
	"""
    dcr_output_1 = __dcr_block__(x, nb_filters)
    dcr_output_2 = __dcr_block__(dcr_output_1, nb_filters)
    num_filters = 2*nb_filters
    
    # pooling feature map size
    conv = Conv2D(filters = num_filters,
                   kernel_size = (3,3),
                   strides = [2,2],
                   padding = 'same')(dcr_output_2)
    batch = BatchNormalization()(conv)
    output = Activation('relu')(batch)
    return output


def DCR_block_retain(x, nb_filters):
    """
	DCR_block (the number of channels is maintained)
	x: input
	nb_filters: the number of input channels
	"""
    dcr_output_1 = __dcr_block__(x, nb_filters)
    dcr_output_2 = __dcr_block__(dcr_output_1, nb_filters)
    
    # pooling feature map size
    conv = Conv2D(filters = nb_filters,
                   kernel_size = (3,3),
                   strides = [2,2],
                   padding = 'same')(dcr_output_2)
    batch = BatchNormalization()(conv)
    output = Activation('relu')(batch)
    
    return output


def proposed_model(img_rows, img_cols, channel):
    model_input = keras.layers.Input(shape=(img_rows, img_cols, channel))
    entrance_featuremap = Conv2D(filters = 64,
                                kernel_size = (7, 7),
                                strides = [2,2],
                                input_shape = (img_rows, img_cols, channel),
                                padding = 'same')(model_input)
    entrance_pooling = MaxPooling2D(pool_size = (3,3), strides = 2)(entrance_featuremap)
    
    dcr_output_1 = DCR_block_sizeup(entrance_pooling, 64)
    dcr_output_2 = DCR_block_sizeup(dcr_output_1, 128)
    dcr_output_3 = DCR_block_sizeup(dcr_output_2, 256)
    
    dcr_output_4 = DCR_block_retain(dcr_output_3, 512)
    dcr_output_5 = DCR_block_retain(dcr_output_4, 512)
    dcr_output_6 = DCR_block_retain(dcr_output_5, 512)
    
    result_1 = conv_pool_softmax(dcr_output_1, 28)
    result_2 = conv_pool_softmax(dcr_output_2, 14)
    result_3 = conv_pool_softmax(dcr_output_3, 7)
    result_4 = conv_pool_softmax(dcr_output_4, 4)
    result_5 = conv_pool_softmax(dcr_output_5, 2)
    result_6 = conv_pool_softmax(dcr_output_6, 1)
    
    result = keras.layers.concatenate([result_1, result_2, result_3, result_4, result_5, result_6])
    #fire_scores from diffrent resolution of results
    fire_result = keras.layers.Lambda(lambda x: x[:,1::2])(result) 
    #no_fire_scores from different resolution of results
    no_fire_result = keras.layers.Lambda(lambda x: x[:,::2])(result) 
    
    fire_score = Dense(1)(fire_result)
    no_fire_score = Dense(1)(no_fire_result) 
    
    output = keras.layers.Concatenate()([fire_score, no_fire_score])
    output_ = Dense(2, activation = 'softmax')(output)

    model = keras.models.Model(inputs = model_input, outputs = output_)
    
    return model


model = proposed_model(img_rows, img_cols, channel)

model.summary()

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)    

model_path = MODEL_SAVE_FOLDER_PATH + 'proposed_method_' + '{val_accuracy:.4f}-{epoch:02d}-{val_loss:.4f}.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [0.1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def step_decay(losses):
    
    if float(2*np.sqrt(np.array(history.losses[-1])))<0.5:
        lrate=0.01*1/(1 + len(history.losses))
        return lrate
    else:
        lrate=1e-4
    
    return lrate


history=LossHistory()
lrate=LearningRateScheduler(step_decay)
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph/bow_sharma', histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

history = model.fit(x_train_bow, y_train_bow,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(x_test_bow, y_test_bow),
              callbacks=[history,lrate, cb_checkpoint, tb_hist])

predictions_valid = model.predict(x_test_bow, batch_size=batch_size, verbose=1)

# Combine 3 set of outputs using averaging
predictions_valid = sum(predictions_valid)/len(predictions_valid)
