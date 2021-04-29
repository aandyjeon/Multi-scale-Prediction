#!/usr/bin/env python
# coding: utf-8

import cv2
import glob
import numpy as np
import os.path as path
import os
from scipy import misc
import re
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.models import Model, Input
from keras import models
from keras import layers
import keras
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import pandas as pd
import sklearn.metrics as sm
from keras import backend as K

from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Add

os.environ['CUDA_VISIBLE_DEVICES']='0'

weight_file = './model_weight/multi/lr_original_case1_3/proposed_method_0.9795-64-0.0751.hdf5'

img_rows, img_cols, channel = 224, 224, 3 # Resolution of inputs
    
with open ('./datafiles_aug/x_test_fismo', 'rb') as fp:
    x_test_fismo = pickle.load(fp)
with open ('./datafiles_aug/x_test_bow', 'rb') as fp:
    x_test_bow = pickle.load(fp)
with open ('./datafiles_aug/x_test_sharma', 'rb') as fp:
    x_test_sharma = pickle.load(fp)

print('load input data clear~!! =========================================')

with open ('./datafiles_aug/y_test_fismo', 'rb') as fp:
    y_test_fismo = pickle.load(fp)
with open ('./datafiles_aug/y_test_bow', 'rb') as fp:
    y_test_bow = pickle.load(fp)
with open ('./datafiles_aug/y_test_sharma', 'rb') as fp:
    y_test_sharma = pickle.load(fp)

print('load label data clear~!! =========================================')
print('load dataset clear~!! ============================================')
    
print('x_test_fismo: '+ str(np.shape(x_test_fismo)))
print('y_test_fismo: ' + str(np.shape(y_test_fismo)))
print('==================================================================')
print('x_test_bow: '+ str(np.shape(x_test_bow)))
print('y_test_bow: ' + str(np.shape(y_test_bow)))
print('==================================================================')
print('x_test_sharma: '+ str(np.shape(x_test_sharma)))
print('y_test_sharma: ' + str(np.shape(y_test_sharma)))

x_test_fismo = np.array(x_test_fismo) 
y_test_fismo = np.array(y_test_fismo)
x_test_bow = np.array(x_test_bow) 
y_test_bow = np.array(y_test_bow)
x_test_sharma = np.array(x_test_sharma) 
y_test_sharma = np.array(y_test_sharma)
x_test = np.concatenate([x_test_fismo,x_test_bow,x_test_sharma])/ 255.
y_test = np.concatenate([y_test_fismo,y_test_bow,y_test_sharma])

print('x_test: '+ str(np.shape(x_test)))
print('y_test: ' + str(np.shape(y_test)))

print('dataset shape check clear~!! =====================================')


# parameter setting

img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 2 
batch_size = 64
nb_epoch = 50


def conv_pool_softmax(x, input_size):
    conv_output = Conv2D(filters = 2,
        kernel_size = (1,1), 
        padding = 'same')(x)
    pool_output = MaxPooling2D(pool_size = (input_size,input_size), strides = 1)(conv_output)
    flatten = Flatten()(pool_output)
    output = Dense(2, activation = 'softmax')(flatten)
    
    return output


def dcr_block(x, nb_filters):
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
    dcr_output_1 = dcr_block(x, nb_filters)
    dcr_output_2 = dcr_block(dcr_output_1, nb_filters)
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
    dcr_output_1 = dcr_block(x, nb_filters)
    dcr_output_2 = dcr_block(dcr_output_1, nb_filters)
    
    # pooling feature map size
    conv = Conv2D(filters = nb_filters,
                   kernel_size = (3,3),
                   strides = [2,2],
                   padding = 'same')(dcr_output_2)
    batch = BatchNormalization()(conv)
    output = Activation('relu')(batch)
    
    return output


def DCR_block_classification(img_rows, img_cols, channel):
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
    fire_result = keras.layers.Lambda(lambda x: x[:,1::2])(result)
    no_fire_result = keras.layers.Lambda(lambda x: x[:,::2])(result)
    
    fire_score = Dense(1)(fire_result)
    no_fire_score = Dense(1)(no_fire_result)
    
    output = keras.layers.Concatenate()([fire_score, no_fire_score])
    output_ = Dense(2, activation = 'softmax')(output)

    model = keras.models.Model(inputs = model_input, outputs = output_)
    return model

# accuracy test part

loaded_model = DCR_block_classification(img_rows, img_cols, channel)
loaded_model.load_weights(weight_file)
opt_adam = keras.optimizers.Adam(lr=0.0001)
loaded_model.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=['acc'])


loaded_model.summary()

def atoi(text):
    return int(text) if text.isdigit() else text
    
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


loss_and_metrics = loaded_model.evaluate(np.array(x_test_fismo) / 255., np.array(y_test_fismo), batch_size=128)
print('')
print('loss_and_metrics_fismo : ' + str(loss_and_metrics))
loss_and_metrics = loaded_model.evaluate(np.array(x_test_bow) / 255., np.array(y_test_bow), batch_size=128)
print('')
print('loss_and_metrics_bow : ' + str(loss_and_metrics))
loss_and_metrics = loaded_model.evaluate(np.array(x_test_sharma) / 255., np.array(y_test_sharma), batch_size=128)
print('')
print('loss_and_metrics_sharma : ' + str(loss_and_metrics))


loss_and_metrics = loaded_model.evaluate(np.array(x_test), np.array(y_test), batch_size=128)
print('')
print('loss_and_metrics_all : ' + str(loss_and_metrics))


predictions_entire = loaded_model.predict(x_test, batch_size=128, verbose=1)
predictions_fismo = loaded_model.predict(np.array(x_test_fismo) / 255. , batch_size=128, verbose=1)
predictions_bow = loaded_model.predict(np.array(x_test_bow) / 255. , batch_size=128, verbose=1)
predictions_sharma = loaded_model.predict(np.array(x_test_sharma) / 255. , batch_size=128, verbose=1)


def prediction_result(predictions_valid):
    result_list = []
    for i in range(len(predictions_valid)):
        if (predictions_valid[i][0] > predictions_valid[i][1]):
            result_list.append(0)
        else : result_list.append(1)
    return result_list


predictions_entire_tmp = prediction_result(predictions_entire)
y_test_tmp = prediction_result(y_test)
predictions_fismo_tmp = prediction_result(predictions_fismo)
y_test_fismo_tmp = prediction_result(y_test_fismo)
predictions_bow_tmp = prediction_result(predictions_bow)
y_test_bow_tmp = prediction_result(y_test_bow)
predictions_sharma_tmp = prediction_result(predictions_sharma)
y_test_sharma_tmp = prediction_result(y_test_sharma)


print('PRECISION')
print  sm.precision_score(y_test_tmp, predictions_entire_tmp)
print  sm.precision_score(y_test_fismo_tmp, predictions_fismo_tmp)
print  sm.precision_score(y_test_bow_tmp, predictions_bow_tmp)
print  sm.precision_score(y_test_sharma_tmp, predictions_sharma_tmp)

print('')
print('RECALL')
print  sm.recall_score(y_test_tmp, predictions_entire_tmp)
print  sm.recall_score(y_test_fismo_tmp, predictions_fismo_tmp)
print  sm.recall_score(y_test_bow_tmp, predictions_bow_tmp)
print  sm.recall_score(y_test_sharma_tmp, predictions_sharma_tmp)

print('')
print('F1SCORE')
print  sm.f1_score(y_test_tmp, predictions_entire_tmp)
print  sm.f1_score(y_test_fismo_tmp, predictions_fismo_tmp)
print  sm.f1_score(y_test_bow_tmp, predictions_bow_tmp)
print  sm.f1_score(y_test_sharma_tmp, predictions_sharma_tmp)


print 'all :'
print sm.confusion_matrix(y_test_tmp, predictions_entire_tmp)
print 'fismo :'
print sm.confusion_matrix(y_test_fismo_tmp, predictions_fismo_tmp)
print 'bow :'
print sm.confusion_matrix(y_test_bow_tmp, predictions_bow_tmp)
print 'sharma :'
print sm.confusion_matrix(y_test_sharma_tmp, predictions_sharma_tmp)


Pval, Nval = sm.confusion_matrix(y_test_tmp, predictions_entire_tmp)
_, Fp = Pval
_, Tn = Nval
FAR = float(Fp)/(Tn+Fp)
print("FAR value - all : %f"%FAR)

Pval, Nval = sm.confusion_matrix(y_test_fismo_tmp, predictions_fismo_tmp)
_, Fp = Pval
_, Tn = Nval
FAR = float(Fp)/(Tn+Fp)
print("FAR value - fismo : %f"%FAR)

Pval, Nval = sm.confusion_matrix(y_test_bow_tmp, predictions_bow_tmp)
_, Fp = Pval
_, Tn = Nval
FAR = float(Fp)/(Tn+Fp)
print("FAR value - bow : %f"%FAR)

Pval, Nval = sm.confusion_matrix(y_test_sharma_tmp, predictions_sharma_tmp)
_, Fp = Pval
_, Tn = Nval
FAR = float(Fp)/(Tn+Fp)
print("FAR value - sharma : %f"%FAR)
