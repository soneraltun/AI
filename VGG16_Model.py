# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:16:06 2024

@author: Pc
"""
import tensorflow as tf
import pydicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import shutil
import gc
import SimpleITK as sitk
import BarChart as bar
from utils import *

from load_data import dataProcess
import os
import sys
import time


from scipy.ndimage import median_filter, gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy,SparseCategoricalAccuracy
from tensorflow.keras.models import Model, model_from_json
 
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Input

from skimage import io, exposure, img_as_float, transform, morphology
from tqdm import tqdm



num_classes = 2

" Parameters..."

l_rate = 0.000001
epochs = 100
batch_size=32

def GetModel_BatchSize():
    return batch_size

def GetModel_Epochs():
    return epochs

def GetModel_l_rate():
    return l_rate

def GetModel_num_classes():
    return num_classes

def VGG_model(num_classes):
    

    image_input = Input(shape=(224, 224, 3))

    
    base_model = VGG16(input_tensor = image_input, include_top = False, weights = 'imagenet')   

    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)
    '''
    # Freeze all the layers in the base model
    for layer in base_model.layers:
        if layer.name in ['block5_conv1']:
            layer.trainable = True
        else:
            layer.trainable = False
    '''
    # Freeze all layers except the last two
    for layer in base_model.layers:
        layer.trainable = False  # Varsayılan olarak tüm katmanları dondur
    
    # Son iki katmanı eğitilebilir yap
    for layer in base_model.layers[-8:]:
        layer.trainable = True
        

    base_model.summary()

    " build a classifier model to put on top of the convolutional model "    
    last_layer = base_model.output
    x= Flatten(name='flatten')(last_layer)
    #x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.8)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(image_input, out)
    model.summary()

    return model