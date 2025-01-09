# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:46:39 2024

@author: Pc

"""
import tensorflow as tf

from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from MobileNetModel import MobileNet_model,GetModel_BatchSize,GetModel_Epochs,GetModel_l_rate,GetModel_num_classes

from tensorflow.keras import backend as K
from tensorflow.keras.utils import img_to_array
import pydicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import shutil
import gc
import SimpleITK as sitk

import BarChart as bar
import numpy as np

#from utils import *
#from load_data import dataProcess

import os
import sys
import time

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import Callback,ReduceLROnPlateau
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy,Precision, Recall,AUC
from tensorflow.keras.models import Model
from skimage import io, exposure, img_as_float, transform, morphology
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img

" Parameters..."
batch_size = GetModel_BatchSize()
l_rate = GetModel_l_rate()
epochs = GetModel_Epochs()
num_classes = GetModel_num_classes()

short_names = {
    'Hernia': 'HN',
    'Cardiomegaly': 'CM',
    'Infiltration': 'IN',
    'Atelectasis': 'AT',
    'Mass': 'Mass',
    'Nodule': 'Nodule',
    'Pneumonia': 'PN',
    'Pneumothorax': 'PT',
    'Effusion': 'EF',
    'Consolidation': 'CS',
    'Edema': 'ED',
    'Emphysema': 'EM',
    'No Finding': 'No Finding',
    'Fibrosis': 'FB',
    'Pleural_Thickening': 'PTK',
}

DATABASEPATH = os.path.dirname(os.path.abspath(__file__))
PATH=os.path.join(DATABASEPATH,"NIH Chest X-ray Dataset")

'''
# En düşük 3 değeri bul
top_counts = df['Finding Labels'].value_counts().nsmallest(num_classes)
bottom_labels = top_counts.index

# Yeni DataFrame oluştur
df = df[df['Finding Labels'].isin(bottom_labels)]

label_processor = tf.keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(df["Finding Labels"])
)
print(label_processor.get_vocabulary())
'''


filename = 'Train_Entry.csv'
DataEntryPATH=os.path.join(PATH, filename)

train_df=pd.read_csv(DataEntryPATH)

train_df = train_df[train_df['Folder'] != 'aug_img']

#train_df = train_df[train_df['Finding Labels'] != 'Hernia']

train_df = train_df.reset_index(drop=True)

filename = 'Val_Entry.csv'
DataEntryPATH=os.path.join(PATH, filename)

val_df=pd.read_csv(DataEntryPATH)

#val_df = val_df[val_df['Finding Labels'] != 'Hernia']
val_df = val_df.reset_index(drop=True)


filename = 'Test_Entry.csv'
DataEntryPATH=os.path.join(PATH, filename)

test_df=pd.read_csv(DataEntryPATH)

test_df = test_df[test_df['Finding Labels'] != 'Hernia']
test_df = test_df.reset_index(drop=True)

label_processor = tf.keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(test_df["Finding Labels"])
)
print(label_processor.get_vocabulary())


train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
'''
train_df=, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
'''

" read train data...both department..."


def readXray(file_name_with_path,file_name):
 
   
   " check file type " 
   main_fn, ext_fn = os.path.splitext(file_name)
   
 
   if(ext_fn == '.dcm'): 
       itk_image = sitk.ReadImage(file_name_with_path)
       img = sitk.GetArrayFromImage(itk_image)
       img = np.squeeze(img)
   else: 
       #img = img_as_float(io.imread(file_name_with_path))
       img = load_img(file_name_with_path, target_size=(224, 224), color_mode="rgb")

       img = img_to_array(img)
       
   if(len(img.shape) == 2): 
       img = np.expand_dims(img, axis=0)

   '''
   # Preprocess öncesi
   plt.imshow(img.astype('uint8'))  # Orijinal görüntü
   plt.title("Original Image")
   plt.show()
   '''
   #img = preprocess_input(img)
   
   '''
   # Preprocess sonrası (normalize edilmiş) 
   plt.imshow((img - img.min()) / (img.max() - img.min()))
   plt.title("Preprocessed Image")
   plt.show()
   '''
   #img = preprocess_input(img)
   #img=img/255.0
   return img
def GetImages(df):
    
    Data, Label = [], []
    
    for index, row in df.iterrows():
        filename = row['Image Index']
        filefolder= row['Folder']
        cropfolderName="CroppedImages"
        
        ImagePath=os.path.join(PATH,filefolder)
        ImagePath=os.path.join(ImagePath,cropfolderName)
        ImagePath=os.path.join(ImagePath,filename)
     
        img=readXray(ImagePath,filename)
        '''
        # Görüntü geçerli mi kontrol et
        if img is None:
            raise ValueError("Görüntü yüklenemedi. Lütfen geçerli bir dosya yolu kullanın.")
            
        img = cv2.resize(img, (224, 224),cv2.INTER_CUBIC)
        # Görüntüyü göstermek için imshow
        


        if len(img.shape) == 2:  # Gri tonlamalı kontrolü
            img = np.expand_dims(img, axis=-1)  # (yükseklik, genişlik, 1)
            img = np.repeat(img, 3, axis=-1)  # (yükseklik, genişlik, 3)

        # prepare the image for the VGG model
        img = preprocess_input(img)
        # VGG16 için ön işleme (normalize etme)
        #img = preprocess_input(img)  # VGG16 için uygun şekilde normalize etme
        
        '''
        img = preprocess_input(img)
        Data.append(img)
        Label.append(row['Finding Labels'])

    return Data,Label

# Custom callback to trigger gc.collect() at a certain step
class GcCallback(Callback):
    def __init__(self, step_interval=1000):
        super().__init__()
        self.step_interval = step_interval

    def on_batch_end(self, batch, logs=None):
        if batch % self.step_interval == 0:
            print(f"Step {batch}: Garbage collection triggered")
            gc.collect()

group_counts = train_df['Finding Labels'].value_counts()
print(group_counts)

TrainData,TrainLabel=GetImages(train_df)
ValData,ValLabel=GetImages(val_df)
#TestData,TestLabel=GetImages(test_df)

gc.collect()
TrainData = np.array(TrainData)
ValData = np.array(ValData)

#TrainLabel = to_categorical(TrainLabel, num_classes=num_classes)
#ValLabel = to_categorical(ValLabel, num_classes=num_classes)


label_encoder = LabelEncoder()
TrainLabel = label_encoder.fit_transform(TrainLabel)
ValLabel = label_encoder.fit_transform(ValLabel)

gc.collect()

TrainLabel = to_categorical(TrainLabel, num_classes=num_classes)
ValLabel = to_categorical(ValLabel, num_classes=num_classes)

model = MobileNet_model(num_classes)

gc.collect()

'''
# learning rate`in kademeli azaltilmasi
lr_decay_params = {
    "monitor": "val_loss",
    "factor": 0.5,
    "patience": 2,
    "min_lr": 1e-5
}
lr_decay = ReduceLROnPlateau(**lr_decay_params)
'''

" early stop and callbacks..."

es_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

checkpoint = ModelCheckpoint("MobileNet_model_Original.h5", monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='min')

    

" compile... "
model.compile(Adam(learning_rate=l_rate), loss='categorical_crossentropy',
        metrics=[categorical_accuracy])


gc_callback = GcCallback(step_interval=1000)  # Set the interval as needed


callbacks_list = [es_loss,checkpoint,gc_callback]

'''
dataset = tf.data.Dataset.from_tensor_slices((TrainData, TrainLabel))
dataset = dataset.batch(batch_size)
'''
history = model.fit(TrainData, TrainLabel, batch_size=batch_size,shuffle=True, epochs=epochs, 
                    verbose=1, validation_data=(ValData, ValLabel),
                    callbacks = [callbacks_list])

" plot history for loss " 
def plot_loss(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.grid(True) 
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)
    plt.savefig('Loss.png')

plot_loss(history, "Original MobileNet_model Loss Function")

os.chdir(DATABASEPATH)



def save_model(model, name):
 
        " serialize model to JSON "
        model_json = model.to_json() 
        fname = name +'.json'
        
        with open(fname, "w") as json_file: 
            json_file.write(model_json)
            
            " serialize weights to HDF5 "
            fname = name+'.h5'
            model.save_weights(fname) 
            print("Saved MobileNet_model_Original to disk")
            
save_model(model,'MobileNet_model_Original')
    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


ResultPATH=os.path.join(DATABASEPATH, "Results")


# Accuracy Graph
plt.figure(figsize=(12,6))
plt.plot(history.history['categorical_accuracy'], label='Train')
plt.plot(history.history['val_categorical_accuracy'], label='Validation')
plt.title('Original Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
accuracyPATH=os.path.join(ResultPATH, "Original MobileNet_model Accuracy.png")
plt.savefig(accuracyPATH)
plt.show()

# Draw loss function
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Original Model Loss Function')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
lossPATH=os.path.join(ResultPATH, "Original MobileNet_model Loss Function.png")
plt.savefig(lossPATH)
plt.show()




'''
loss_train = history.history["loss"]
acc_train = history.history["categorical_accuracy"]
loss_val = history.history["val_loss"]
acc_val = history.history["val_categorical_accuracy"]
epochs = np.arange(1, len(loss_train) + 1)



plot_loss(history)

plt.plot(epochs, loss_train, "b", label="Training loss")
plt.plot(epochs, loss_val, "b--", label="Validation loss")
plt.title("Original Losses")
plt.legend()
plt.show()

'''
'''
plt.plot(epochs, acc_train, "b", label="Training acc")
plt.plot(epochs, acc_val, "b--", label="Validation acc")
plt.title("Accuracy")
plt.legend()
plt.savefig('Accuracy.png')
plt.show()

'''