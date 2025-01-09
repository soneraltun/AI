# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:14:32 2024

@author: Pc
"""

from tensorflow.keras.applications import MobileNet   
from MobileNetModel import MobileNet_model,GetModel_BatchSize,GetModel_Epochs,GetModel_l_rate,GetModel_num_classes
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2


import numpy as np


from load_data import dataProcess
import os
import sys
import time
import SimpleITK as sitk

from sklearn.preprocessing import LabelEncoder

from scipy.ndimage import median_filter, gaussian_filter
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall,AUC
from tensorflow.keras.models import Model, model_from_json
from skimage import io, exposure, img_as_float, transform, morphology
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.losses import SparseCategoricalCrossentropy


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



" Evaluation "
name = 'MobileNet_model_Augmented'
fname = name +'.json'
json_file = open(fname, 'r')
loaded_model_json = json_file.read()
json_file.close()
        
sequence_model = model_from_json(loaded_model_json)
        
" load weights into new model..."
fname = name + '.h5'
sequence_model.load_weights(fname) 
print("Loaded model from disk")

filename = 'Test_Entry.csv'
DataEntryPATH=os.path.join(PATH, filename)

test_df=pd.read_csv(DataEntryPATH)

test_df = test_df[test_df['Finding Labels'] != 'Hernia']

test_df = test_df.reset_index(drop=True)

label_processor = tf.keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(test_df["Finding Labels"])
)
print(label_processor.get_vocabulary())

class_names=label_processor.get_vocabulary()

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
    
    #img = preprocess_input(img)
    img=img/255.0
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
        # VGG16 için ön işleme (normalize etme)
        img = preprocess_input(img)  # VGG16 için uygun şekilde normalize etme
        # Görüntü geçerli mi kontrol et
        if img is None:
            raise ValueError("Görüntü yüklenemedi. Lütfen geçerli bir dosya yolu kullanın.")
        
        # Görüntüyü 224x224 boyutuna yeniden boyutlandır
        img = cv2.resize(img, (224, 224))
        
        # Eğer gri tonlamalı bir görüntü ise, 3 kanallı hale getirme (gri -> RGB)
        if len(img.shape) == 2:  # Eğer görüntüde kanal yoksa (gri tonlamalı)
            img = np.expand_dims(img, axis=-1)  # (224, 224, 1)
            img = np.repeat(img, 3, axis=-1)  # (224, 224, 3) olarak tekrar et
        
        '''
        
            
        Data.append(img)
        Label.append(row['Finding Labels'])

    return Data,Label
    

TestData,TestLabel=GetImages(test_df)

TestData = np.array(TestData)

label_encoder = LabelEncoder()

TestLabel = label_encoder.fit_transform(TestLabel)

TestLabel_single=TestLabel

TestLabel = to_categorical(TestLabel)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)  # X eksenini dikleştir
    plt.gca().xaxis.set_ticks_position('top')
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], '.0f' if not normalize else '.2f'),  # Tam sayılar için '.0f' kullan
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

" early stop and callbacks..."





" compile... "
sequence_model.compile(Adam(learning_rate=l_rate), loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall(),AUC()])


y_pred = sequence_model.predict(TestData)

# evaluate the model
results=sequence_model.evaluate(TestData, TestLabel, verbose=0)

#print('Test: %.3f' % (test_acc))

predicted_classes = np.argmax(y_pred, axis=1)
TestLabel_single = np.argmax(TestLabel, axis=1)


# accuracy hesaplama
#test_loss, test_accuracy = sequence_model.evaluate(TestData, TestLabel)

#test_accuracy=np.around(test_accuracy,2)

# Test doğruluğunu yazdır
#print(f'Original Test Accuracy: {results[1]}')
# Precision ve Recall nesnelerini oluşturma
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

# Tahmin sonuçlarını güncelleme
precision.update_state(TestLabel_single, predicted_classes)
recall.update_state(TestLabel_single, predicted_classes)


test_accuracy=results[1]
precision = results[2]  # Precision değeri
recall = results[3]     # Recall değeri
auc = results[4]     # Recall değeri



def f1_score(precision,recall):
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

f1 = f1_score(precision, recall)

'''
# F1 skorunu hesapla
if precision_value + recall_value > 0:
    f1_score = 2 * (precision_value * recall_value) / (precision_value + recall_value)
else:
    f1_score = 0.0  # Kesinlik ve duyarlılık sıfırsa F1 skoru da sıfırdır.

f1_score=np.around(f1_score,2)
'''


# F1 skorunu hesaplamak için kesinlik ve duyarlılığı al
test_accuracy = round(test_accuracy,2)#
precision_value = round(precision,2)#
recall_value = round(recall,2)#
f1_score = round(f1,2)#
auc = round(auc,2)#
'''
precision_value = np.around(test_accuracy.numpy(),2)
recall_value = np.around(recall.result().numpy(),2)
f1_score=np.around(f1_score.result().numpy(),2)
'''
# Sonuçları yazdırma
print(f"Accuracy: {test_accuracy}")    # Accuracy
print(f"Precision: {precision_value:.2f}")
print(f"Recall: {recall_value:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"AUC: {auc:.2f}")
# Compute confusion matrix

cnf_matrix = confusion_matrix(TestLabel_single, predicted_classes)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Augmented:Confusion matrix, without normalization')


file_name = "test_results.csv"
DataEntryPATH = os.path.join(DATABASEPATH, "Results", file_name)

# Eğer dosya varsa yükle, yoksa boş bir DataFrame oluştur
if os.path.exists(DataEntryPATH):
    df = pd.read_csv(DataEntryPATH)
    # Eksik sütunları kontrol et ve ekle
    for col in ["test_name", "classes", "batch_size", "epochs", "optimizer", "learning_rate", "accuracy", "precision", "recall", "f1-score","auc"]:
        if col not in df.columns:
            df[col] = None
else:
    # Sütun adlarını oluştur
    columns = ["test_name", "classes", "batch_size", "epochs", "optimizer", "learning_rate", "accuracy", "precision", "recall", "f1-score","auc"]
    df = pd.DataFrame(columns=columns)

# Yeni veri
test_results = {
    "test_name": "MobileNet_Test Augmented",
    "classes": class_names,  # Liste yerine virgüllerle ayrılmış metin
    "batch_size": batch_size,
    "epochs": epochs,
    "optimizer": "adam",
    "learning_rate": float(l_rate),
    "accuracy": test_accuracy,
    "precision": precision_value,
    "recall": recall_value,
    "f1-score": f1_score,
    "auc":auc
}

# Yeni veriyi DataFrame'e ekle
df = pd.concat([df, pd.DataFrame([test_results])], ignore_index=True)

# Güncellenmiş DataFrame'i kaydet
df.to_csv(DataEntryPATH, index=False)






# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


