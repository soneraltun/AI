# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:45:32 2024

@author: cyborg
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_float, transform, morphology
from scipy.ndimage import median_filter, gaussian_filter
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, RandomRotation, RandomTranslation, RandomZoom, RandomFlip, RandomContrast
from tensorflow.keras.models import Model, model_from_json
import BarChart as bar
from itertools import islice


import shutil
import gc

import time

from tqdm import tqdm

start = time.time()

DATABASEPATH = os.path.dirname(os.path.abspath(__file__))
PATH=os.path.join(DATABASEPATH,"NIH Chest X-ray Dataset")
save_dir=os.path.join(PATH,"aug_img")
save_dir=os.path.join(save_dir,"images")

   



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def LoadDataEntry(path):
    return pd.read_csv(path)
 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def readXray(file_name_with_path,file_name):
   
   " check file type " 
   main_fn, ext_fn = os.path.splitext(file_name)
   
 
   if(ext_fn == '.dcm'): 
       itk_image = sitk.ReadImage(file_name_with_path)
       img = sitk.GetArrayFromImage(itk_image)
       img = np.squeeze(img)
   else: 
       img = img_as_float(io.imread(file_name_with_path))
           
   if(len(img.shape) == 3): 
       img = img[:,:,0]

    # Median filter
   img = median_filter(img, size=3)  # 3x3 median filter (you can adjust the filter size)

    # Sharpening (Gaussian filter)
   img = gaussian_filter(img, sigma=1)  # Adjust the sigma parameter for the desired sharpness

    
   return img

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def GetImages(df):
    
    Data, Label = [], []
    
    for index, row in df.iterrows():
        filename = row['Image Index']
        filefolder= row['Folder']
        #cropfolderName="CroppedImages"
        cropfolderName="images"
        
        ImagePath=os.path.join(PATH,filefolder)
        ImagePath=os.path.join(ImagePath,cropfolderName)
        ImagePath=os.path.join(ImagePath,filename)
     
        img=readXray(ImagePath,filename)

        
        # Görüntü geçerli mi kontrol et
        if img is None:
            raise ValueError("Görüntü yüklenemedi. Lütfen geçerli bir dosya yolu kullanın.")
        
        
        # Görüntüyü 224x224 boyutuna yeniden boyutlandır
        img_resized = cv2.resize(img, (224, 224))
        
        # Eğer gri tonlamalı bir görüntü ise, 3 kanallı hale getirme (gri -> RGB)
        if len(img_resized.shape) == 2:  # Eğer görüntüde kanal yoksa (gri tonlamalı)
            img_resized = np.expand_dims(img_resized, axis=-1)  # (224, 224, 1)
            img_resized = np.repeat(img_resized, 3, axis=-1)  # (224, 224, 3) olarak tekrar et
        
        
        # VGG16 için ön işleme (normalize etme)
        #img_preprocessed = preprocess_input(img_resized)  # VGG16 için uygun şekilde normalize etme
        
        Data.append(img_resized)
        Label.append(row['Finding Labels'])

    return Data,Label

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def augment_image(image, label):
    # Veri artırma pipeline
    data_augmentation = Sequential([                        # Yeniden ölçeklendirme
        RandomRotation(factor=(-5/360, 5/360),fill_mode="nearest"),             # Döndürme (-5° ile +5°)
        RandomTranslation(height_factor=0,width_factor=0.05,fill_mode="nearest"),  # Kaydırma
        RandomZoom(height_factor=0.1, width_factor=0.1,fill_mode="nearest"),  # Yakınlaştırma
    ])
    # Görüntü üzerinde veri artırma
    image = data_augmentation(image)
    return image, label
    '''
    # Görüntü artırma işlemleri
    rotation_layer = tf.keras.layers.RandomRotation(factor=(-5/360, 5/360), fill_mode="nearest")
    image = rotation_layer(image)
    image = tf.image.random_flip_left_right(image)        # Rastgele yatay çevirme
    #image = tf.image.random_flip_up_down(image)          # Rastgele dikey çevirme
    #image = tf.image.random_brightness(image, max_delta=0.2)  # Parlaklık değişikliği
    #image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Kontrast değişikliği
    #image = tf.image.random_zoom(image, (0.8, 1.2))      # Yakınlaştırma
    translation_layer = RandomTranslation(height_factor=0.1, width_factor=0.1)
    image = translation_layer(image)
    zoom_layer = tf.keras.layers.RandomZoom(
        height_factor=(-0.2, 0.2),  # -0.2 ile 0.2 arasında ölçekleme
        width_factor=(-0.2, 0.2),   # -0.2 ile 0.2 arasında ölçekleme
        fill_mode="nearest"         # Kenarları doldurma yöntemi
    )
    image = zoom_layer(image)
    '''
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Oversampling: F ve C sınıflarını artırmak için
def oversample_class(dataset, target_count, save_dir, class_label,df):

    images, labels = [], []
    for img, lbl in dataset.as_numpy_iterator():
        images.append(img)
        labels.append(lbl)
    images = np.array(images)
    labels = np.array(labels)
    
    label = label_encoder.inverse_transform([class_label])[0]

    count_diff = target_count - len(images)
    datagen = tf.data.Dataset.from_tensor_slices((images, labels))
    
    
    # Görüntü artırma ve kaydetme
    os.makedirs(save_dir, exist_ok=True)
    augmented_count = 0

    for img, lbl in datagen.repeat().map(augment_image).take(count_diff):
        # TensorFlow tensörünü NumPy formatına dönüştür
        img = img.numpy()

        # Görüntüyü kaydet
        img = (img * 255).astype(np.uint8)  # [0, 1] aralığını [0, 255]'e dönüştür
        img = Image.fromarray(img)
        file_name = f"{label}_augmented_{augmented_count}.png"
        filepath=f"{save_dir}\{file_name}"
        img.save(filepath)
        
        
        data = {
            "Image Index": file_name,
            "Finding Labels":label,
            "Folder": "aug_img",
        }
        df = df.append(data, ignore_index=True)
        augmented_count += 1
        gc.collect()
    
    dataset = None  # Dataset referansını serbest bırak
    gc.collect()  # Çöp toplama işlemini zorla
    
    augmented = datagen.repeat().map(augment_image).take(count_diff)


    balanced_dataset = datagen.concatenate(augmented)
       
    return balanced_dataset,df


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def cropImg(img, pr):
    " computes the non-zero borders, and draws a rectangle around the predicted lung area"
    " the area inside the rectangle can be sent to neural network to process"
    pr = pr.astype(np.float32) 
    vertical_sum = np.sum(pr, axis = 0)
    horizontal_sum = np.sum(pr, axis = 1)
    

    indexes = np.nonzero(vertical_sum)[0]
    border_l = indexes[0]
    border_r = indexes[len(indexes)-1]
    
    border_l-border_r
    
    indexes = np.nonzero(horizontal_sum)[0]
    border_up = indexes[0]
    border_down = indexes[len(indexes)-1]
    
    width=border_r-border_l
    height=border_down-border_up
    
    maximum = max(height, width)
    
    a=(int)((maximum-width)/2)
    b=(int)((maximum-height)/2)
    
    
    border_l=border_l-a
    border_r=border_l+maximum
    
    border_up=border_up-b
    border_down=border_up+maximum
    
    if(border_l<0):
        border_l=0
    
    if(border_up<0):
        border_up=0
        
    # Kırpma parametreleri
    offset_height = border_up  # Yükseklik başlangıç noktası
    offset_width = border_l   # Genişlik başlangıç noktası
    target_height = maximum  # Kırpılacak yüksekliğin uzunluğu
    target_width = maximum   # Kırpılacak genişliğin uzunluğu
    
    image_height, image_width = img.shape[:2]
    
    if offset_height + target_height > image_height:
        target_height = image_height - offset_height

    if offset_width + target_width > image_width:
        target_width = image_width - offset_width
    
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)    
           
    # Kırpma işlemi
    crop = tf.image.crop_to_bounding_box(
        tf.cast(img, tf.float32), offset_height, offset_width, target_height, target_width
    )
    
    crop = tf.squeeze(crop, axis=-1)
    '''
    print(f"Orijinal Görüntü Boyutu: {img.shape}")
    print(f"Kırpılmış Görüntü Boyutu: {crop.shape}")
    '''
    #img = cv2.rectangle(img, start_point, end_point, color, thickness) 
    
    #crop = img[ border_up:border_down, border_l:border_r]
   
    return crop

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def segmentLung(seg_model, img,file_name):
 
     "input shape..." 
     seg_w = seg_model.input.shape[1]         
     seg_h = seg_model.input.shape[2]    
     im_shape = (seg_w,seg_h)  
     
    
     "pre-process"
     img = transform.resize(img, im_shape)
     orig_img = img 
     img = exposure.equalize_hist(img)
     
     " normalize "
     img -= img.mean()
     img /= img.std()
     
     " Predict the lung region "
     img = np.expand_dims(img, -1)
     img = np.expand_dims(img,axis = 0)
    
     pred = seg_model.predict(img)[..., 0].reshape(im_shape)
     img = np.squeeze(img)
     
     " 0-1 mask conversion "
     pr = pred > 0.5
     pr = morphology.remove_small_objects(pr, int(0.1*im_shape[1])) 
     
     " show predicted results "  
     #show_frames(file_name, orig_img, pred, pr)
     
     return pr
 
def loadSegmentModel():
 
    " load lung segment model and weights... (model U-net) " 
    json_file = open('segment_model.json', 'r') 
    loaded_model_json = json_file.read() 
    json_file.close() 
    model = model_from_json(loaded_model_json)
        
    " load weights into the model " 
    model.load_weights('segment_model.hdf5') 
    print("Loaded model from disk")
    
    
    '''
    DIR = os.path.dirname(__file__)
    
    WIDTH = 512
    HEIGHT = 512
    
    model = UNet()
    data_loader = dataProcess(WIDTH, HEIGHT)
    
    # Train
    # images_train, labels_train = data_loader.loadTrainData()
    # model.fit(images_train, labels_train)
    # model.save(MODEL_PATH)
    
    # Predict
    model.load()
    images_test, images_info = data_loader.loadTestData()
    start = time.process_time()
    model.predict(images_test, images_info)
    print('Time eslapsed: {}'.format(time.process_time() - start))
    
    '''
    return model
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  
def write_dataframe(outdf,output_file):
  
  output_file_path=os.path.join(DATABASEPATH, output_file)
  # Dosyanın varlığını kontrol et
  if os.path.exists(output_file_path):
      os.remove(output_file_path,)

  # Dosya mevcut değilse yeni bir dosya olarak yaz
  outdf.to_csv(output_file_path, index=False)
  print(f"DataFrame başarıyla '{output_file_path}' dosyasına yazıldı.")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def ReduceLabel(df,labelColumn):
    # Yeni bir sütun oluşturmak için bir döngü kullan
    for i, row in df.iterrows():
        # Virgül var mı kontrol et
        if '|' in row[labelColumn]:

            # Virgül varsa parçala ve ilk değeri mevcut sütuna ata
            parcalanmis_degerler = row[labelColumn].split('|')
            df.at[i, labelColumn] = parcalanmis_degerler[0]

    inner_dict = {"değer": "", "renk": ""}


    elemanlar = {}

    # Gruplama işlemi
    gruplanmis_df = df.groupby(labelColumn)

    # Gruplanmış verileri görüntüleme
    for group, data in gruplanmis_df:
        inner_dict = {"değer": "", "renk": ""}
        elemanlar.update({group: inner_dict})
        elemanlar[group]['değer'] = data.shape[0]

    return df, elemanlar

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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


num_classes=3


filename = 'Data_Entry.csv'



DataEntryPATH=os.path.join(DATABASEPATH, filename)

df=pd.read_csv(DataEntryPATH)

df = df[df['Finding Labels'].isin(['Fibrosis', 'Nodule'])]

df = df.reset_index(drop=True)

'''
# En düşük 3 değeri bul
top_counts = df['Finding Labels'].value_counts().nsmallest(num_classes)
bottom_labels = top_counts.index

# Yeni DataFrame oluştur
df = df[df['Finding Labels'].isin(bottom_labels)]
'''
os.chdir(DATABASEPATH)



X, y = [], []
items=os.listdir(PATH)




labels = df['Finding Labels']  # Hedef sütun

train_val_df,test_df,train_val_df_y,test_df_y = train_test_split(df, labels, test_size=0.2, stratify=labels, random_state=42,)

train_df, val_df,train_df_y,val_df_y = train_test_split(train_val_df,train_val_df_y, test_size=0.25, stratify=train_val_df_y,random_state=42)

del labels

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

group_counts = train_df['Finding Labels'].value_counts()
print(group_counts)

train_df = train_df.reset_index(drop=True)

grouped_data = train_df.groupby("Finding Labels")

# Her grubu ayrı bir DataFrame olarak saklama
grouped_dfs = {label: group for label, group in grouped_data}

datasets=[]

num_classes=1

# Ortak LabelEncoder oluştur
label_encoder = LabelEncoder()

# Tüm gruplardan etiketleri toplama
all_labels = []
'''
for label, group in grouped_dfs.items():   
    #, TrainLabel = GetImages(group)
    all_labels.extend(TrainLabel)
'''

TrainLabel=train_df["Finding Labels"]
all_labels.extend(TrainLabel)
# Ortak tüm etiketler üzerinde fit işlemi
label_encoder.fit(all_labels)

#sorted_groups = sorted(grouped_dfs.items(), key=lambda x: len(x[1]), reverse=True)
sorted_grouped_dfs = dict(
    sorted(
        grouped_dfs.items(),
        key=lambda x: x[1].shape[0],  # Satır sayısına göre sıralama
         reverse=True  # Büyükten küçüğe sıralama
    )
)
for label, df in sorted_grouped_dfs.items():
    if isinstance(df, pd.DataFrame):
        print(f"Label: {label}, Boyut: {len(df)}")
    else:
        print(f"Label {label} bir DataFrame değil!")

max_length=max(df.shape[0] for df in sorted_grouped_dfs.values())

# Her grup için dataset oluştur ve birleştir
for label, group in islice(sorted_grouped_dfs.items(), 1, None):
    
    TrainData, TrainLabel = GetImages(group)
    TrainData = np.array(TrainData)

    # Ortak LabelEncoder kullanarak etiketleri kodla
    TrainLabel = label_encoder.transform(TrainLabel)
    class_label=unique_labels = np.unique(TrainLabel)

    
    new_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(TrainData, tf.float32),
        tf.convert_to_tensor(TrainLabel, dtype=tf.float64)
    ))
    
    #class_label = int(dataset.map(lambda _, lbl: lbl).take(1).as_numpy_iterator().__next__())

    print(f"Dataset {label} için oversampling yapılıyor...")
    
    
    balanced_dataset,train_df= oversample_class(new_dataset, max_length,save_dir,class_label,train_df)
        
    new_dataset = None  # Dataset referansını serbest bırak
    gc.collect()  # Çöp toplama işlemini zorla
    
     # Dengelenmiş dataset'in uzunluğunu hesapla
    balanced_length = sum(1 for _ in balanced_dataset)
    
    print(f"Dengelenmiş Dataset {class_label} uzunluğu: {balanced_length}")
    
    #datasets.append(new_dataset)
    gc.collect()

'''
dataset_lengths = [sum(1 for _ in dataset) for dataset in datasets]

# En büyük uzunluğu ve ilgili dataset'i bul
max_length = max(dataset_lengths)



print(f"En büyük dataset uzunluğu: {max_length}")

'''
os.chdir(DATABASEPATH)

write_dataframe(train_df,'Train_Entry.csv')
write_dataframe(test_df,'Test_Entry.csv')
write_dataframe(val_df,'Val_Entry.csv')

seg_model = loadSegmentModel()


"input shape..." 
seg_w = seg_model.input.shape[1]         
seg_h = seg_model.input.shape[2]    
im_shape = (seg_w,seg_h)


df = pd.concat([train_df,val_df, test_df,], ignore_index=True)

'''
unique_folders_df = df.drop_duplicates(subset='Folder')

for index, row in unique_folders_df.iterrows():
    current_path=os.path.join(PATH, row['Folder'])

    # Specify the name of the new folder
    new_folder_name = "CroppedImages"

    # Combine the current directory path with the new folder name
    newImageFolderPath = os.path.join(current_path, new_folder_name)
    # Check if the folder doesn't exist already
    if os.path.exists(newImageFolderPath):
        shutil.rmtree(newImageFolderPath)
       
    os.makedirs(newImageFolderPath)
    
'''


for index, row in df.iterrows():
    
    ImageFolderPath=os.path.join(PATH, row['Folder'])
    current_path=ImageFolderPath
       
    ImageFolderPath=os.path.join(ImageFolderPath, "images")     

        
    # Specify the name of the new folder
    new_folder_name = "CroppedImages"

    # Combine the current directory path with the new folder name
    newImageFolderPath = os.path.join(current_path, new_folder_name)
    
    '''
    # Check if the folder doesn't exist already
    if os.path.exists(newImageFolderPath):
        shutil.rmtree(newImageFolderPath)
       
    os.makedirs(newImageFolderPath)
    '''
    fig = plt.figure(figsize=(30, 10))
    files=os.listdir(ImageFolderPath)
    imageFileName= row['Image Index']
    
    imagePath=os.path.join(ImageFolderPath, imageFileName) 
    
    img = readXray(imagePath,imageFileName)

    "pre-process"
    img = transform.resize(img, im_shape)
    orig_img = img 
    img = exposure.equalize_hist(img)
        
    " normalize "
    img -= img.mean()
    img /= img.std()
        
    " Predict the lung region "
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img,axis = 0)
    
    pred = seg_model.predict(img)[..., 0].reshape(im_shape)
    img = np.squeeze(img)
        
    " 0-1 mask conversion "
    pr = pred > 0.5
    pr = morphology.remove_small_objects(pr, int(0.1*im_shape[1]))    
  
    if np.all(pr == False):
        train_df = train_df[train_df["Image Index"] != imageFileName]
        
        df = df[df["Image Index"] != imageFileName]
        continue
    
    " show predicted results "  
    #show_frames(imageFileName, orig_img, pred, pr)
    
    crop=cropImg(orig_img,pr)
    
 
    os.chdir(newImageFolderPath)
    
    # Opsiyonel olarak, SimpleITK görüntüsünü kaydetme
    #sitk.WriteImage(image, imageFileName)
    plt.imsave(imageFileName, crop, cmap='gray')

    if index % 500 == 0:
        gc.collect()





frame, elemanlar = ReduceLabel(train_df,"Finding Labels")

elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}

df2 = bar.DrawBarChart(elemanlar_abbv)


frame, elemanlar = ReduceLabel(val_df,"Finding Labels")

elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}

df2 = bar.DrawBarChart(elemanlar_abbv)


frame, elemanlar = ReduceLabel(test_df,"Finding Labels")

elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}

df2 = bar.DrawBarChart(elemanlar_abbv)



def draw_rectangle(img, pr):
    " computes the non-zero borders, and draws a rectangle around the predicted lung area"
    " the area inside the rectangle can be sent to neural network to process"
    pr = pr.astype(np.float32) 
    vertical_sum = np.sum(pr, axis = 0)
    horizontal_sum = np.sum(pr, axis = 1)
    

    indexes = np.nonzero(vertical_sum)[0]
    border_l = indexes[0]
    border_r = indexes[len(indexes)-1]
    
    indexes = np.nonzero(horizontal_sum)[0]
    border_up = indexes[0]
    border_down = indexes[len(indexes)-1]
    
    start_point = (border_l, border_up)
    end_point = (border_r, border_down)
    color = (1, 1, 0) 
    thickness = 2
    img = cv2.rectangle(img, start_point, end_point, color, thickness) 
    
    return img

def show_frames(filename, img, pred, pr):
    
    " show predicted results "
    plt.subplot(131)
    plt.title('input', fontsize = 40)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray)
        
    plt.subplot(132)
    plt.title('Predicted Lung', fontsize = 40)
    plt.axis('off')
    plt.imshow(pred,cmap='jet')  
        
    " draw rectangle... "
    img = draw_rectangle(pr, pr)
    plt.subplot(133)
    plt.title('ProcessArea', fontsize = 40)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    filename = filename +'.png'
    plt.savefig(filename)
    
    plt.show()

