# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:30:33 2024

@author: Pc
 """
    
import itertools
import numpy as np
import tensorflow as tf
import pydicom
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import shutil
import gc
import SimpleITK as sitk
import PrepareData as ip
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
from skimage import io, exposure, img_as_float, transform, morphology
from tqdm import tqdm



with tf.device('/GPU:0'):
    start = time.time()
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
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    def readXray(file_name_with_path,file_name):
       fig = plt.figure(figsize=(30, 10))
       print(file_name)
       
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
    def LoadDataEntry(path):
        return pd.read_csv(path)
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ## Define hyperparameters
    """
    
    IMG_SIZE = 512
    BATCH_SIZE = 64
    EPOCHS = 300
    
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048
    
    ## Data preparation
    """
    
    
    
    num_classes=15
    DATABASEPATH = os.path.dirname(os.path.abspath(__file__))
    PATH=os.path.join(DATABASEPATH,"NIH Chest X-ray Dataset")
    
    
    filename = 'Data_Entry_2017.csv'
    DataEntryPATH=os.path.join(PATH, filename)
    
    df=LoadDataEntry(DataEntryPATH)
    
    df = df[["Image Index", "Finding Labels"]]
    
    df['Folder'] = None

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
            
        
        start_point = (border_l, border_up)
        end_point = (border_r, border_down)
        color = (1, 1, 0) 
        thickness = 2
        
        #img = cv2.rectangle(img, start_point, end_point, color, thickness) 
        
        crop = img[ border_up:border_down, border_l:border_r]
       
        return crop
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    def GetFilesNames(path):
        FileNames=LoadDataEntry(path)
        return FileNames
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def GetPathAllFiles():
        PATH = os.getcwd()
        #PATH=BASEPATH+"/NIH Chest X-ray Dataset"
        
        folders=os.listdir(PATH)
        #folders = [item for item in items if os.path.isdir(os.path.join(PATH, item))]
        file_dict={}
        for folder in (folders):
            ImageFolderPath=os.path.join(PATH, folder)
            #ImageFolderPath=os.path.join(ImageFolderPath, "newImages")
            
            if os.path.exists(ImageFolderPath):
                #print("File exists!")
                pass
            else:
                try:
                   os.makedirs(ImageFolderPath)
                   print(f"Folder '{ImageFolderPath}' created successfully!")
                except OSError as e:
                   print(f"Failed to create folder '{ImageFolderPath}': {e}")
                   
            file_list=os.listdir(PATH)
            for file_name in file_list:            
                file_path = os.path.join(PATH,file_name)
                file_dict[file_name] = file_path
    
        return file_dict
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
    def CreateAugmentedImage(train_dataframe,maxValue):
        train_dataframe = train_dataframe.reset_index(drop=True)
        augmented_df=pd.DataFrame(columns=train_dataframe.columns)
       
        aug_dir="aug_img"
        dest_dir = os.path.join(PATH, aug_dir)
        dest_dir = os.path.join(dest_dir, "images")
        dest_dir2=dest_dir
        
        os.makedirs(dest_dir, exist_ok=True)
               
        # Augument the data
        data_gen_param = {
            "rotation_range": 5,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.2,
            "rescale": 1.,
            "fill_mode":"nearest",
            "cval":0
        }
       
        data_generator = ImageDataGenerator(**data_gen_param)
        num_images_each_label = maxValue
    
        label=train_dataframe['Finding Labels'][0]
        
        add_size=maxValue-train_dataframe.shape[0]
        batch_size = 1

        
        for i in range(0,add_size):
            dest_dir=dest_dir2
            temp_img_dic=os.path.join(PATH,"Temp")
            os.makedirs(temp_img_dic,exist_ok=True)
            #temp_img_dic = temp_img_dic.replace("\\", "/")
            
            
            selected_row = train_dataframe.sample(n=1)
            src_dir= os.path.join(PATH,selected_row['Folder'].iloc[0])
            src_dir= os.path.join(src_dir,"images")
            
            class_name = label
            class_dir = os.path.join(temp_img_dic, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            '''
            dest_dir = os.path.join(dest_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            '''
            
            
            shutil.copy(os.path.join(src_dir,selected_row['Image Index'].iloc[0]), class_dir)
    
            newDircName=str(i)+"_aug_"+selected_row['Image Index'].iloc[0]
            
            '''
            files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
            for image_name in files:
                shutil.copy(os.path.join(src_dir, image_name), os.path.join(dest_dir, image_name)) '''
    
            
            data_flow_param = {
                "directory": temp_img_dic,
                "batch_size": batch_size,
                "shuffle": True,
                "save_to_dir": dest_dir,
                "save_format": "png",
            }
            aug_data_gen = data_generator.flow_from_directory(**data_flow_param)
    
            for i in range(batch_size):
                next(aug_data_gen)
            
            # Dizindeki dosyaları al ve sıralayarak en son dosyayı seç
            generated_image_filenames = sorted(os.listdir(dest_dir), key=lambda x: os.path.getmtime(os.path.join(dest_dir, x)))
            
            # En son kaydedilen dosya
            filename = generated_image_filenames[-1]  # En son dosya
            
            for filename in generated_image_filenames:
                data = {
                'Image Index': [filename],
                'Finding Labels': [label],
                'Folder': "aug_img"
                }
                new_row=pd.DataFrame(data)
            augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)
            
            if os.path.exists(temp_img_dic):
                shutil.rmtree(temp_img_dic)
               
        return augmented_df
    
    def ApplyDataAugmentation(train_df):
        
        #max_value_of_patient_image = GetMaxValueFromPatientGroup(train_df)
        
        labels = train_df.groupby('Finding Labels')
        sizes = train_df.groupby('Finding Labels').size()
    
        max_value_of_patient_image=sizes.max()
        '''
        classSize=labels.size()
        max_value_of_patient_image=int(len(train_df)/len(classSize))
        '''
        for _, group in labels:
          augmented_df=CreateAugmentedImage(group,max_value_of_patient_image)
          train_df = pd.concat([train_df, augmented_df], ignore_index=True)
          gc.collect()
        return train_df
    
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    frame, elemanlar = ReduceLabel(df,"Finding Labels")

    elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}

    #df2 = bar.DrawBarChart(elemanlar_abbv)
    
    
    # En düşük 3 değeri bul
    top_counts = df['Finding Labels'].value_counts().nsmallest(num_classes)
    bottom_labels = top_counts.index
    
    # Yeni DataFrame oluştur
    df = df[df['Finding Labels'].isin(bottom_labels)]
    '''
    
    # En yüksek 4 değeri bul
    top_counts = df['Finding Labels'].value_counts().nlargest(num_classes)
    
    # 2., 3. ve 4. en yüksek değerleri al
    top_labels_2_to_4 = top_counts.index[1:]  # 2., 3. ve 4. en yüksek olanları seç
    
    # Yeni DataFrame oluştur
    df = df[df['Finding Labels'].isin(top_labels_2_to_4)]
    '''
    
    df = df.reset_index(drop=True)
    
    os.chdir(DATABASEPATH)
    
    seg_model = ip.loadSegmentModel()
    
    # 10000 görüntüyü işleme alın
    total_images = 10000
    
    
    " read train data...both department..."
    X, y = [], []
    items=os.listdir(PATH)
    
    folders = [item for item in items if os.path.isdir(os.path.join(PATH, item))]
    
    
    for folder in (folders):
    
        ImageFolderPath=os.path.join(PATH, folder)
        ImageFolderPath=os.path.join(ImageFolderPath, "images") 
        
        file_list=os.listdir(ImageFolderPath)
        
        for file_name in file_list:            
            df.loc[df['Image Index'] == file_name, 'Folder'] = folder
    
    #write_dataframe(df,'Data_Entry.csv')
    
    #foldername = 'Data_Entry.csv'
    
    #DataEntryPATH=os.path.join(DATABASEPATH, foldername)
    
    #foldernameframe=LoadDataEntry(DataEntryPATH)
    
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    group_counts = train_df['Finding Labels'].value_counts()
    print(group_counts)
    
    train_df=ApplyDataAugmentation(train_df)
    
    group_counts = train_df['Finding Labels'].value_counts()
    print(group_counts)
    
    "input shape..." 
    seg_w = seg_model.input.shape[1]         
    seg_h = seg_model.input.shape[2]    
    im_shape = (seg_w,seg_h) 
    
    '''
    # İlk DataFrame'in her satırındaki Image Index'e göre ikinci DataFrame'den Folder değerini al
    for index, row in df.iterrows():
        image_index = row['Image Index']
        
        # İkinci DataFrame'de Image Index ile eşleşen satırı bul
        matching_row = foldernameframe[foldernameframe['Image Index'] == image_index]
        
        if not matching_row.empty:
            # Eşleşen satır bulunduysa, Folder değerini birinci DataFrame'e yaz
            df.at[index, 'Folder'] = matching_row['Folder'].values[0]
    
    write_dataframe('DataframeWillClassificate.csv')
    '''
    #df=train_df[train_df['Folder']=='aug_img']
    df = pd.concat([train_df,val_df, test_df,], ignore_index=True)
    
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
        
        img = ip.readXray(imagePath,imageFileName)
    
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
            val_df = val_df[val_df["Image Index"] != imageFileName]
            test_df = test_df[test_df["Image Index"] != imageFileName]
            
            df = df[df["Image Index"] != imageFileName]
            continue
        
        " show predicted results "  
        #show_frames(imageFileName, orig_img, pred, pr)
        
        crop=cropImg(orig_img,pr)
        
        "pre-process"
        #crop = transform.resize(crop, im_shape)
        
        #plt.imshow(crop, cmap='gray')
        #plt.show()
        
        # Yeni dizine geçiş yapma
        #os.chdir(ImageFolderPath)
        
        '''
        # NumPy dizisini SimpleITK görüntüsüne dönüştürme
        sitk_image = sitk.GetImageFromArray(crop)
    
        # Görüntünün veri türünü unsigned char olarak ayarla
        np_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
       
        # Değerleri 0-255 arasında ölçeklendir
        np_image = ((np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image)) * 255).astype(np.uint8)
       
        '''
        os.chdir(newImageFolderPath)
        
        # Opsiyonel olarak, SimpleITK görüntüsünü kaydetme
        #sitk.WriteImage(image, imageFileName)
        plt.imsave(imageFileName, crop, cmap='gray')
    
        if i % 500 == 0:
            gc.collect()
    
    
    write_dataframe(train_df,'Train_Entry.csv')
    write_dataframe(test_df,'Test_Entry.csv')
    write_dataframe(val_df,'Val_Entry.csv')
    
    frame, elemanlar = ReduceLabel(train_df,"Finding Labels")
    
    elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}
    
    df2 = bar.DrawBarChart(elemanlar_abbv)
    
    
    frame, elemanlar = ReduceLabel(val_df,"Finding Labels")
    
    elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}
    
    df2 = bar.DrawBarChart(elemanlar_abbv)
    
    
    frame, elemanlar = ReduceLabel(test_df,"Finding Labels")
    
    elemanlar_abbv = {short_names[key]: value for key, value in elemanlar.items()}
    
    df2 = bar.DrawBarChart(elemanlar_abbv)
    
    def a():
        for folder in (folders):
    
            ImageFolderPath=os.path.join(PATH, folder)
            current_path=os.path.join(PATH, folder)
            ImageFolderPath=os.path.join(ImageFolderPath, "images")
    
            # Specify the name of the new folder
            new_folder_name = "CroppedImages"
    
            # Combine the current directory path with the new folder name
            newImageFolderPath = os.path.join(current_path, new_folder_name)
            
            # Check if the folder doesn't exist already
            if os.path.exists(newImageFolderPath):
                shutil.rmtree(newImageFolderPath)
                
            os.makedirs(newImageFolderPath)
            
            for i in tqdm(range(len(os.listdir(ImageFolderPath)))):
    
                #os.chdir(ImageFolderPath)
                fig = plt.figure(figsize=(30, 10))
                files=os.listdir(ImageFolderPath)
                imageFileName=files[i]
                
                imagePath=os.path.join(ImageFolderPath, imageFileName)
                
                
                
                img = ip.readXray(imagePath,imageFileName)
            
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
                #show_frames(imageFileName, orig_img, pred, pr)
                
                crop=cropImg(orig_img,pr)
                
                "pre-process"
                crop = transform.resize(crop, im_shape)
                
                #plt.imshow(crop, cmap='gray')
                #plt.show()
                
                # Yeni dizine geçiş yapma
                os.chdir(ImageFolderPath)
                
                '''
                # NumPy dizisini SimpleITK görüntüsüne dönüştürme
                sitk_image = sitk.GetImageFromArray(crop)
            
                # Görüntünün veri türünü unsigned char olarak ayarla
                np_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
               
                # Değerleri 0-255 arasında ölçeklendir
                np_image = ((np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image)) * 255).astype(np.uint8)
               
                '''
                os.chdir(newImageFolderPath)
                
                # Opsiyonel olarak, SimpleITK görüntüsünü kaydetme
                #sitk.WriteImage(image, imageFileName)
                plt.imsave(imageFileName, crop, cmap='gray')
                '''X.append(img)
                y.append(0)
                time.sleep(0.1'''
            
    
    
    
    # NIH Chest X-ray Dataset deki tüm görüntülerin pathini bul
    #filename_path_dictionary=GetPathAllFiles()
    
    
    
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
    
    print("GPU time:", time.time() - start)
