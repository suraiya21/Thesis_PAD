import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import math
import os 
from glob import glob
import asyncio

import imgaug.augmenters as iaa


 


#face_scale = 0.9  #default for test, for training , can be set from [0.8 to 1.0]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Sometimes(0.3,iaa.Add(value=(-40,40), per_channel=True)), # Add color 
    iaa.Sometimes(0.3,iaa.GammaContrast(gamma=(0.5,1.5))), # GammaContrast with a gamma of 0.5 to 1.5
    iaa.Sometimes(0.3,iaa.AverageBlur(k=((5, 7), (1, 3)))),
    #iaa.Sometimes(0.3,iaa.Affine(rotate=35))
])

def crop_face_from_scene(image, scale):
    y1,x1,w,h = 0, 0, image.shape[1], image.shape[0]
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    w_scale=scale/1.5*w
    h_scale=scale/1.5*h
    h_img, w_img = image.shape[0], image.shape[1]
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)
    region=image[x1:x2,y1:y2]
    return region


class Spoofing_train(Dataset):
    
    def __init__(self, info_list, root_dir, transform=None, scale_up=1.5, scale_down=1.0, img_size=256, map_size=32, UUID=-1):
        #print(info_list)
        self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        self.root_dir = root_dir
        #self.map_root_dir = root_dir.replace("Train_files", "Depth/Train_files")
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID
        #self.flag1 = 0
        #self.flag2 = 0

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        video_name = str(self.landmarks_frame.iloc[idx, 0])
        #print(video_name)
        #face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
        #face_scale = face_scale/10.0
        #image_x = cv2.imread(video_name)
        #image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        image_x, map_x1 = self.get_single_image_x_RGB(video_name)
        
        #image_x_ir = self.get_single_image_x(image_path_ir)
        #image_x = cv2.resize(crop_face_from_scene(image_x, face_scale), (self.img_size, self.img_size))
        # Checking if the image is empty or not
        if image_x is None:
            print(video_name)
            
        #image_x = cv2.resize(image_x, (self.img_size, self.img_size))
        #image_dir = os.path.join(self.root_dir, video_name)
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0             # fake'''

        #depth_x = self.get_image_map(video_name, spoofing_label)
        

        #image_x, map_x = self.get_single_image_x(image_dir, video_name, spoofing_label)
        '''if spoofing_label == 1:
            #print(video_name)
            #if (video_name.find('/1_3_mobai_collected/Train/1_bonafide_crop_rgb/') != -1 or video_name.find('/2_4_CASIA_Anti/Tain/1_bonafide_crop_rgb/') != -1 or video_name.find('/3_1_ntnu_evaluation/Train/1_bonafide_crop_rgb/') != -1 or video_name.find('/4_6_oulu/Train/1_bonafide_crop_rgb/') != -1 or video_name.find('/5_5_NBL/Train/1_bonafide_crop_rgb/') != -1):
            video_name_temp = video_name
            
            #map_tail = os.path.split(map_name)
            #map_tail = list(map_tail)
            
            
            if('TrainingPro_Bona' in video_name_temp):
                #print(video_name)
                
                #map_tail[1] = map_tail[1].replace(".bmp",'')
                #map_tail[1] = map_tail[1].replace(".png",'')
                #map_name = "{}/{}_depth.jpg".format(map_tail[0], map_tail[1])
                map_name = video_name_temp.replace("RGB", "Depth")
                #map_x = self.get_single_image_x(map_name)
                #map_name = map_name.replace(".bmp", "_depth.jpg")
                #print(map_name)
            if('SynthASpoof' in video_name_temp):
                
                map_name = video_name_temp.replace("Mobai_Data_RGB", "Mobai_Data_Depth")
                #print("xxxxxxxxxxxxx")

                #print(map_name)
                #image_x_temp = cv2.imread(map_name)
            #print('###')
            
            if('Datatang_2' in video_name_temp):
                map_name = video_name_temp.replace("3D_RGB", "3D_Depth")
                map_name = map_name.replace("_color", "_depth")

            if('Datatang_1' in video_name_temp):
                
                map_name = video_name_temp.replace("Combine_Datatang_RGB", "Combine_Datatang_Depth")
                map_name = map_name.replace("_normal", "_depth")
                
                
            if('Datatang_3' in video_name_temp):
                
                map_name = video_name_temp.replace("Frames_live_crop_v2", "Frames_live_crop_v2_depth")
                #map_name = map_name.replace("_normal", "_depth")
                
            if('Mobai_1' in video_name_temp):
                
                map_name = video_name_temp.replace("Frames_live_crop_combine", "Frames_live_crop_combine_Depth")
                
                
            #if('7_WMCA' in video_name_temp):
                
            #    map_name = video_name_temp.replace("7_WMCA_rgb_224", "7_WMCA_depth_224")
                #map_name = map_name.replace("_normal", "_depth")
            
            if('4_6_oulu' in video_name_temp or "5_5_NBL" in video_name_temp or "3_1_ntnu_evaluation" in video_name_temp):
                
                map_name = video_name_temp.replace("1_bonafide_crop_rgb", "2_bonafide_crop_depth")
                map_name = map_name.replace(".bmp", "_depth.jpg")
                #map_name = map_name.replace("_normal", "_depth")
            
            
            
                
                
            if('Mobai_2' in video_name_temp):
                
                map_name = video_name_temp.replace("Mobai_2_rgb", "Mobai_2_depth")
                #map_name = map_name.replace("_normal", "_depth")
            #cv2.imwrite('temp.jpg', image_x_temp)
            
            
            
            if('Mobai_4' in video_name_temp):
                
                map_name = video_name_temp.replace("Mobai_4_rgb", "Mobai_4_depth")
            
            
            
            
            
            
            
            
            
                
            #print(map_name)
            #face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
            #face_scale = face_scale/10.0
            #print(map_name)
            
            map_x = self.get_single_image_x(map_name)
            #map_x = self.get_single_image_x_zero()
            #map_x = cv2.resize(crop_face_from_scene(map_x, face_scale), (self.map_size, self.map_size))
            
            #map_x = cv2.resize(map_x, (self.map_size, self.map_size), interpolation = cv2.INTER_AREA)'''
            #print(map_x.shape)
                
           
            
        #else:
        map_x = self.get_single_image_x_zero()
                
                
            
            
            
        #print(map_x.max())
        #map_x = np.zeros((self.map_size, self.map_size))
        #print(image_x)
        #print(map_x.shape)
        sample = {"image_x": image_x, "spoofing_label": spoofing_label, "image_x_depth": map_x, "UUID": self.UUID}
        #self.flag1=self.flag2=0
        if self.transform:
            sample = self.transform(sample)
        return sample
        
        
        
    '''def get_image_map(self, video_name, spoofing_label):
        #print(spoofing_label)
        if spoofing_label == 1:
            #print(video_name)
            #if (video_name.find('/1_3_mobai_collected/Train/1_bonafide_crop_rgb/') != -1 or video_name.find('/2_4_CASIA_Anti/Tain/1_bonafide_crop_rgb/') != -1 or video_name.find('/3_1_ntnu_evaluation/Train/1_bonafide_crop_rgb/') != -1 or video_name.find('/4_6_oulu/Train/1_bonafide_crop_rgb/') != -1 or video_name.find('/5_5_NBL/Train/1_bonafide_crop_rgb/') != -1):
            video_name_temp = video_name
            map_name = video_name_temp.replace("1_bonafide_crop_rgb", "2_bonafide_crop_depth")
            map_tail = os.path.split(map_name)
            map_tail = list(map_tail)
            map_tail[1] = map_tail[1].replace(".bmp",'')
            
            map_name = "{}/{}_depth.jpg".format(map_tail[0], map_tail[1])
            #print(map_name)
            #face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
            #face_scale = face_scale/10.0
            print(map_name)
            map_x = cv2.imread(map_name,0)
            #map_x = cv2.resize(crop_face_from_scene(map_x, face_scale), (self.map_size, self.map_size))
            
            #map_x = cv2.resize(map_x, (self.map_size, self.map_size))
            #print(map_x.shape)
                
           
        else:
            map_x = np.zeros((self.map_size, self.map_size))
        return map_x'''
    def get_single_image_x(self, image_path):
        
        #image_x = np.zeros((224, 224, 3))
        #print(image_path)

        # RGB
        
        image_x_temp = cv2.imread(image_path)
            

  
        image_x = cv2.resize(image_x_temp, (224, 224))
        #print(image_x)
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        #image_x_aug = seq.augment_image(image_x.astype(np.uint8)) 
 
        
        return image_x
    
    
    
    def get_single_image_x_zero(self):
        
        
        #random = np.random.randint(200, 240)
        Zero=0
        image_x = np.full((self.map_size, self.map_size, 3), Zero).astype(np.uint8)

        # RGB
        #image_x_temp = cv2.imread(image_path)
        
        #cv2.imwrite('temp.jpg', image_x_temp)
  
        #image_x = cv2.resize(image_x_temp, (224, 224))
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        #image_x_aug = seq.augment_image(image_x) 
 
        
        return image_x


    def get_single_image_x_RGB(self, image_path):
        
        #image_x = np.zeros((224, 224, 3))
        binary_mask = np.zeros((28, 28))

        # RGB
        if os.path.exists(image_path):
            image_x_temp = cv2.imread(image_path)
            
        else:
            #image_path = image_path.replace("jpg", "png")
            image_x_temp = cv2.imread(image_path)
            #print(image_path)
        
        #cv2.imwrite('temp.jpg', image_x_temp)
  
        image_x = cv2.resize(image_x_temp, (224, 224))
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        #image_x_aug = seq.augment_image(image_x) 
        
        '''image_x_temp_gray = cv2.imread(image_path, 0)
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (28, 28))
        for i in range(28):
            for j in range(28):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0'''
        
        return image_x, binary_mask
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    '''def get_single_image_x(self, image_dir, video_name, spoofing_label):
        frames_total = len(glob(os.path.join(image_dir, "*.jpg")))
        map_dir = os.path.join(self.map_root_dir, video_name)
        for temp in range(500):
            image_id = np.random.randint(0, frames_total-1)
            image_name = "{}_{}_scene.jpg".format(video_name, image_id)
            image_path = os.path.join(image_dir, image_name)
            if spoofing_label==1:
                map_name = "{}_{}_depth1D.jpg".format(video_name, image_id)
                map_path = os.path.join(map_dir, map_name)
                if os.path.exists(image_path) and os.path.exists(map_path):
                    image_x_temp = cv2.imread(image_path)
                    map_x_temp = cv2.imread(map_path, 0)
                    if os.path.exists(image_path) and (image_x_temp is not None) and (map_x_temp is not None):
                        break
            else:
                if os.path.exists(image_path):
                    image_x_temp = cv2.imread(image_path)
                    if os.path.exists(image_path) and (image_x_temp is not None):
                        break
        face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
        face_scale = face_scale/10.0
        if spoofing_label == 1:
            map_x = cv2.resize(crop_face_from_scene(map_x_temp, face_scale), (self.map_size, self.map_size))
        else:
            map_x = np.zeros((self.map_size, self.map_size))
        # RGB
        try:
            image_x_temp = cv2.imread(image_path)
            image_x = cv2.resize(crop_face_from_scene(image_x_temp, face_scale), (self.img_size, self.img_size))
        except:
            print(image_path)
        return image_x, map_x'''
