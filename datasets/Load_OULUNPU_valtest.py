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
import torchvision.transforms as T
import cv2
#from facenet_pytorch import MTCNN
from yunet import YuNet

frames_total = 8

'''def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()

    for det in (results if results is not None else []):
        bbox = det['box'][0:4]
        #print(bbox)

        cropped_img = output[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        
        print(cropped_img.shape)

        #cropped_img = cv2.resize(cropped_img, dsize=(120,120))


    return cropped_img'''


def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)

        cropped_img = output[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        cropped_img = cv2.resize(cropped_img, dsize=(224,224))


    return cropped_img

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


class Spoofing_valtest(Dataset):
    
    def __init__(self, info_list, root_dir, transform=None, face_scale=1.3, img_size=256, map_size=32, UUID=-1):
        self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        self.face_scale = face_scale
        self.root_dir = root_dir
        #self.map_root_dir = root_dir.replace("Test_files", "Depth/Test_files")
        self.transform = transform
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID
        #self.detector = MTCNN(image_size=224, select_largest=True)
        #self.transform = T.Resize((3, 224,224))
        backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
        targets = [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16]
        '''self.detector = YuNet(modelPath='face_detection_yunet_2022mar.onnx',
              inputSize=[300, 300],
              confThreshold=0.9,
              nmsThreshold=0.3,
              topK=5000,
              backendId=backends[0],
              targetId=targets[0])'''
        

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        video_name = str(self.landmarks_frame.iloc[idx, 0])
        #print(video_name)
        #image_dir = os.path.join(self.root_dir, video_name)
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        #face_scale = np.random.randint(int(self.scale_down*10), int(self.scale_up*10))
        #face_scale = face_scale/10.0
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0   
        image_x = cv2.imread(video_name)
        #print(image_x)
        #image_x  = cv2.cvtColor(image_x , cv2.COLOR_BGR2RGB)
        #image_x = cv2.resize(image_x, (300,300), interpolation = cv2.INTER_AREA)
        map_name = video_name.replace("rgb", "depth")
        map_name = map_name.replace("video", "Depth")
        map_name = map_name.replace("subject", "Subject")
        #print(map_name)
        map_x = cv2.imread(map_name)
        
        #print(self.detector.detect_faces(image_x))
        
        
        #location = self.detector.detect_faces(image_x)
        #print(location)
        
        #print(location)
        
        #image_x =  visualize(image_x, location)
        #h, w, _ = image_x.shape
    # Inference
        #self.detector.setInputSize([w, h])
        #results = self.detector.infer(image_x)
        #if results is not None:
            
        #    image_x = visualize(image_x,  results)
        
        
        #image_x= image_x.view(())
        
        '''if image_x is None:
            image_x = torch.zeros((3, 224, 224))
        
        image_x = image_x * 255.0'''
        #print(map_x)
        #image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        image_x = cv2.resize(image_x, (self.img_size, self.img_size))
        #print(image_x.shape)
        map_x = cv2.resize(map_x, (self.img_size, self.img_size))
        #print(map_x.shape)
        #image_x = cv2.resize(crop_face_from_scene(image_x, face_scale), (self.img_size, self.img_size))
        #image_x, map_x = self.get_single_image_x(image_dir, video_name, spoofing_label)
        #sample = {'image_x': image_x, 'label': spoofing_label, "map_x": map_x, "UUID": self.UUID}
        
        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label, "UUID": self.UUID}
        
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
            #print(sample)
        return sample

    def get_single_image_x(self, image_dir, video_name, spoofing_label):
        files_total = len([name for name in glob(os.path.join(image_dir, "*.jpg")) if os.path.isfile(os.path.join(image_dir, name))])
        map_dir = os.path.join(self.map_root_dir, video_name)
        interval = files_total//10
        image_x = np.zeros((frames_total, self.img_size, self.img_size, 3))
        map_x = np.ones((frames_total, self.map_size, self.map_size))
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            for temp in range(500):
                image_name = "{}_{}_scene.jpg".format(video_name, image_id)
                image_path = os.path.join(image_dir, image_name)
                map_name = "{}_{}_depth1D.jpg".format(video_name, image_id)
                map_path = os.path.join(map_dir, map_name)
                if os.path.exists(image_path) and os.path.exists(map_path):
                    image_x_temp = cv2.imread(image_path)
                    map_x_temp = cv2.imread(map_path, 0)
                if os.path.exists(image_path) and (image_x_temp is not None) and (map_x_temp is not None):
                    break
                image_id+=1
            # RGB
            image_x[ii,:,:,:] = cv2.resize(crop_face_from_scene(image_x_temp, self.face_scale), (self.img_size, self.img_size))
            temp = cv2.resize(crop_face_from_scene(map_x_temp, self.face_scale), (self.map_size, self.map_size))
            map_x[ii,:,:] = temp
        return image_x, map_x