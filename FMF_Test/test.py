import cv2
import torch
import torch.nn as nn
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from PIL import Image

from utils import *
import time
import numpy as np
import random

from networks.ViT_Base_CA import ViT_AvgPool_3modal_CrossAtten_Channel
from torchvision import transforms

torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
import torch.optim as optim
torch.set_printoptions(precision=5)
from collections import OrderedDict
from PIL import Image
import csv
from csv import writer
import cv2
    
from yunet import YuNet


backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
targets = [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16]


detector = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                  #inputSize=[500, 500],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000,
                  backendId=backends[0],
                  targetId=targets[0])
# helper function to type cast list
def cast_list(test_list, data_type):
    return list(map(data_type, test_list))
    
def visualize(image, results, ratio, ori, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):

    output = ori.copy()


    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        bbox = bbox/ratio
        bbox = cast_list(bbox, int)
        
        x, y, w, h = bbox
        confidence = det[-1]

        cropped_img = output[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
      
        try:
            cropped_img = cv2.resize(cropped_img, (224, 224), interpolation = cv2.INTER_AREA)
        except Exception:
            cropped_img = cv2.resize(output, (224, 224), interpolation = cv2.INTER_AREA)


    return cropped_img#, right_eye


    

def main():

    map_score_val_filename = "test.txt"
        # test
    model = ViT_AvgPool_3modal_CrossAtten_Channel().cpu()

    state_dict = torch.load("V1.pth", map_location=torch.device('cpu'))['state_dict']
    

    
    model.load_state_dict(state_dict)
    model.eval()
    
    
    with torch.no_grad():
        
        
        #with open("./Data_mask/test.csv", 'r') as file:
        #    csvreader = csv.reader(file)
        #    for row in csvreader:
                #print(row)
                
                
        #list_scores.append(row[0])
        #write_list_string_to_file(str_file_path, list_scores)
        test_image_path = "test_real.jpg"
        #image_ori = cv2.imread(test_image_path)
        image_ori = cv2.imread(test_image_path)
        
        
        #test_image_path = "./swap_test/frame1826.jpg"
        #image_ori = cv2.imread(test_image_path)
        image_ori = cv2.cvtColor(image_ori,cv2.COLOR_BGR2RGB)
        scale_width = 300

        wpercent = (scale_width/float(image_ori.shape[1]))
        #print(wpercent)
        scale_height = int((float(image_ori.shape[0])*float(wpercent)))

        image = cv2.resize(image_ori, (scale_width, scale_height), interpolation = cv2.INTER_AREA)
        detector.setInputSize([scale_width, scale_height])
        #print(image.shape)
        results = detector.infer(image)
        #print(results)
        if results is not None:
            cropped_img = visualize(image,  results, wpercent, image_ori)
            
        else:
            pass
        
        image_x = cv2.resize(cropped_img, (224, 224), interpolation = cv2.INTER_AREA)

        #image_x=image_ori
        
        
        
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        
        image_x = torch.from_numpy(image_x.astype(np.double)).view(1,3,224,224).float()
        
        
    
        image_x = (image_x - 127.5)/128.0
        
        image_x_zeros = torch.zeros((image_x.size())).float()

        
        logits  =  model(image_x, image_x_zeros)
        
        #print(str(F.softmax(logits)[0][1].item()))
        
        print('Pred: ' + str(F.softmax(logits)[0][1].item())) #+ ' Label: ' + str(label))
        
        #with open(map_score_val_filename, 'a') as file:
        #    file.writelines(str(F.softmax(logits)[0][1].item()) + ' ' + row[1] + ' ' + row[0] + '\n')

    

if __name__ == '__main__':
    #args = parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
