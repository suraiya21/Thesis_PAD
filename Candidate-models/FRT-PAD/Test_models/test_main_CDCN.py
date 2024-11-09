# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import argparse
import os
import random
from tqdm import *
import logging
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import get_dataset, AverageMeter, set_log
from utils.evaluate import accuracy, eval
from utils.config import DefaultConfig
from models.pad_model import PA_Detector, Face_Related_Work, Cross_Modal_Adapter
from models.networks import PAD_Classifier

from CDCNs import Conv2d_cd, CDCN, CDCNpp


from torchsummary import summary

#from numba import jit, cuda

config = DefaultConfig()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--train_data", type=str, default='mobai_train', help='Training data (om/ci)')
parser.add_argument("--test_data", type=str, default='mobai_test', help='Testing data (ci/om)')
parser.add_argument("--downstream", type=str, default='FA', help='FR/FE/FA')
parser.add_argument("--graph_type",type=str, default='direct', help='direct/dense')
parser.add_argument("--model_type",type=str, default='SSAN_R', help='direct/dense')
args = parser.parse_args()

log_dir = config.root + 'face_log/'+ args.downstream+'/'
logger = set_log(log_dir, args.train_data, args.test_data)
logger.info("Log path:" + log_dir)
logger.info("Training Protocol")
logger.info("Epoch Total number:{}".format(config.Epoch_num))
logger.info("Batch Size is {:^.2f}".format(config.batch_size))
logger.info("Shuffle Data for Training is {}".format(config.shuffle_train))
logger.info("Training set is {}".format(config.dataset[args.train_data]))
logger.info("Test set is {}".format(config.dataset[args.test_data]))
logger.info("Face related work is {}".format(config.face_related_work[args.downstream]))
logger.info("Graph type is {}".format(config.graph[args.graph_type]))
logger.info("savedir:{}".format(config.savedir))





def load_net_datasets():
    net = CDCNpp(basic_conv=Conv2d_cd, theta=0.7).cuda()
    #net_pad = PA_Detector()
    #net_downstream = Face_Related_Work(config.face_related_work[args.downstream])
    #net_adapter = Cross_Modal_Adapter(config.graph[args.graph_type], config.batch_size)
    #net = PAD_Classifier(net_pad,net_downstream,net_adapter,args.downstream)
    train_data_loader, test_data_loader = get_dataset('./labels',config.dataset[args.train_data], config.sample_frame, config.dataset[args.test_data], config.sample_frame, config.batch_size)
    return net, train_data_loader, test_data_loader
    
net, train_loader, test_loader = load_net_datasets()
#net.eval()
net.load_state_dict(torch.load('CDCN_pO_C_I_to_M_best.pth')['state_dict'])
valid_args = eval(test_loader, net)

best_model_HTER = valid_args[1]
best_model_AUC  = valid_args[2]
best_model_TDR  = valid_args[3]

logger.info(
'%5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  |'
% (
valid_args[0], valid_args[1] * 100, valid_args[2] * 100, valid_args[3] *100,
float(best_model_HTER * 100), float(best_model_AUC * 100), float(best_model_TDR * 100)))


