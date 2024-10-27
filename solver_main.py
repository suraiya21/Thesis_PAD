import torch
import torch.nn as nn
import os
#from networks import get_model
from datasets import data_merge
from optimizers import get_optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import *
from utils import *
from configs import parse_args
import time
import numpy as np
from torchsummary import summary
import random
from loss import *
from Load_FAS_MultiModal_DropModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
#from Load_FAS_MultiModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
from Load_FAS_MultiModal_DropModal_test import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
import cv2
from networks.ViT_Base_CA import ViT_AvgPool_3modal_CrossAtten_Channel
from torchvision import transforms
from utils_FAS_MultiModal2 import AvgrageMeter, performances_FAS_MultiModal
torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
import torch.optim as optim
torch.set_printoptions(precision=5)




# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap(x, x2):
    ## initial images 
    ## initial images
    for i in range(0, 31):
        org_img = x[i,:,:,:].cpu()  
        org_img = org_img.data.numpy()*128+127.5
        org_img = org_img.transpose((1, 2, 0))
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    
        cv2.imwrite(args.log+'/'+args.log + str(i)+'_x_visual.jpg', org_img)
        
        
        org_img = x2[i,:,:,:].cpu()  
        org_img = org_img.data.numpy()*128+127.5
        org_img = org_img.transpose((1, 2, 0))
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    
        cv2.imwrite(args.log+'/'+args.log + str(i)+'_x_depth.jpg', org_img)
    

def main(args):
    data_bank = data_merge(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define train loader
    if args.trans in ["o"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transforms.Compose([RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]), debug_subset_size=args.debug_subset_size)
    '''elif args.trans in ["p"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transformer_train_pure(), debug_subset_size=args.debug_subset_size)
    elif args.trans in ["I"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transformer_train_ImageNet(), debug_subset_size=args.debug_subset_size)
    else:
        raise Exception'''
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    max_iter = args.num_epochs*len(train_loader)
    # define model
    model = ViT_AvgPool_3modal_CrossAtten_Channel()
    
    
    model = model.to(device)
        
    
    
    #state_dict = torch.load("1_Flex_mask_IDIAP.pth")['state_dict']
    #model.load_state_dict(state_dict)
    #for name, para in model.named_parameters():
    #    if para.requires_grad == True:
    #        para.requires_grad = False
    
    #for param in model.ConvFuse.parameters():
    #    param.requires_grad = True
        
    #for param in model.fc.parameters():
    #    param.requires_grad = True    
    
        
        
    #print(model)
    #for name, param in model.named_parameters():
        #param.requires_grad = True
        
        
    
            
        #print("-"*20)
        #print(name)
        #print("values: ")
        #print(para)
    model.train()


    #model = model.cuda()
    lr = args.lr
#optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # Define ReduceLROnPlateau scheduler
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # def scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'a')

    echo_batches = args.echo_batches



    print('finetune!\n')
    log_file.write('finetune!\n')
    log_file.flush()

    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    csv_root_path = os.path.join(args.result_path, args.result_name, "csv")
    check_folder(csv_root_path)
    echo_batches = args.echo_batches

    # define loss
    binary_fuc = nn.CrossEntropyLoss()
    map_fuc = nn.MSELoss()
    contra_fun = ContrastLoss()

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }
    
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.start_epoch, args.num_epochs):
        #scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()
        #state_dict = torch.load("1_Flex_IDIAP.pth")['state_dict']
    

    
        #model.load_state_dict(state_dict)
        
        # train
        
        #num_lines_read=0
        for i, sample_batched in enumerate(train_loader):
            #if num_lines_read == 300: break  # early exit
            #num_lines_read += 2  # batch size
            #print(num_lines_read)
            image_x, label = sample_batched["image_x"].cuda(), sample_batched["spoofing_label"].cuda()
            #print(image_x.shape)
            #print(label)
            if args.model_type in ["SSAN_R"]:
                rand_idx = torch.randperm(image_x.shape[0])
                cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])
                binary_loss = binary_fuc(cls_x1_x1, label[:, 0].long())
                contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
                contrast_label = torch.where(contrast_label==True, 1, -1)
                constra_loss = contra_fun(fea_x1_x1, fea_x1_x2, contrast_label)
                adv_loss = binary_fuc(domain_invariant, UUID.long())
                loss_all = binary_loss + constra_loss + adv_loss
            elif args.model_type in ["SSAN_M"]:
                map_x = sample_batched["image_x_depth"].cuda()
                map_x = map_x.view(image_x.shape)
                #print(map_x)
                #rand_idx = torch.randperm(image_x.shape[0])
                #cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])
                optimizer.zero_grad()
                #print(map_x.size())
                #print(image_x)
                #print(map_x)
                
                logits =  model(image_x, map_x)
            
            
                #logits =  model(inputs, inputs_depth, inputs_ir)
            
                loss_global =  criterion(logits, label.squeeze(-1))

 
             
                loss =  loss_global
             
                loss.backward()
            
                optimizer.step()
            
                n = image_x.size(0)
                loss_absolute.update(loss_global.data, n)
                
                
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
            
                
                # visualization
                FeatureMap2Heatmap(image_x, map_x)
                model_path = os.path.join(model_root_path, "{}_Flex.pth".format(epoch+1))
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler,
                    'args':args,
                }, model_path)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f  \n' % (epoch + 1, i + 1, lr,  loss_absolute.avg))
        scheduler.step(loss_absolute.avg)
        # whole epoch average
        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f  \n' % (epoch + 1, i + 1, lr,  loss_absolute.avg))
        log_file.flush()
        model_path = os.path.join(model_root_path, "{}_Flex.pth".format(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler,
            'args':args,
        }, model_path)

        
        # test
        epoch_test = 1
        if epoch % epoch_test == epoch_test-1:
            if args.trans in ["o"]:
                test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]), debug_subset_size=args.debug_subset_size)
            elif args.trans in ["I"]:
                test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size, transform=transformer_test_video_ImageNet(), debug_subset_size=args.debug_subset_size)
            else:
                raise Exception
            map_score_name_list = []
            score_path = os.path.join(score_root_path, "epoch_{}".format(epoch+1))
            check_folder(score_path)
            for i, test_name in enumerate(test_data_dic.keys()):
                print("[{}/{}]Testing {}...".format(i+1, len(test_data_dic), test_name))
                test_set = test_data_dic[test_name]
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
                HTER, auc_test = test_video(model, args, test_loader, score_path, epoch, name=test_name)


def test_video(model, args, test_loader, score_root_path, epoch, name=""):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        scores_list = []
        #map_score = []
        #result = []
        map_score_list = []
        num_lines_read=0
        for i, sample_batched in enumerate(test_loader):
            if num_lines_read == 50: break  # early exit
            num_lines_read += 2  # batch size
            image_x, label= sample_batched["image_x"].cuda(), sample_batched["spoofing_label"].cuda()

            if args.model_type in ["SSAN_R"]:
                #print(image_x.shape[1])
                
                cls_x1_x1, fea_x1_x1, fea_x1_x2, _ = model(image_x[:,frame_i,:,:,:], image_x[:,frame_i,:,:,:])
                score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]

                scores_list.append("{} {}\n".format(score_norm.item(), label[frame_i][0].item()))
            elif args.model_type in ["SSAN_M"]:

                image_x_zeros = torch.zeros((image_x.size())).cuda()

                logits  =  model(image_x, image_x_zeros)
                for test_batch in range(image_x.shape[0]):
                    map_score = 0.0
                    map_score += F.softmax(logits)[test_batch][1]
                    map_score_list.append('{} {}\n'.format(map_score, label[test_batch][0]))

                
            

            
        map_score_val_filename = os.path.join(score_root_path, "{}_score.txt".format(name))
        print("score: write test scores to {}".format(map_score_val_filename))
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)
            
        map_result_val_filename = os.path.join(score_root_path, "{}_result.txt".format(name))
        test_ACC, fpr, FRR, HTER, auc_test, test_err = performances_val(map_score_val_filename)

        print("## {} score:".format(name))
        print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}".format(epoch+1, test_ACC, HTER, auc_test, test_err, test_ACC))
        print("test phase cost {:.4f}s".format(time.time()-start_time))
        
        with open(map_result_val_filename, 'w') as file:
            file.writelines('test_ACC: '+str(test_ACC)+' fpr: '+str(fpr)+' FRR: '+str(FRR)+' HTER: '+str(HTER)+' AUC_test: '+str(auc_test)+ 'test_ERROR: '+str(test_err))
            
    return HTER, auc_test

    

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args=args)
