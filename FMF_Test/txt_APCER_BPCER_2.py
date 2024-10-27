import json
from decimal import Decimal  
#str_file_path_bona = 'PAD_Black/mn_bona.txt'
#str_file_path_attack = 'PAD_Black/mn_attack.txt'
# load the json file
with open('test_maskk.txt') as f:
    #load each element using load() function
    lines = f.readlines()
    print(len(lines))
    #print(dictionary)
    # iterate the dictionary
    error=0
    data_count = 0
    attack = []
    bona = [0.99]
    for line in lines:
        count=0
        line = line.split(' ')
        #print(line)
        line[1] = line[1].replace('\n', '')
        #print(line[1])
        if(line[1]=='1'):
            #print(line[1])
            bona.append(float(line[0]))
            '''with open(str_file_path_bona, 'a') as fp:
                fp.write("%f\n" %
                         (float(line[0])))'''
            
        else:
            attack.append(float(line[0]))
            '''with open(str_file_path_attack, 'a') as fp:
                fp.write("%f\n" %
                         (float(line[0])))'''
        '''data_count=data_count+1
        for key in iterator:
            #print(key+ " " + iterator[key])
            
            if(count == 0):
                predicted = float(iterator[key])
                #count=count+1   
            
            if(count==1):
                true = float(iterator[key])
                if(true==0):
                    attack.append(predicted)
                else:
                    bona.append(predicted)
                #diff = abs(Decimal(true-predicted))
                #print(diff>)
                #count=count+1
                
                #if(diff > .4):
                   # error=error+1
            count+=1'''
    
    #thresh = max(attack)
    #print(len(bona))
    thresh=0.72                                                                                                                   
    #thresh=0.001

    print('BPCER: '+str(sum(i<thresh for i in bona)/len(bona)*100))
    print('APCER: '+str(sum(i>thresh for i in attack)/len(attack)*100))
    