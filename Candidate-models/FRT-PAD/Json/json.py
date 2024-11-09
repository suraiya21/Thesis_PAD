import json
from decimal import Decimal  
# load the json file
with open('Result_FA.json') as value:
    #load each element using load() function
    dictionary = json.load(value)
    #print(dictionary)
    # iterate the dictionary
    error=0
    data_count = 0
    attack = []
    bona = []
    for iterator in dictionary:
        count=0
        data_count=data_count+1
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
            count+=1
    
    #thresh = max(attack)
    thresh=0.9996
    #thresh=0.001

    print('BPCER: '+str(sum(i<thresh for i in bona)/len(bona)*100))
    print('APCER: '+str(sum(i>thresh for i in attack)/len(attack)*100))
    
    
