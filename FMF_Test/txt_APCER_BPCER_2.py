import json
from decimal import Decimal  

with open('test.txt') as f:
    lines = f.readlines()
    attack = []
    for line in lines:
        count=0
        line = line.split(' ')
      
        line[1] = line[1].replace('\n', '')
        
        if(line[1]=='1'):
            
            bona.append(float(line[0]))

            
        else:
            attack.append(float(line[0]))

    

    thresh=0.90                                                                                                                   


    print('BPCER: '+str(sum(i<thresh for i in bona)/len(bona)*100))
    print('APCER: '+str(sum(i>thresh for i in attack)/len(attack)*100))
    
