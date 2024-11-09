from decimal import Decimal
with open('Test_score.txt') as f:
    attack = []
    bona = []
    lines = f.readlines()
    for x in lines:

        if(float(x.split(' ')[1]) == 0):
            attack.append(float(x.split(' ')[0]))
            #print(attack)

        else:
            bona.append(float(x.split(" ")[0]))
            #print(bona)


    thresh=0.99950

    print('BPCER: '+str(sum(i < thresh for i in bona)/len(bona)*100))
    print('APCER: '+str(sum(i > thresh for i in attack)/len(attack)*100))
