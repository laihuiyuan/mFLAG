import sys

data = open(sys.argv[1], 'r').readlines()

with open(sys.argv[2], 'w') as f0,\
   open(sys.argv[3], 'w') as f1:
    for i in range(0, len(data), 2):
        line0 = data[i].strip().split('\t')
        line1 = data[i+1].strip().split('\t')
        if float(line0[-1])<0.9 and float(line1[-1])>0.9:
            f0.write(line0[0].strip()+'\n')
            f1.write(line1[0].strip()+'\n')
        if float(line1[-1])<0.9 and float(line0[-1])>0.9:
            f0.write(line1[0].strip()+'\n')
            f1.write(line0[0].strip()+'\n')

"""
hypo: 0.1, 0.98    107887
idom: 0.1, 0.9996  138385
iron: 0.5, 0.9     27550
meta: 0.1, 0.9996  211554
pers: 0.3, 0.9     140318
simi: 0.4, 0.92    62566
"""
