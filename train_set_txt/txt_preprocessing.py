import os
import numpy as np
depth = 3
txt_file = './train_tumor_set.txt'
def gen_3D_train(depth,txt_file):
    train_set = []
    with open (txt_file) as f:
        lines = f.readlines()
        f.close()
    print(len(lines))
    for i in range(len(lines)):
        tmp_set = []
        dir = lines[i].split('/')[0]
        for j in range(depth):
            if i+j < len(lines) and lines[i+j].split('/')[0] == dir:
                tmp_set.append(lines[i+j]) 
            else:
                tmp_set = []
        if tmp_set != []:
            train_set.append(tmp_set)
            with open ('./train_tumor_2.5D_set.txt','a') as f:
                f.writelines(tmp_set)
                f.write('\n')
                f.close()
    print(len(train_set))
    train_set = np.array(train_set)
    np.save('train_set',train_set)
gen_3D_train(depth,txt_file)


