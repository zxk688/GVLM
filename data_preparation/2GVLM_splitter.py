import os
import numpy as np
import random
from shutil import copyfile

"""
CD data set 
├─A
├─B
├─label
└─list
"""

def clearfolder(train_path):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        for i in os.listdir(train_path):
            os.remove(os.path.join(train_path,i))
            
if __name__ == '__main__':
    source_path = "../dataset/GVLM-CD256/"
    
    train_txt_path = "../dataset/GVLM-CD256/list/train.txt"
    val_txt_path = "../dataset/GVLM-CD256/list/val.txt"
    test_txt_path = "../dataset/GVLM-CD256/list/test.txt"
    
    val_percentage = 0.2
    test_percentage = 0.2
    
    image_list = os.listdir(os.path.join(source_path,'A'))
    image_num = len(os.listdir(os.path.join(source_path,'A')))
    rand_list = range(0,image_num-1)
        
    indexes_for_test = random.sample(rand_list, np.round(image_num*test_percentage).astype(np.int16))  
    indexes_for_val = random.sample([i for i in range(0,image_num) if i not in indexes_for_test], np.round(image_num*val_percentage).astype(np.int16))
     
    if (os.path.exists(train_txt_path)):  os.remove(train_txt_path)
    if (os.path.exists(val_txt_path)):  os.remove(val_txt_path)
    if (os.path.exists(test_txt_path)):  os.remove(test_txt_path)  
    f_train = open(train_txt_path,'w')
    f_val = open(val_txt_path,'w')
    f_test = open(test_txt_path,'w')

    
    #split test and val. The rests are training set.
    for i in range(image_num):      
        if i in indexes_for_test:
            # print(sourcefile_path)
            f_test.write(image_list[i])
            f_test.write('\n')
        elif i in indexes_for_val:
            # print(sourcefile_path)
            f_val.write(image_list[i])
            f_val.write('\n')
        else:
            f_train.write(image_list[i])
            f_train.write('\n')

    
            
