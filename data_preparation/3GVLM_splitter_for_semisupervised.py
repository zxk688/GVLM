import os
import numpy as np
import random
from shutil import copyfile
#split the training set into a labeled set and an unabeled set 
#not copy image files
            
if __name__ == '__main__':
    labeled_percentage_list = {0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}
    
    text_file = open( "../dataset/GVLM-CD256/list/train.txt", "r")
    train_list = text_file.readlines()
    image_num = len(train_list)
    
    for labeled_percentage in labeled_percentage_list:
        labeled_txt_path = '../dataset/GVLM-CD256/list/'+str(int(labeled_percentage*100))+'_train_supervised'+'.txt'
        unlabeled_txt_path = '../dataset/GVLM-CD256/list/'+str(int(labeled_percentage*100))+'_train_unsupervised'+'.txt'
  
        indexes_for_labeled = random.sample(train_list, np.round(image_num*labeled_percentage).astype(np.int16))  
 
        if (os.path.exists(labeled_txt_path)):  os.remove(labeled_txt_path)
        if (os.path.exists(unlabeled_txt_path)):  os.remove(unlabeled_txt_path)  
        f_labeled = open(labeled_txt_path,'w')
        f_unlabeled = open(unlabeled_txt_path,'w')

        for i_name in (train_list):
            
            if i_name in indexes_for_labeled:
                f_labeled.write(i_name)

            else:
                f_unlabeled.write(i_name)

    
    
            
