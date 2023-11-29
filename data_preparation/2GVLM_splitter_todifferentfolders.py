import os
import numpy as np
import random
from shutil import copyfile

def clearfolder(train_path):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        for i in os.listdir(train_path):
            os.remove(os.path.join(train_path,i))


"""
CD data set 
├─train ├─im1
        ├─im2
        ├─label
        └─
├─val   ├─im1
        ├─im2
        ├─label
        └─
├─test  ├─im1
        ├─im2
        ├─label
        └─
"""
            
if __name__ == '__main__':
    source_path = "../dataset/GVLM-CD256/"
    
    train_path = "../dataset/GVLM-CD256-splitted/train"
    val_path = "../dataset/GVLM-CD256-splitted/val"
    test_path = "../dataset/GVLM-CD256-splitted/test"
    

    
    clearfolder(os.path.join(train_path,'im1'))
    clearfolder(os.path.join(train_path,'im2'))
    clearfolder(os.path.join(train_path,'label'))
    
    clearfolder(os.path.join(val_path,'im1'))
    clearfolder(os.path.join(val_path,'im2'))
    clearfolder(os.path.join(val_path,'label')) 
    
    clearfolder(os.path.join(test_path,'im1'))
    clearfolder(os.path.join(test_path,'im2'))
    clearfolder(os.path.join(test_path,'label'))
    
    #1,1,3
    val_percentage = 0.2
    test_percentage = 0.2
    
    image_list = os.listdir(os.path.join(source_path,'A'))
    image_num = len(os.listdir(os.path.join(source_path,'A')))
    rand_list = range(0,image_num-1)
        
    indexes_for_test = random.sample(rand_list, np.round(image_num*test_percentage).astype(np.int16))  
    indexes_for_val = random.sample([i for i in range(0,image_num) if i not in indexes_for_test], np.round(image_num*val_percentage).astype(np.int16))
     

    index1,index2,index3 = 0,0,0
    #split test and val. The rests are training set.
    for i in range(image_num):  
 
        label_name = os.path.splitext(image_list[i])[0]+'.png'
        sourcefile_path1 = os.path.join(os.path.join(source_path,'A'),image_list[i])
        sourcefile_path2 = os.path.join(os.path.join(source_path,'B'),image_list[i])
        sourcefile_path3 = os.path.join(os.path.join(source_path,'label'),label_name)   
        if i in indexes_for_test:
            
            outputfile_path1 = os.path.join(os.path.join(test_path,'im1'),str(index1)+'.jpg')
            outputfile_path2 = os.path.join(os.path.join(test_path,'im2'),str(index1)+'.jpg')
            outputfile_path3 = os.path.join(os.path.join(test_path,'label'),str(index1)+'.png')
            copyfile(sourcefile_path1,outputfile_path1)
            copyfile(sourcefile_path2,outputfile_path2)
            copyfile(sourcefile_path3,outputfile_path3)
            index1 = index1+1
            print(outputfile_path1)
        elif i in indexes_for_val:
            # print(sourcefile_path)
            outputfile_path1 = os.path.join(os.path.join(val_path,'im1'),str(index2)+'.jpg')
            outputfile_path2 = os.path.join(os.path.join(val_path,'im2'),str(index2)+'.jpg')
            outputfile_path3 = os.path.join(os.path.join(val_path,'label'),str(index2)+'.png')
            copyfile(sourcefile_path1,outputfile_path1)
            copyfile(sourcefile_path2,outputfile_path2)
            copyfile(sourcefile_path3,outputfile_path3)
            index2 = index2+1
            print(outputfile_path1)
        else:
            
            outputfile_path1 = os.path.join(os.path.join(train_path,'im1'),str(index3)+'.jpg')
            outputfile_path2 = os.path.join(os.path.join(train_path,'im2'),str(index3)+'.jpg')
            outputfile_path3 = os.path.join(os.path.join(train_path,'label'),str(index3)+'.png')
            copyfile(sourcefile_path1,outputfile_path1)
            copyfile(sourcefile_path2,outputfile_path2)
            copyfile(sourcefile_path3,outputfile_path3)
            index3 = index3+1
            print(outputfile_path1)
    

