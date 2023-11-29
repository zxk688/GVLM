
import numpy as np
import numpy as np
import os
from PIL import Image

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def clearfolder(train_path):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        for i in os.listdir(train_path):
            os.remove(os.path.join(train_path,i))


#Clip original images 
sites_name_list = ['Los Lagos_Chile', 'Tbilisi_Georgia', 'Shimen_China', 'Askja_Iceland', 'Kodagu_India', 
'Asakura_Japan', 'Osh_Kyrgyzstan', 'Tenejapa_Mexico', 'Taitung_China', 'A Luoi_Vietnam', 
'Santa Catarina_Brazil', 'Jiuzhaigou_China', 'Chimanimani_Zimbabwe', 
'Big Sur_United States', 'Kupang_Indonesia', 'Kurucasile_Turkey', 'Kaikoura_New Zealand']

overlap=0
split_width=256
split_height=256
count = 0

outputpath1 = '../dataset/GVLM-CD256/A/'
outputpath2 = '../dataset/GVLM-CD256/B/'
outputpath3 = '../dataset/GVLM-CD256/label/'
clearfolder(outputpath1)
clearfolder(outputpath2)
clearfolder(outputpath3)
    
for mode in sites_name_list:
    img1="../dataset/GVLM_Change_Detection/"+mode+"/im1.png"
    img2="../dataset/GVLM_Change_Detection/"+mode+"/im2.png"
    label ="../dataset/GVLM_Change_Detection/"+mode+"/ref.png"
    img1= np.asarray(Image.open(img1))
    img2= np.asarray(Image.open(img2))
    label = np.asarray(Image.open(label))
    img_h, img_w,_ = img1.shape
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    for i in Y_points:
        for j in X_points:  
            split1 = img1[i:i + split_height, j:j + split_width,:]
            split2 = img2[i:i + split_height, j:j + split_width,:]
            split3 = label[i:i + split_height, j:j + split_width]
            Image.fromarray(split1).save(os.path.join(outputpath1, str(count) + ".jpg"))
            Image.fromarray(split2).save(os.path.join(outputpath2, str(count) + ".jpg"))
            Image.fromarray(split3).save(os.path.join(outputpath3, str(count) + ".png"))
            count += 1
