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


#Clip original images 
sites_name_list = ['Los Lagos_Chile', 'Tbilisi_Georgia', 'Shimen_China', 'Askja_Iceland', 'Kodagu_India', 
'Asakura_Japan', 'Osh_Kyrgyzstan', 'Tenejapa_Mexico', 'Taitung_China', 'A Luoi_Vietnam', 
'Santa Catarina_Brazil', 'Jiuzhaigou_China', 'Chimanimani_Zimbabwe', 
'Big Sur_United States', 'Kupang_Indonesia', 'Kurucasile_Turkey', 'Kaikoura_New Zealand']


#chose overlap percentage and path size
overlap = 0
split_width = 256
split_height = 256

count = 0
outputpath_im1 = './Split_all/t1/'
outputpath_im2 = './Split_all/t2/'
outputpath_label = './Split_all/label/'

if not os.path.exists(outputpath_im1):
    os.makedirs(outputpath_im1)
else:
    for i in os.listdir(outputpath_im1):
        os.remove(outputpath_im1+i)

if not os.path.exists(outputpath_im2):
        os.makedirs(outputpath_im2)
else:
    for i in os.listdir(outputpath_im2):
        os.remove(outputpath_im2+i)

if not os.path.exists(outputpath_label):
        os.makedirs(outputpath_label)
else:
    for i in os.listdir(outputpath_label):
        os.remove(outputpath_label+i)

for mode in sites_name_list:
    img1 = "./GVLM Dataset/"+mode+"/im1.png"
    img2 = "./GVLM Dataset/"+mode+"/im2.png"
    ref = "./GVLM Dataset/"+mode+"/ref.png"
    img1 = np.asarray(Image.open(img1))
    img2 = np.asarray(Image.open(img2))
    ref = np.asarray(Image.open(ref))
    img_h, img_w, _ = img1.shape
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    for i in Y_points:
        for j in X_points:  
            split1 = img1[i:i + split_height, j:j + split_width,:]
            split2 = img2[i:i + split_height, j:j + split_width,:]
            label = ref[i:i + split_height, j:j + split_width]
            Image.fromarray(split1).save(os.path.join(outputpath_im1, str(count) + ".png"))
            Image.fromarray(split2).save(os.path.join(outputpath_im2, str(count) + ".png"))
            Image.fromarray(label).save(os.path.join(outputpath_label, str(count) + ".png"))
            count += 1
