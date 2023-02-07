from PIL import Image
import cv2
from tools import normalize,start_points,ComposePatches
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from ResUnet import ResUnet
import pandas as pd
#Test the whole input images by clipping them and composing their prediction results

def test(method_name,clip_size,model_path):

    model=ResUnet()
    device = torch.device('cuda:0')
    model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    mode_name_list=["Taitung_China"]

    for mode in mode_name_list:
        print(mode)
        label_path = "./GVLM Dataset/"+mode+"/ref.png"
        img1 = Image.open("./GVLM Dataset/"+mode+"/im1.png")
        img2 = Image.open("./GVLM Dataset/"+mode+"/im2.png")
        img1=normalize(np.array(img1))
        img2=normalize(np.array(img2))

        img_h, img_w, _ = img2.shape
        split_width = clip_size
        split_height = clip_size
        overlap = 0
        X_points = start_points(img_w, split_width, overlap)
        Y_points = start_points(img_h, split_height, overlap)

        count = 0
        results = []
        for i in Y_points:
            for j in X_points:
                split1 = img1[i:i + split_height, j:j + split_width]
                split2 = img2[i:i + split_height, j:j + split_width]
                split1 = torch.from_numpy(split1.transpose(( 2, 0, 1)))
                split2 = torch.from_numpy(split2.transpose(( 2, 0, 1)))
                
                split1 = Variable(torch.unsqueeze(split1, dim=0).float(), requires_grad=False)
                split2 = Variable(torch.unsqueeze(split2, dim=0).float(), requires_grad=False)
                split1=split1.to(device)
                split2=split2.to(device)
                pred=model(split1,split2)
            
                zero = torch.zeros_like(pred)
                one = torch.ones_like(pred)
                pred = torch.where(pred > 0.5, one, pred)
                pred = torch.where(pred <= 0.5, zero, pred)
                pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
                count += 1
                results.append(pred)
        comp = np.squeeze(ComposePatches(results, X_points, Y_points,split_width),-1)
        
        
        # cv2.imwrite("./result/"+method_name+"_"+mode+".png", comp*255)
        
    
        res=np.array(comp).astype(np.int64)
        ref=np.array(Image.open(label_path)).astype(np.int64)

        ref[ref==255]=1

        TP = np.sum(res*ref==1)
        FN = np.sum(ref*(1-res)==1)
        FP = np.sum(res*(1-ref)==1)
        TN = np.sum((1-res)*(1-ref)==1)

        print(f'TP={TP} | TN={TN} | FP={FP} | FN={FN}')

        Accu=(TP+TN)/(TP+TN+FP+FN)
        Precision=(TP)/(TP+FP)
        Recall=TP/(TP+FN)
        Specificity=TN/(TN+FP)
        Sensitivity = TP/(TP+FN)
        F1=2*((Precision*Recall)/(Precision+Recall))

        pe=((TP+FN)*(TP+FP)+(TN+FP)*(TN+FN))/((TP+TN+FP+FN)**2) 
        kappa=(Accu-pe)/(1-pe)
        IoU=TP/(TP+FP+FN)

        print(f'Accu={Accu} Precision={Precision} Recall={Recall} F1={F1} kappa={kappa} IoU={IoU} Specificity={Specificity} Sensitivity ={Sensitivity}')


def main():

    test(method_name = "ResUnet", clip_size = 256, model_path = "snapshot/2023-02-06_17_39_12_ResUnet_10.pth")
    
if __name__=="__main__":
    main()