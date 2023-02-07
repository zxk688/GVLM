import torch
import torch.nn as nn
from PIL import Image
import os
from torch.autograd import Variable
import numpy as np
import cv2
from tqdm  import tqdm
from PIL import Image  
from tools import normalize
from ResUnet import ResUnet
#Evaluate the algorithms using test patches and output the patch-level results.

def main():
    
    test_path1="./dataset/test/t1/"
    test_path2="./dataset/test/t2/"
    label_path="./dataset/test/label/"


    model = ResUnet()
    device = torch.device('cuda:0')
    model.to(device)
    model.load_state_dict(torch.load("snapshot/2023-02-06_17_39_12_ResUnet_10.pth"))
    model.eval()

    TP , FN, FP, TN= 0, 0,0,0
    for i in tqdm(range(len(os.listdir(test_path1)))):

        img1=Image.open(test_path1+str(i)+".png")
        img2=Image.open(test_path2+str(i)+".png")

        img1=normalize(np.array(img1))
        img2=normalize(np.array(img2))
        split1 = torch.from_numpy(img1.transpose(( 2, 0, 1)))
        split2 = torch.from_numpy(img2.transpose(( 2, 0, 1)))

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
        # cv2.imwrite("./result/"+str(i)+".png", pred*255)

        ref=Image.open(label_path+str(i)+".png")
        res=np.array(np.squeeze(pred,-1)).astype(np.int64)
        ref=np.array(ref).astype(np.int64)
        ref[ref==255]=1
           

        TP = TP + np.sum(res*ref==1)
        FN = FN + np.sum(ref*(1-res)==1)
        FP = FP + np.sum(res*(1-ref)==1)
        TN = TN + np.sum((1-res)*(1-ref)==1)
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




    



if __name__=="__main__":
    main()


