import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import time
from dataloader import Dataset
from ResUnet import ResUnet
#A simple demo for training the deep learning models 

batch_size = 8
epoch = 200
lr_o = 1e-2
save_iter = 10
set_snapshot_dir="./snapshot/"
device = torch.device('cuda:0')


def loss_calc(pred,label):
    label=torch.squeeze(label,dim=1)
    pred=torch.squeeze(pred,dim=1)
    loss=nn.BCELoss()
    return loss(pred,label)


def main():
    model=ResUnet()
    model.to(device)

    trainloader=data.DataLoader(
            Dataset(path_root="./dataset/",mode="train"),
            batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)


    optimizer = optim.Adam(model.parameters(),lr=lr_o)

    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',verbose=True,patience=5,cooldown=3,min_lr=1e-8,factor=0.5)
    for i in range(epoch):
        torch.cuda.empty_cache()
        loss_list=[]
        model.train()

        for _,batch in enumerate((trainloader)):
            optimizer.zero_grad()
            sst1,sst2,label=batch
            sst1=sst1.to(device)
            sst2=sst2.to(device)
            label=label.to(device)
            pred=model(sst1,sst2)
            loss=loss_calc(pred,label)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step(sum(loss_list)/len(loss_list))
        lr=optimizer.param_groups[0]['lr']
        print(time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+f', epoch={i+1} | loss={sum(loss_list)/len(loss_list):.7f} | lr={lr:.7f}')
       
        if (i+1)%save_iter==0 and i!=0:
            torch.save(model.state_dict(),set_snapshot_dir+time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+"_ResUnet_"+str(i+1)+".pth")
            print(f'model saved at epoch{i+1}')

if __name__=="__main__":
    main()


    
    
