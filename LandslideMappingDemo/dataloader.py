import torchvision.transforms as tfs
import os
from PIL import Image
import numpy as np
from torch.utils import data
class Dataset(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(Dataset,self).__init__()
        self.path_root=path_root+mode+"/"
        self.sst1_images_dir=os.listdir(os.path.join(self.path_root,"t1"))
        self.sst1_images=[os.path.join(self.path_root,"t1",img) for img in self.sst1_images_dir]
        self.sst2_images_dir = os.listdir(os.path.join(self.path_root, "t2"))
        self.sst2_images = [os.path.join(self.path_root, "t2", img) for img in self.sst1_images_dir]
        self.gt_images_dir=os.listdir(os.path.join(self.path_root,"label"))
        self.gt_images=[os.path.join(self.path_root,"label",img) for img in self.sst1_images_dir]

    def __getitem__(self, item):
        sst1=Image.open(self.sst1_images[item])
        sst2 = Image.open(self.sst2_images[item])
        gt=Image.open(self.gt_images[item])

        sst1 = tfs.ToTensor()(sst1)
        sst2=tfs.ToTensor()(sst2)
        gt=tfs.ToTensor()(gt)
        return sst1,sst2,gt

    def __len__(self):
        return len(self.sst1_images)

