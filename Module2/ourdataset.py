import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
class ListDataset(Dataset):
    def __init__(self, list_path, transform = None, seg_flag = False):
        self.transform = transform
        self.seg_flag = seg_flag
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

    def __getitem__(self, index):

        temp_row = self.img_files[index % len(self.img_files)].rstrip()
        img_path = temp_row.split(' ')[0]
        label = np.array(int(temp_row.split(' ')[-1]))
        #classification task: label is <img_path classID>
        #segmentation task: label is <img_path maskPath>
        img = Image.open(img_path)
        if not self.seg_flag:
            if self.transform is not None:
                img = self.transform(img)
            return img_path, img, torch.tensor(label)
        if self.seg_flag:
            mask = cv2.imread(label)
            mask = np.transpose(mask,(2,0,1))
            aug = self.transform(image=img,mask=mask)
            return img_path, aug['image'], aug['mask']

    def __len__(self):
        return len(self.img_files)
