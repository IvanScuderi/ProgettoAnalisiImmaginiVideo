import os
import json
import time
from datetime import datetime
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models
from libsvm import svmutil
from brisque import BRISQUE
from PIL import Image
from skimage import io
import pandas as pd
import numpy as np
import cv2
import random
from libsvm import svmutil
from brisque import BRISQUE

class VideoDataset(Dataset):

    def __init__(self, dataset_folder, labels_dict, label_idx, limit, train, shuffle):
        """
        Args:
            dataset_folder (string): Path to the folder with mp4 files.
            labels_dict (dict): dict filename - list of label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_dict = labels_dict
        self.root_dir = dataset_folder
        self.label_idx = label_idx
        self._files = np.array(list(self.labels_dict.keys()))
        self.train = train
        self.limit = limit
        self.shuffle = shuffle

    def __len__(self):
        return len(self.labels_dict)

    def load_images(self,folder):
      images = []
      for filename in os.listdir(folder):
          img = cv2.imread(os.path.join(folder,filename))
          if img is not None:
              img = img[:,:,[2,1,0]]
              images.append(img)
      return images

    def find_best_img(self,folder):
        best_img = None
        best_score = 200
        brisq = BRISQUE()
        for filename in os.listdir(folder):
            path_img = os.path.join(folder,filename)
            img = cv2.imread(path_img)
            if img is not None:
                img = img[:,:,[2,1,0]]
                score = brisq.get_score(path_img)
                if score < best_score:
                    best_img = img
                    best_score = score
        return best_img        
                
    def __getitem__(self, idx):
        train_tr = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=(-15, 15)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        test_tr = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        name = self._files[idx]
        
        x = torch.zeros(self.limit, 3, 224, 224)
        folder = os.path.join(self.root_dir, name)
        images = self.load_images(folder)
        
        last_idx = 0
        for i, image in enumerate(images):
            if i < self.limit:
                if self.train:
                    image = train_tr(Image.fromarray(image))
                else:
                    image = test_tr(Image.fromarray(image))
                x[i] = image.unsqueeze(0)
                last_idx = i
        if len(images) < self.limit:
            best_img = self.find_best_img(folder)
            if self.train:
                best_img = train_tr(Image.fromarray(best_img))
            else:
                best_img = test_tr(Image.fromarray(best_img))
            for i in range(last_idx+1, self.limit):
                x[i] = torch.clone(best_img)
        if self.shuffle:
            p = torch.randperm(self.limit)
            x = x[p][:,:,:]
        labels = torch.zeros(len(self.label_idx), dtype=torch.float32)
        for label in self.labels_dict[name]:
            labels[self.label_idx[label]] = 1
        return x, labels 