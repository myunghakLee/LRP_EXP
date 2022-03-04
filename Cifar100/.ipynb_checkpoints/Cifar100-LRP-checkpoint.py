# +
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torch

import torchvision.transforms as transforms
from torchvision import models
import torchvision

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import json

# +
import os
import pickle
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch


# +
class Cifar10(Dataset):
    def __init__(self, data_dir, transform = None):
        self.transform = transform
        
        self.train_data = []
        self.train_files = glob(data_dir + "/*.pickle")
        
    def __getitem__(self,idx):
        with open(self.train_files[idx], 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.train_files)
    


# -

train_data = Cifar10("LRP_Data/train/")
test_data = Cifar10("LRP_Data/test/")

len(test_data), len(train_data)

# +
train_loader = torch.utils.data.DataLoader(train_data,
    batch_size = 40,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(test_data,
    batch_size = 100,
    shuffle = True)


# -

model = models.vgg16(pretrained = False)
input_lastLayer = model.classifier[6].in_features
model.classifier[6] = nn.Linear(input_lastLayer,100)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)


# +
def normalize_max1(w):
    for i in range(len(w)):
        w[i] = w[i] / torch.max(abs(w[i]))
    return w
criterion_KLD=nn.KLDivLoss(reduction="batchmean")
to_gaussian = lambda arr, mean = 1, std = 1: ((arr - torch.mean(arr))/ (torch.std(arr) + 0.00001)) * std + mean

softmax = torch.nn.Softmax(dim=1)
softmax2d = lambda b: softmax(torch.flatten(b, start_dim = 1)).reshape(b.shape)
f2 = lambda w, _=None: softmax2d(normalize_max1(-w)) * len(w[0])


criterion1 = lambda a,b : criterion_KLD(torch.log_softmax(a, dim=1),torch.softmax(b, dim=1))
# criterion2 = lambda a,b : criterion_KLD(torch.softmax(a, dim=1),torch.log_softmax(b, dim=1))
criterion3 = lambda a,b : criterion_l2(torch.softmax(a, dim=1),torch.softmax(b, dim=1))
criterion4 = lambda a,b : criterion_l1(torch.log_softmax(a, dim=1),torch.log_softmax(b, dim=1))
criterion5 = lambda a,b : criterion_l1(torch.softmax(a, dim=1),torch.softmax(b, dim=1))

criterion6 = lambda a,b : criterion_l2(torch.log_softmax(a, dim=1),torch.log_softmax(b, dim=1))

# -

import datetime
import time
now = time.localtime()
now = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
now = now.replace(":", "_").replace("/", "_")
f = open("results/" +now + ".txt", "w")

# +

n_total_step = len(train_loader)

from tqdm import tqdm
for epoch in range(100):

    all_data, correct = 0, 0
    for idx, batch in enumerate(train_loader):
        label, softlabel , img, lrp_img = batch['label'].cuda(), batch['softlabel'].cuda(), batch['img'].cuda(), batch['lrp_img'].cuda()
        output = model(img)
        comp = (label == torch.argmax(output, dim=1))
        correct += sum(comp).item()
        all_data += len(label)
        loss_value = criterion1(output, softlabel)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (idx+1) % 250 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step: {idx+1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100*(correct/all_data):.2f}%')
            f.write(f'epoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100*(n_corrects/n_labels):.2f}%\n')

    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        for i, batch in enumerate(test_loader):
            label, _ , img, _ = batch['label'].cuda(), batch['softlabel'].cuda(), batch['img'].cuda(), batch['lrp_img'].cuda()

            y_predicted = model(img)
            labels_predicted = y_predicted.argmax(axis = 1)
            number_corrects += (labels_predicted==label).sum().item()
            number_samples += label.size(0)
        print(f'Overall accuracy {(number_corrects / number_samples)*100}%')
        print()
        f.write(f'Overall accuracy {(number_corrects / number_samples)*100}%\n\n')
f.close()
            
