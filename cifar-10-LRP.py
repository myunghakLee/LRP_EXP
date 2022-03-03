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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_epochs = 100
batch_size = 40
learning_rate = 0.001
classes = ('plane', 'car' , 'bird',
    'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck')
# -


transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize( 
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
    )
])
train_dataset = torchvision.datasets.CIFAR10(
    root= './data', train = True,
    download =True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(
    root= './data', train = False,
    download =True, transform = transform)


train_loader = torch.utils.data.DataLoader(train_dataset
    , batch_size = batch_size
    , shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset
    , batch_size = 256
    , shuffle = True)
n_total_step = len(train_loader)
print(n_total_step)

teacher_model = torch.load("cifar-10.pth")

# +
model = models.vgg16(pretrained = True)
input_lastLayer = model.classifier[6].in_features
model.classifier[6] = nn.Linear(input_lastLayer,10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()

fake_optimizer = torch.optim.SGD(teacher_model.parameters(), lr = 0.0, momentum=0.9,weight_decay=5e-4)


# -

def normalize_max1(w):
    for i in range(len(w)):
        w[i] = w[i] / torch.max(abs(w[i]))
    return w

to_gaussian = lambda arr, mean = 1, std = 1: ((arr - torch.mean(arr))/ (torch.std(arr) + 0.00001)) * std + mean

softmax = torch.nn.Softmax(dim=1)
softmax2d = lambda b: softmax(torch.flatten(b, start_dim = 1)).reshape(b.shape)
f2 = lambda w, _=None: softmax2d(normalize_max1(-w)) * len(w[0])

# +
criterion_l2 = nn.MSELoss()

criterion_l1 = nn.L1Loss()

criterion_KLD=nn.KLDivLoss(reduction="batchmean")

# +
criterion1 = lambda a,b : criterion_KLD(torch.log_softmax(a, dim=1),torch.softmax(b, dim=1))
# criterion2 = lambda a,b : criterion_KLD(torch.softmax(a, dim=1),torch.log_softmax(b, dim=1))
criterion3 = lambda a,b : criterion_l2(torch.softmax(a, dim=1),torch.softmax(b, dim=1))
criterion4 = lambda a,b : criterion_l1(torch.log_softmax(a, dim=1),torch.log_softmax(b, dim=1))
criterion5 = lambda a,b : criterion_l1(torch.softmax(a, dim=1),torch.softmax(b, dim=1))

criterion6 = lambda a,b : criterion_l2(torch.log_softmax(a, dim=1),torch.log_softmax(b, dim=1))

# -


a = torch.tensor([1,2,3]).float()
b = torch.tensor([1,2,3]).float()


# +
lr_weight = [1, 1, 1, 1, 1]
teacher_model = teacher_model.eval()
for crit_idx ,real_criterion in enumerate([criterion1,criterion3,criterion4,criterion5,criterion6]):
    val_acc = []
    losses_all = []
    
    print("Model Renew!!!")
    s_model = models.vgg16(pretrained = True)
    input_lastLayer = model.classifier[6].in_features
    s_model.classifier[6] = nn.Linear(input_lastLayer,10)
    s_model = s_model.to(device)
    optimizer = torch.optim.SGD(s_model.parameters(), lr = learning_rate*lr_weight[crit_idx], momentum=0.9,weight_decay=5e-4)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    
    
    for epoch in range(num_epochs):
        correct_T, correct_S, all = 0, 0, 0
        losses = []
        for i, (imgs , labels) in enumerate(train_loader):
            all += len(labels)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # teacher model
            img_clone = imgs.clone()
            labels_clone = labels.clone()

            img_clone.requires_grad = True
            img_clone.retain_grad = True

            t_output = teacher_model(img_clone)
            loss = criterion(t_output, labels_clone)
            loss.backward()

            fake_optimizer.zero_grad()
            img_lrp = img_clone * img_clone.grad
            img_lrp = f2(img_lrp)

            with torch.no_grad():
                for ii in range(len(img_lrp)):
                    img_lrp[ii] = to_gaussian(img_lrp[ii], std = 0.1)

                img_clone = img_clone*img_lrp
                softlabel = teacher_model(img_clone)

                correct_T += sum(labels == torch.argmax(softlabel, dim=1))
            
            softlabel = softlabel.detach()
            
            
            # student model
            output = s_model(imgs)


            correct_S += sum(labels == torch.argmax(output, dim=1))
            
#             print(output, softlabel)
            loss = real_criterion(output, softlabel)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if (i+1) % 250 == 0:
                print(f'student: \tepoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {sum(losses) / len(losses):.5f}, acc = {100*(correct_S/all):.2f}%')
                print(f'teacher: \tepoch {epoch+1}/{num_epochs}, step: {i+1}/{n_total_step}: loss = {loss:.5f}, acc = {100*(correct_T/all):.2f}%\n')
                
                losses_all.append(sum(losses) / len(losses))
                losses = []


        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (test_images_set , test_labels_set) in enumerate(test_loader):
                test_images_set = test_images_set.to(device)
                test_labels_set = test_labels_set.to(device)

                y_predicted = s_model(test_images_set)
                labels_predicted = y_predicted.argmax(axis = 1)
                number_corrects += (labels_predicted==test_labels_set).sum().item()
                number_samples += test_labels_set.size(0)
            print(f'Overall accuracy {(number_corrects / number_samples)*100}%\n')
        val_acc.append(number_corrects / number_samples)
        if len(val_acc) > 5 and sum([val_acc[-1] - val_acc[-2],
                                    val_acc[-2] - val_acc[-3],
                                    val_acc[-3] - val_acc[-4],
                                    val_acc[-4] - val_acc[-5],
                                    val_acc[-5] - val_acc[-6]]) < 0.01:
            break
            
    print("\n=\n" *100)
    with open(str(crit_idx) + "_LRP.json", "w") as js:
        json.dump({"val_acc" : val_acc, "loss" : losses_all}, js,indent=4)
    torch.save(s_model, f"cifar-10_LRP{str(crit_idx)}.pth")
# -


