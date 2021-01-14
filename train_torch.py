#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:22:34 2021

@author: rongzhao
"""
import torch
import torch.nn as nn
import numpy as np
import time
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def batch_loader(X, y, batch_size):
    shuffled_indices = np.random.permutation(X.shape[0])
    mini_batch_index = 0
    num_remain = X.shape[0]
    num_remain -= batch_size
    while num_remain >= 0:
        indices = shuffled_indices[mini_batch_index:mini_batch_index + batch_size]
        mini_batch_index += batch_size
        num_remain -= batch_size
        yield X[indices], y[indices]
    
    if mini_batch_index < X.shape[0]:
        indices = shuffled_indices[mini_batch_index:]
        yield X[indices], y[indices]

def evaluate(model, X, y, device_id):
    model.to(device_id)
    X, y = X.to(device_id), y.to(device_id)
    pred = model(X)
    pred = torch.argmax(pred, dim=1)
    acc = torch.mean((pred==y).float())
    return acc.item()

class LeNet(nn.Module):
    def __init__(self, pooler=nn.MaxPool2d):
        super(LeNet, self).__init__()
#        self.conv_blocks = nn.Sequential(
#                nn.Conv2d(1, 6, 3, 1, 1),
#                )
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool1 = pooler(2)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.pool2 = pooler(2)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        
        self.fc2 = nn.Linear(120, 84)
        
        self.fc3 = nn.Linear(84, 10)
        
#        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
#        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
#        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
#        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
#        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        for m in self.modules():
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias'):
                    m.bias.data.zero_()
        
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(self.relu(out))
        
        out = self.conv2(out)
        out = self.pool2(self.relu(out))
        
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNetv2(nn.Module):
    def __init__(self, pooler=nn.MaxPool2d):
        super(LeNetv2, self).__init__()
        self.conv_blocks = nn.Sequential(
                nn.Conv2d(1, 8, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, 1, 1),
                nn.ReLU(),
                pooler(2),
                nn.Conv2d(8, 16, 3, 1, 0),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, 1, 0),
                nn.ReLU(),
                pooler(2),
                nn.Conv2d(16, 32, 3, 1, 0),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                )
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        out = self.conv_blocks(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

# data preprocessing
(train_X, train_y), (test_X, test_y) = np.load('mnist_data.npz', allow_pickle=True)['arr_0']
#data = ((train_X, train_y), (test_X, test_y))
#np.savez_compressed('mnist_data.npz', data)
#mnist.load_data()
train_y_scalar = train_y
test_y_scalar = test_y
train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]
mu, std = train_X.mean(), train_X.std()
train_X = (train_X-mu) / std
test_X = (test_X-mu) / std

train_X = train_X[:,np.newaxis].astype('float32')
test_X = test_X[:,np.newaxis].astype('float32')
train_X = torch.from_numpy(train_X)
test_X = torch.from_numpy(test_X)
train_y = torch.from_numpy(train_y_scalar).long()
test_y = torch.from_numpy(test_y_scalar).long()

# 1/10
#train_X, train_y, train_y_scalar = train_X[:6000], train_y[:6000], train_y_scalar[:6000]
#test_X, test_y, test_y_scalar = test_X[:6000], test_y[:6000], test_y_scalar[:6000]

# config
max_epoch = 20
batch_size = 64
lr = 2e-2
momentum = 0.9
l2 = 1e-4
device_id = 'cuda:1'
pooler = nn.MaxPool2d

print('PyTorch LeNet with %s on %s' % (pooler.__name__, device_id))
model = LeNet(pooler=pooler)
optimizer = torch.optim.SGD(model.parameters(), lr, momentum, weight_decay=l2)
criterion = nn.CrossEntropyLoss()

losses = []
acc_test_arr = []
acc_train_arr = []
model.to(device_id)
t0 = time.time()
for i in range(1, max_epoch+1):
    model.train()
    dataloader = batch_loader(train_X, train_y, batch_size)
    ii = 0
    for X, y in dataloader:
        X, y = X.to(device_id), y.to(device_id)
        ii += 1
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
#        if ii % 50 == 0:
#            print('Iter %d: loss = %.4f, lr = %.2e, time = %.2fs' % 
#                  (ii, loss, optimizer.param_groups[0]['lr'], time.time()-t0))
    
    model.eval()
    acc_train = evaluate(model, train_X, train_y, device_id)
    acc_train_arr.append(acc_train)
    acc_test = evaluate(model, test_X, test_y, device_id)
    acc_test_arr.append(acc_test)
    print('Epoch %d: loss = %.4f, acc_train = %.4f, acc_test = %.4f, lr = %.2e, time = %.1fs' % 
          (i, loss, acc_train, acc_test, optimizer.param_groups[0]['lr'], time.time()-t0))
    if i % (max_epoch//3) == 0:
        optimizer.param_groups[0]['lr'] *= 0.3

t1 = time.time()
evaluate(model, train_X, train_y, device_id)
print('Evaluation time is %.4fs' % (time.time()-t1))

plt.figure()
plt.plot(losses)
plt.title('Loss curve')

plt.figure()
plt.plot(acc_train_arr)
plt.plot(acc_test_arr)
plt.legend(['train', 'test'])
plt.title('Accuracy curve')










