#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:31:37 2021

@author: rongzhao
"""
import numpy as np
import time
import layers
#from tensorflow.keras.datasets import mnist
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

def evaluate(model, X, y):
    pred = model(X)
    pred = np.argmax(pred, axis=1)
    acc = (pred==y).mean()
    return acc

# data preprocessing
(train_X, train_y), (test_X, test_y) = np.load('mnist_data.npz', allow_pickle=True)['arr_0']
train_y_scalar = train_y
test_y_scalar = test_y
train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]
mu, std = train_X.mean(), train_X.std()
train_X = (train_X-mu) / std
test_X = (test_X-mu) / std

train_X = train_X[...,np.newaxis].astype('float32')
test_X = test_X[...,np.newaxis].astype('float32')

# 1/10
#train_X, train_y, train_y_scalar = train_X[:6000], train_y[:6000], train_y_scalar[:6000]
#test_X, test_y, test_y_scalar = test_X[:6000], test_y[:6000], test_y_scalar[:6000]

# config
max_epoch = 20
batch_size = 64
lr = 2e-2
momentum = 0.9
l2 = 1e-4
pooler = layers.AvgPool2d_fast

print('LeNet with', pooler.__name__)

model = layers.LeNet(pooler)
optimizer = layers.SGD(model, lr, momentum, l2)
criterion = layers.CrossEntropy(True)

losses = []
acc_test_arr = []
acc_train_arr = []
t0 = time.time()
for i in range(1, max_epoch+1):
    dataloader = batch_loader(train_X, train_y, batch_size)
    ii = 0
    for X, y in dataloader:
        ii += 1
        pred = model(X)
        loss = criterion(pred, y)
        losses.append(loss)
        model.backward(criterion.backward())
        optimizer.update()
#        if ii % 50 == 0:
#            print('Iter %d: loss = %.4f, lr = %.2e, time = %.1fs' % (ii, loss, optimizer.lr, time.time()-t0))
    
    acc_train = evaluate(model, train_X, train_y_scalar)
    acc_train_arr.append(acc_train)
    acc_test = evaluate(model, test_X, test_y_scalar)
    acc_test_arr.append(acc_test)
    print('Epoch %d: loss = %.4f, acc_train = %.4f, acc_test = %.4f, lr = %.2e, time = %.1fs' % 
          (i, loss, acc_train, acc_test, optimizer.lr, time.time()-t0))
    if i % (max_epoch//3) == 0:
        optimizer.lr *= 0.3

t1 = time.time()
evaluate(model, train_X, train_y_scalar)
print('Evaluation time is %.4fs' % (time.time()-t1))

plt.figure()
plt.plot(losses)
plt.title('Loss curve')

plt.figure()
plt.plot(acc_train_arr)
plt.plot(acc_test_arr)
plt.legend(['train', 'test'])
plt.title('Accuracy curve')











