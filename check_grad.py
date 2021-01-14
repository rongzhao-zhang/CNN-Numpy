#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:12:39 2021

@author: rongzhao
"""
from layers import *
import numpy as np
        
def check_grad_CE():
    criterion = CrossEntropy(False)
    n, h, w, c = 2, 2, 4, 3
    x = np.random.randn(n,h,w,c).astype('float32')
    y = np.zeros([n,h,w,c], dtype='uint8')
    for i in range(n):
        for j in range(h):
            for k in range(w):
                y[i,j,k,np.random.randint(c)] = 1
    loss0 = criterion(x, y)
    grad = criterion.backward()
    print(loss0)
    gt = np.zeros_like(x)
    eps = 1e-5
    for i in range(n):
        for j in range(h):
            for k in range(w):
                for p in range(c):
                    tmp = x[i,j,k,p]
                    x[i,j,k,p] += eps
                    gt[i,j,k,p] = (criterion(x, y)-loss0) / (eps)
                    x[i,j,k,p] = tmp
    print(abs(grad-gt).mean())
    return grad, gt

def check_grad_conv():
    criterion = CrossEntropy(False)
    kernel, s, p = 3, 2, 1
    cin, cout = 2, 3
    conv = Conv2d(cin,cout,kernel,s,p)
    w = conv.weight
    b = conv.bias
    gt_w = np.zeros_like(w)
    gt_b = np.zeros_like(b)
    
    n, h, w = 2, 4, 4
    x = np.random.randn(n,h,w,cin).astype('float32')
    y = np.zeros([n,h//s,w//s,cout], dtype='uint8')
    for i in range(n):
        for j in range(h//s):
            for k in range(w//s):
                y[i,j,k,np.random.randint(cout)] = 1
    gt = np.zeros_like(x)
    
    loss0 = criterion(conv(x), y)
    eps = 1e-3
    eeps = 1e-10
    grad = conv.backward(criterion.backward())
    grad_w = conv.gradient_w
    grad_b = conv.gradient_b
    for i in range(n):
        for j in range(h):
            for k in range(w):
                for p in range(cin):
                    tmp = x[i,j,k,p]
                    x[i,j,k,p] += eps
                    gt[i,j,k,p] = (criterion(conv(x), y)-loss0) / (eps)
                    x[i,j,k,p] = tmp
    print('Gout:')
    print(abs(grad-gt).mean())
    print((grad+eeps)/(gt+eeps))
    
    for i in range(kernel):
        for j in range(kernel):
            for k in range(cin):
                for p in range(cout):
                    tmp = conv.weight[i,j,k,p]
                    conv.weight[i,j,k,p] += eps
                    gt_w[i,j,k,p] = (criterion(conv(x), y)-loss0) / (eps)
                    conv.weight[i,j,k,p] = tmp
    print('G_weight:')
    print(abs(grad_w-gt_w).mean())
    print(gt_w/grad_w)
    
    for i in range(cout):
        tmp = conv.bias[i]
        conv.bias[i] += eps
        gt_b[i] = (criterion(conv(x), y)-loss0) / (eps)
        conv.bias[i] = tmp
    print('G_bias:')
    print(abs(grad_b-gt_b).mean())
    print(gt_b/grad_b)
    
    return grad_w, gt_w
    
def check_grad_linear():
    criterion = CrossEntropy(False)
    cin, cout = 2, 3
    module = Linear(cin, cout)
    w = module.weight
    b = module.bias
    gt_w = np.zeros_like(w)
    gt_b = np.zeros_like(b)
    
    n = 16
    x = np.random.randn(n,cin).astype('float32')
    y = np.zeros([n,cout], dtype='uint8')
    for i in range(n):
        y[i,np.random.randint(cout)] = 1
    gt = np.zeros_like(x)
    
    loss0 = criterion(module(x), y)
    eps = 1e-3
    grad = module.backward(criterion.backward())
    grad_w = module.gradient_w
    grad_b = module.gradient_b
    for i in range(n):
        for p in range(cin):
            tmp = x[i,p]
            x[i,p] += eps
            gt[i,p] = (criterion(module(x), y)-loss0) / (eps)
            x[i,p] = tmp
    print('Gout:')
    print(abs(grad-gt).mean())
    print(grad/gt)

    for i in range(cin):
        for p in range(cout):
            tmp = module.weight[i,p]
            module.weight[i,p] += eps
            gt_w[i,p] = (criterion(module(x), y)-loss0) / (eps)
            module.weight[i,p] = tmp
    print('G_weight:')
    print(abs(grad_w-gt_w).mean())
    print(gt_w/grad_w)
    
    for i in range(cout):
        tmp = module.bias[i]
        module.bias[i] += eps
        gt_b[i] = (criterion(module(x), y)-loss0) / (eps)
        module.bias[i] = tmp
    print('G_bias:')
    print(abs(grad_b-gt_b).mean())
    print(gt_b/grad_b)
    
    return grad_w, gt_w

def check_grad_maxpool():
    criterion = CrossEntropy(False)
    kernel, s, p = 3, 2, 1
    c = 3
    module = MaxPool2d(kernel,s,p)
#    w = module.weight
#    b = module.bias
#    gt_w = np.zeros_like(w)
#    gt_b = np.zeros_like(b)
    
    n, h, w = 2, 5, 4
    x = np.random.randn(n,h,w,c).astype('float32')
    hh = (h + 2*p - kernel) // s + 1
    ww = (w + 2*p - kernel) // s + 1
    y = np.zeros([n,hh,ww,c], dtype='uint8')
    for i in range(n):
        for j in range(hh):
            for k in range(ww):
                y[i,j,k,np.random.randint(c)] = 1
    gt = np.zeros_like(x)
    
    loss0 = criterion(module(x), y)
    eps = 1e-4
    eeps = 1e-10
    grad = module.backward(criterion.backward())
#    grad_w = module.gradient_w
#    grad_b = module.gradient_b
    for i in range(n):
        for j in range(h):
            for k in range(w):
                for p in range(c):
                    tmp = x[i,j,k,p]
                    x[i,j,k,p] += eps
                    gt[i,j,k,p] = (criterion(module(x), y)-loss0) / (eps)
                    x[i,j,k,p] = tmp
    print('Gout:')
    print(abs(grad-gt).mean())
    print((eeps+grad)/(eeps+gt))
    
    return grad, gt

def check_grad_avgpool():
    criterion = CrossEntropy(False)
    kernel, s, p = 3, 2, 2
    c = 3
    module = AvgPool2d(kernel,s,p)
#    w = module.weight
#    b = module.bias
#    gt_w = np.zeros_like(w)
#    gt_b = np.zeros_like(b)
    
    n, h, w = 2, 7, 4
    x = np.random.randn(n,h,w,c).astype('float32')
    hh = (h + 2*p - kernel) // s + 1
    ww = (w + 2*p - kernel) // s + 1
    y = np.zeros([n,hh,ww,c], dtype='uint8')
    for i in range(n):
        for j in range(hh):
            for k in range(ww):
                y[i,j,k,np.random.randint(c)] = 1
    gt = np.zeros_like(x)
    
    loss0 = criterion(module(x), y)
    eps = 1e-4
    eeps = 0
    grad = module.backward(criterion.backward())
#    grad_w = module.gradient_w
#    grad_b = module.gradient_b
    for i in range(n):
        for j in range(h):
            for k in range(w):
                for p in range(c):
                    tmp = x[i,j,k,p]
                    x[i,j,k,p] += eps
                    gt[i,j,k,p] = (criterion(module(x), y)-loss0) / (eps)
                    x[i,j,k,p] = tmp
    print('Gout:')
    print(abs(grad-gt).mean())
    print((eeps+grad)/(eeps+gt))
    
    return grad, gt

def check_grad_relu():
    criterion = CrossEntropy(False)
    nla = ReLU()
    n, h, w, c = 2, 2, 4, 3
    x = np.random.randn(n,h,w,c).astype('float32')
    y = np.zeros([n,h,w,c], dtype='uint8')
    for i in range(n):
        for j in range(h):
            for k in range(w):
                y[i,j,k,np.random.randint(c)] = 1
    loss0 = criterion(nla(x), y)
    grad = nla.backward(criterion.backward())
    print(loss0)
    gt = np.zeros_like(x)
    eps = 1e-2
    eeps = 1e-10
    for i in range(n):
        for j in range(h):
            for k in range(w):
                for p in range(c):
                    tmp = x[i,j,k,p]
                    x[i,j,k,p] += eps
                    gt[i,j,k,p] = (criterion(nla(x), y)-loss0) / (eps)
                    x[i,j,k,p] = tmp
    print(abs(grad-gt).mean())
    print((eeps+grad)/(eeps+gt))
    return grad, gt

def check_grad_logsoftmax():
    criterion = CrossEntropy(False)
    lsm = LogSoftMax()
    n, h, w, c = 2, 2, 4, 3
    x = np.random.randn(n,h,w,c).astype('float32')
    y = np.zeros([n,h,w,c], dtype='uint8')
    for i in range(n):
        for j in range(h):
            for k in range(w):
                y[i,j,k,np.random.randint(c)] = 1
    loss0 = criterion(lsm(x), y)
    grad = lsm.backward(criterion.backward())
    print(loss0)
    gt = np.zeros_like(x)
    eps = 1e-2
    eeps = 1e-10
    for i in range(n):
        for j in range(h):
            for k in range(w):
                for p in range(c):
                    tmp = x[i,j,k,p]
                    x[i,j,k,p] += eps
                    gt[i,j,k,p] = (criterion(lsm(x), y)-loss0) / (eps)
                    x[i,j,k,p] = tmp
    print(abs(grad-gt).mean())
    print((eeps+grad)/(eeps+gt))
    return grad, gt

if __name__ == '__main__':
#    data = np.random.randn(16, 64,64,3).astype('float32')
#    cols = im2col(data, (3,3), (1,1), (1,1))
#    m1 = Conv2d(3, 5, 3, 1, 1)
#    pool = MaxPool2d(2,2)
#    m2 = Linear(32*32*5, 1024)
#    out = m1(data)
#    out = pool(out)
#    out = m2(out.reshape(16,-1))
    grad, gt = check_grad_linear()
#    xgrad = grad.reshape(-1,3)
#    xgt = gt.reshape(-1,3)
    print(grad/gt)