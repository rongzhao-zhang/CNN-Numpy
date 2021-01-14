#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:29:07 2021

@author: rongzhao
"""

import numpy as np

def im2col(data, kernel, stride, padding):
    '''
    data: N H W C
    
    '''
    k0, k1 = kernel
    s0, s1 = stride
    p0, p1 = padding
    n, h, w, c = data.shape
    newh = (h + 2*padding[0] - kernel[0]) // stride[0] + 1
    neww = (w + 2*padding[1] - kernel[1]) // stride[1] + 1
    data = np.pad(data, pad_width=((0,0), (padding[0],padding[0]), (padding[1],padding[1]), (0,0)))
    cols = np.zeros((n, newh, neww, np.prod(kernel)*c), dtype='float32')
    for i in range(newh):
        for j in range(neww):
            cols[:,i,j] = data[:,i*s0:i*s0+k0,j*s1:j*s1+k1].reshape(n, -1)
    return cols

def maxpool(data, kernel, stride, padding):
    '''
    data: N H W C
    
    '''
    k0, k1 = kernel
    s0, s1 = stride
    p0, p1 = padding
    n, h, w, c = data.shape
    newh = (h + 2*padding[0] - kernel[0]) // stride[0] + 1
    neww = (w + 2*padding[1] - kernel[1]) // stride[1] + 1
    data = np.pad(data, pad_width=((0,0), (padding[0],padding[0]), (padding[1],padding[1]), (0,0)))
    cols = np.zeros((n, newh, neww, c))
    reduce_indices = np.zeros((n, newh, neww, c), dtype=np.int)
    for i in range(newh):
        for j in range(neww):
            patch = data[:,i*s0:i*s0+k0,j*s1:j*s1+k1].reshape(n, -1, c) # n, kH*kW, c
            patch_max = np.max(patch, axis=1) # n, c
            max_indices = np.argmax(patch, axis=1) # n, c 
            cols[:,i,j] = patch_max
            reduce_indices[:,i,j] = max_indices
    return cols, reduce_indices

def avgpool(data, kernel, stride, padding):
    '''
    data: N H W C
    
    '''
    k0, k1 = kernel
    s0, s1 = stride
    p0, p1 = padding
    n, h, w, c = data.shape
    newh = (h + 2*padding[0] - kernel[0]) // stride[0] + 1
    neww = (w + 2*padding[1] - kernel[1]) // stride[1] + 1
    data = np.pad(data, pad_width=((0,0), (padding[0],padding[0]), (padding[1],padding[1]), (0,0)))
    cols = np.zeros((n, newh, neww, c))
    for i in range(newh):
        for j in range(neww):
            patch = data[:,i*s0:i*s0+k0,j*s1:j*s1+k1].reshape(n, -1, c) # n, kH*kW, c
            patch_mean = np.mean(patch, axis=1) # n, c
            cols[:,i,j] = patch_mean
    return cols

class Conv2d(object):
    def __init__(self, inChans, outChans, kernel, stride=1, padding=0, bias=True):
        if isinstance(kernel, int):
            kernel = (kernel,)*2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
        # kaiming init
        self.weight = np.random.randn(*kernel, inChans, outChans) * (np.sqrt(2/np.prod(kernel+(inChans,))))
        self.weight = self.weight.astype('float32')
        
        if bias:
            self.bias = np.zeros(outChans).astype('float32')
        else:
            self.bias = None
        
    def __call__(self, x):
        cols = im2col(x, self.kernel, self.stride, self.padding)
        self.x_cols = cols
        self.x_shape = x.shape
        out = cols @ self.weight.reshape(-1, self.weight.shape[-1])
        if self.bias is not None:
            out += self.bias
        return out
    
    def backward(self, g):
        n, h, w, c2 = g.shape
        g_w = self.x_cols[...,np.newaxis] @ g[...,np.newaxis,:]
        self.gradient_w = g_w.sum(axis=(0,1,2)).reshape(*self.kernel, -1, c2)
#        self.gradient_w = np.zeros_like(self.weight)
#        for i in range(n):
#            for j in range(h):
#                for k in range(w):
#                    self.gradient_w += \
#                    (self.x_cols[i,j,k,:,np.newaxis] @ g[i,j,k,np.newaxis,:]).reshape(*self.kernel, -1, c2)
        if self.bias is not None:
            self.gradient_b = np.sum(g, axis=(0,1,2))
        
        g_cols = g @ self.weight.reshape(-1,c2).T # N H W (kH kW C1)
        k0, k1 = self.kernel
        s0, s1 = self.stride
        p0, p1 = self.padding
        gout = np.pad(np.zeros(self.x_shape, dtype='float32'), 
                      pad_width=((0,0), (p0,p0), (p1,p1), (0,0)))
        for i in range(h):
            for j in range(w):
                gout[:,i*s0:i*s0+k0,j*s1:j*s1+k1] += g_cols[:,i,j].reshape(n, k0, k1, -1)
        if p0 > 0:
            gout = gout[:,p0:-p0]
        if p1 > 0:
            gout = gout[:,:,p1:-p1]
        
        return gout

class Linear(object):
    def __init__(self, inChans, outChans, bias=True):
        self.weight = np.random.randn(inChans, outChans) * np.sqrt(2/inChans)
        self.weight = self.weight.astype('float32')
        self.gradient_w = np.zeros_like(self.weight)
        if bias:
            self.bias = np.zeros(outChans).astype('float32')
            self.gradient_b = np.zeros_like(self.bias)
        else:
            self.bias = None
    
    def __call__(self, x):
        self.x_shape = x.shape
        self.x_cols = x.reshape(x.shape[0], -1)
        
        out = self.x_cols @ self.weight
        if self.bias is not None:
            out += self.bias
        return out
    
    def backward(self, g):
        g_out = g @ self.weight.transpose()
        g_w = self.x_cols[..., np.newaxis] @ g[...,np.newaxis,:]
        self.gradient_w = g_w.sum(axis=0)
#        self.gradient_w = np.zeros_like(self.weight)
#        for i in range(n):
#            self.gradient_w += self.x_cols[i,:,np.newaxis] @ g[i,np.newaxis,:]
            
        if self.bias is not None:
            self.gradient_b = np.sum(g, axis=0)
        return g_out.reshape(self.x_shape)

class MaxPool2d(object):
    def __init__(self, kernel, stride, padding=0, require_indices=False):
        if isinstance(kernel, int):
            kernel = (kernel,)*2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.require_indices = require_indices
        
    def __call__(self, x):
        out, indices = maxpool(x, self.kernel, self.stride, self.padding)
        self.indices = indices
        self.x_shape = x.shape
        if self.require_indices:
            return out, indices
        return out
    
    def backward(self, g):
        # g: N,H,W,C2
        n, h, w, c2 = g.shape
        k0, k1 = self.kernel
        s0, s1 = self.stride
        p0, p1 = self.padding
        # gout: N,H1,W1,C1
        gout = np.pad(np.zeros(self.x_shape, dtype='float32'), 
                      pad_width=((0,0), (p0,p0), (p1,p1), (0,0)))
        for i in range(n):
            for j in range(h):
                for k in range(w):
                    for p in range(c2):
                        ix = self.indices[i,j,k,p]
                        hi = ix // k1
                        wi = ix % k1
                        gout[i,j*s0+hi,k*s1+wi,p] += g[i,j,k,p]
        if p0 > 0:
            gout = gout[:,p0:-p0]
        if p1 > 0:
            gout = gout[:,:,p1:-p1]
            
        return gout

class AvgPool2d_fast(object):
    '''This class utilizs a fast backward implementation but it is applicable only when kernel == stride.
       For general cases, use AvgPool2d instead.'''
    def __init__(self, kernel, stride, padding=0):
        if isinstance(kernel, int):
            kernel = (kernel,)*2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        assert kernel[0] == stride[0] and kernel[1] == stride[1]
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
    def __call__(self, x):
        out = avgpool(x, self.kernel, self.stride, self.padding)
        self.x_shape = x.shape
        return out
    
    def backward(self, g):
        # g: N,H,W,C2
        n, h, w, c2 = g.shape
        k0, k1 = self.kernel
        s0, s1 = self.stride
        p0, p1 = self.padding
        # gout: N,H1,W1,C1
        gout = g/(s0*s1)
        gout = np.repeat(gout, s0, axis=1)
        gout = np.repeat(gout, s1, axis=2)
        if p0 > 0:
            gout = gout[:,p0:-p0]
        if p1 > 0:
            gout = gout[:,:,p1:-p1]
            
        return gout

class AvgPool2d(object):
    def __init__(self, kernel, stride, padding=0):
        if isinstance(kernel, int):
            kernel = (kernel,)*2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
    def __call__(self, x):
        out = avgpool(x, self.kernel, self.stride, self.padding)
        self.x_shape = x.shape
        return out
    
    def backward(self, g):
        # g: N,H,W,C2
        n, h, w, c2 = g.shape
        k0, k1 = self.kernel
        s0, s1 = self.stride
        p0, p1 = self.padding
        # gout: N,H1,W1,C1
        g = g/(k0*k1)
        
        gout = np.pad(np.zeros(self.x_shape, dtype='float32'), 
                      pad_width=((0,0), (p0,p0), (p1,p1), (0,0)))
        for i in range(n):
            for j in range(h):
                for k in range(w):
                    for p in range(c2):
                        gout[i,j*s0:j*s0+k0,k*s1:k*s1+k1,p] += g[i,j,k,p]
        if p0 > 0:
            gout = gout[:,p0:-p0]
        if p1 > 0:
            gout = gout[:,:,p1:-p1]
            
        return gout

class ReLU(object):
    def __init__(self, inplace=False):
        self.inplace = inplace
        
    def __call__(self, x):
        self.sign = x>=0
        if self.inplace:
            x *= self.sign
            return x
        return np.where(self.sign, x, 0)
    
    def backward(self, g):
        return g * self.sign

class LogSoftMax(object):
    def __init__(self):
        pass
    
    def __call__(self, x):
        expx = np.exp(x)
        p = expx / np.sum(expx, axis=-1, keepdims=True)
        self.p = p
        logp = np.log(p)
        return logp
    
    def backward(self, g):
        g_shape = g.shape
        c = g_shape[-1]
        g = g.reshape(-1, c)
        p = self.p.reshape(-1, c)
        gout = np.zeros_like(g)
        for i in range(gout.shape[0]):
            jacobi = np.eye(c) - np.repeat(p[i,:,np.newaxis], c, axis=1)
            gout[i] = g[i] @ jacobi.T
        gout = gout.reshape(g_shape)
        return gout
        
class CrossEntropy(object):
    def __init__(self, size_average=True):
        self.size_average = size_average
    
    def __call__(self, x, y):
        expx = np.exp(x)
        self.p = expx / np.sum(expx, axis=-1, keepdims=True)
        logp = np.log(self.p)
#        logp = x - np.log(np.sum(expx, axis=-1, keepdims=True))
#        self.p = np.exp(logp)
        self.y = y
        if self.size_average:
            self.n = np.prod(y.shape[:-1])
            loss = - np.sum(logp*y) / self.n # c == p.shape[-1]
        else:
            loss = - np.sum(logp*y)
        return loss
        
    def backward(self, g=1):
        if self.size_average:
            gout = g*(self.p - self.y) / self.n
        else:
            gout = g*(self.p - self.y)
        return gout

class LeNet(object):
    def __init__(self, pooler=MaxPool2d):
        self.conv1 = Conv2d(1, 6, 5, 1, 2)
        self.pool1 = pooler(2,2)
        self.relu1 = ReLU()
        
        self.conv2 = Conv2d(6, 16, 5, 1, 0)
        self.pool2 = pooler(2,2)
        self.relu2 = ReLU()
        
        self.fc1 = Linear(16*5*5, 120)
        self.relu3 = ReLU()
        
        self.fc2 = Linear(120, 84)
        self.relu4 = ReLU()
        
        self.fc3 = Linear(84, 10)
        
    def modules(self):
        seq = [self.conv1, self.pool1, self.relu1, self.conv2, self.pool2, self.relu2, 
               self.fc1, self.relu3, self.fc2, self.relu4, self.fc3]
        return seq
        
    def __call__(self, x):
        out = x
        for m in self.modules():
#            print('std: %.4f' % (out.std()))
            out = m(out)
#        out = self.conv1(x)
#        out = self.pool1(out)
#        out = self.relu1(out)
#        
#        out = self.conv2(out)
#        out = self.pool2(out)
#        out = self.relu2(out)
#        
#        out = self.fc1(out)
#        out = self.relu3(out)
#        
#        out = self.fc2(out)
#        out = self.relu4(out)
#        
#        out = self.fc3(out)
        return out
        
    def backward(self, g):
        seq_module = self.modules()
        seq_module.reverse()
        gout = g
        for m in seq_module:
            gout = m.backward(gout)
#        gout = self.fc3.backward(g)
#        
#        gout = self.relu4.backward(gout)
#        gout = self.fc2.backward(gout)
#        
#        gout = self.relu3.backward(gout)
#        gout = self.fc1.backward(gout)
#        
#        gout = self.relu2.backward(gout)
#        gout = self.pool2.backward(gout)
#        gout = self.conv2.backward(gout)
#        
#        gout = self.relu1.backward(gout)
#        gout = self.pool1.backward(gout)
#        gout = self.conv1.backward(gout)
        return gout
    
        
class SGD(object):
    def __init__(self, net, lr, momentum=0.9, l2=0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.buffer = dict()
    
    def update(self):
        modules = self.net.modules()
        for i, m in enumerate(modules):
            if hasattr(m, 'weight'):
                key_w = str(i) + 'w'
                
                grad = m.gradient_w
                # weight decay
                if self.l2 > 0:
                    grad = grad + self.l2 * m.weight
                # momentum
                if self.momentum > 0:
                    if key_w not in self.buffer:
                        self.buffer[key_w] = grad.copy()
                    else:
                        self.buffer[key_w] *= self.momentum
                        self.buffer[key_w] += grad
                    grad = self.buffer[key_w]
                        
                # update
                m.weight -= self.lr * grad
                
                if m.bias is not None:
                    key_b = str(i) + 'b'
                    grad_b = m.gradient_b
                    # weight decay
                    if self.l2 > 0:
                        grad_b = grad_b + self.l2 * m.bias
                    # momentum
                    if self.momentum > 0:
                        if key_b not in self.buffer:
                            self.buffer[key_b] = grad_b.copy()
                        else:
                            self.buffer[key_b] *= self.momentum
                            self.buffer[key_b] += grad_b
                        grad_b = self.buffer[key_b]
                    # update
                    m.bias -= self.lr * grad_b
        


if __name__ == '__main__':
#    data = np.random.randn(16, 64,64,3).astype('float32')
#    cols = im2col(data, (3,3), (1,1), (1,1))
#    m1 = Conv2d(3, 5, 3, 1, 1)
#    pool = MaxPool2d(2,2)
#    m2 = Linear(32*32*5, 1024)
#    out = m1(data)
#    out = pool(out)
#    out = m2(out.reshape(16,-1))
#    grad, gt = check_grad_conv()
#    xgrad = grad.reshape(-1,3)
#    xgt = gt.reshape(-1,3)
#    print(grad/gt)
    # test
    img = np.random.randn(4,28,28,1)
    model = LeNet()
    out = model(img)
    
    
    
    
    