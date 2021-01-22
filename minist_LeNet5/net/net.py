#!/usr/bin/env python3
#-*- coding:utf-8 -*-

'A LeNet Network used for Minist dataset'

__author__ = 'Berg Adolf'

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        #nn.Module子类必须执行父类nn.Module的构造函数
        super(Net,self).__init__()
        #卷积层
        #1是指输入通道，6是指输出通道，5是指卷积核是5*5
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        #全连接层
        #16*5*5是指最后一层输出层，120是指输出全连接层的大小
        self.fc1 = nn.Linear(256,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):
        #卷积-》激活-》池化
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = Net() 
    print(net)

