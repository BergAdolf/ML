import torch as t
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from net.net import Net
from torch import optim

DOWNLOAD_MNIST = False
BATCH_SIZE = 50

#引入minist 数据集
train_data = torchvision.datasets.MNIST(
    root='./data',
    train = True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,  # 表明是测试集
    transform=torchvision.transforms.ToTensor()
)


#提取训练数据集
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle=True
)

#提取测试数据集
test_loader = Data.DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle=True
)

#定义网络
net = Net()

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)


#训练网络
for epoch in range(20):
    
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        #输入数据
        inputs, labels = data

        #梯度清零
        optimizer.zero_grad()

        #forward+backward
        outputs = net(inputs)
        loss =  criterion(outputs,labels)
        loss.backward()

        #更新参数
        optimizer.step()

        #打印log信息
        running_loss += loss.item()
        if i % 2000 == 500:
            print('[%d,%d] loss:%.3f'%(epoch,i,running_loss/2000))
            running_losss = 0
print('Fininsh Training')

total = 0
correct = 0
for data in test_loader:
    images, labels = data
    outputs = net(images)
    _, predicted = t.max(outputs.data,1)
    total += labels.size(0)
    correct += (labels == predicted).sum()

print('the total number of the test sample is %d, and the accuracy rate is %.3f %% '%(total,100*correct/total))