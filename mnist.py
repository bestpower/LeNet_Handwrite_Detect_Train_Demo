#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2


# In[2]:


# 搭建LeNet 网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU()
        )
        
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        #print('x shape: ', x.shape)  # [N, 1, 28, 28]
        x = self.conv1(x) # [N, 6, 14, 14]
        x = self.conv2(x) # [N, 16, 5, 5]
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64


# In[17]:


# 下载和准备数据
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               target_transform=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               download=True)
test_dataset =datasets.MNIST(root='./data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             target_transform=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             download=True)


# In[13]:


#建立一个数据迭代器
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)


# In[14]:


#实现单张图片可视化
images,labels = next(iter(train_loader))
# images,labels = next(iter(test_loader))
img  = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
# img.shape
# std = [0.5,0.5,0.5]
# mean = [0.5,0.5,0.5]
# img = img*std +mean
cv2.imshow('win',img)
key_pressed = cv2.waitKey(0)


# In[8]:


net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()#定义损失函数

LR = 0.001
Momentum = 0.9
optimizer = optim.SGD(net.parameters(),lr=LR,momentum=Momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.8)


# In[7]:


if __name__ == '__main__':
    
#     epoch = 1
    epochs = 20
    
    for epoch in range(epochs):
        print("Epoch = " + str(epoch))
        # 训练模型
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()#将梯度归零
            outputs = net(inputs)#将数据传入网络进行前向运算
            loss = criterion(outputs, labels)#得到损失函数
            loss.backward()#反向传播
            optimizer.step()#通过梯度做一步参数更新
            
            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        scheduler.step()
                
        # 验证测试集
        net.eval()#将模型变换为测试模式
        correct = 0
        total = 0
        for data_test in test_loader:
            images, labels = data_test
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output_test = net(images)
            
            _, predicted = torch.max(output_test, 1)#此处的predicted获取的是最大值的下标
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("correct1: ",correct)
        print("Test acc: {0}".format(correct.item() / len(test_dataset)))#.cpu().numpy()
        
    # Save Trained Model
    SAVE_PATH = "./Models/LeNet_p20.pth.tar"
#     torch.save(net.state_dict(), SAVE_PATH)
#     torch.save(net, SAVE_PATH)
    torch.save({
        'epoch': 20, 
        'state_dict': net.state_dict(), 
        'loss': criterion,
        'optimizer': optimizer.state_dict()
    },SAVE_PATH)


# In[10]:


# device = torch.device("cuda")
# net.load_state_dict(torch.load("./Models/LeNet_p20.pth"))
# net.eval()
# net.to(device)

# SAVE_PATH = "./Models/LeNet_p20.pth.tar"
# checkpoint = torch.load(SAVE_PATH)
# net.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# 验证测试集
# net.eval()#将模型变换为测试模式
# correct = 0
# total = 0
# for data_test in test_loader:
#     images, labels = data_test
#     images, labels = Variable(images).cuda(), Variable(labels).cuda()
#     output_test = net(images)

#     _, predicted = torch.max(output_test, 1)#此处的predicted获取的是最大值的下标
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
# print("correct1: ",correct)
# print("Test acc: {0}".format(correct.item() / len(test_dataset)))#.cpu().numpy()

# pytorch->onnx
net = LeNet().to(torch.device('cpu'))
# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 1, 28, 28)
# Export the model
torch_out = torch.onnx._export(net, x, "LeNet_p20.onnx", export_params=True)


# In[ ]:




