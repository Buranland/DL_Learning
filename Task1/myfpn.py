'''
Description: 
Author: wenzhe
Date: 2023-04-30 17:23:29
LastEditTime: 2023-05-04 19:55:46
Reference: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,List,Callable,Type,Union,Any
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
            self,
            inplanes: int,
            outplanes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,           
            ) -> None:
        super(Bottleneck,self).__init__()
        self.bottleneck=nn.Sequential(
            nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3,stride=stride,bias=False,padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,out_channels=outplanes*self.expansion,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(outplanes*self.expansion),
        )
        self.relu= nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x:Tensor)->Tensor:
        indentity = x

        out = self.bottleneck(x)

        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out += indentity
        out = self.relu(out)
        # print(out)
        return out
    


class FPN(nn.Module):
    def __init__(self,layers) :
        super(FPN,self).__init__()

        # 搭建resnet网络提取特征，使用bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(64,layers[0])
        self.layer2 = self._make_layer(128,layers[1],stride=2)
        self.layer3 = self._make_layer(256,layers[2],stride=2)
        self.layer4 = self._make_layer(512,layers[3],stride=2)
        # 生成各级特征的汇合
        self.toplayer = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0)
        # smooth layers
        self.smooth1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        # lateral layers
        self.latlayer1 = nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.latlayer3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=0)

       
    def _make_layer(self, planes, blocks, stride=1, downsample = None):
        # 残差连接前，需保证尺寸及通道数相同
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
 
        # 更新输入输出层
        self.inplanes = planes * Bottleneck.expansion
 
        # 根据block数量添加bottleneck的数量
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes)) # 后面层stride=1
        return nn.Sequential(*layers)  # nn.Sequential接收orderdict或者一系列模型，列表需*转化
    def _upsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')+y
    def forward(self,x):
        x   =   self.conv1(x)
        x   =   self.bn1(x)
        x   =   self.relu(x)
        c1  =   self.maxpool(x)

        c2  =   self.layer1(c1)
        c3  =   self.layer2(c2)
        c4  =   self.layer3(c3)
        c5  =   self.layer4(c4)
        # top - down
        p5  =   self.toplayer(c5)
        p4  =   self._upsample_add(p5,self.latlayer1(c4))
        p3  =   self._upsample_add(p4,self.latlayer2(c3))
        p2  =   self._upsample_add(p3,self.latlayer3(c2))
        # smooth
        p4  =   self.smooth1(p4)
        p3  =   self.smooth2(p3)
        p2  =   self.smooth3(p2)
        return p2,p3,p4,p5
    
net = FPN([3,4,6,3])
# print(net)
examples = net(Variable(torch.randn(1,3,224,224)))
for example in examples:
    print(example.size())