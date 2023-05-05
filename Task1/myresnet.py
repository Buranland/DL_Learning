'''
Description: 
Author: wenzhe
Date: 2023-04-27 21:00:38
LastEditTime: 2023-05-04 14:52:16
Reference: 
'''
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional,List,Callable,Type,Union,Any
from torchvision import models
from torch.autograd import Variable
# models.resnet101()
class BasicBlock(nn.Module):
    # 输出通道为 outplanes * expansion
    expansion : int=1

    def __init__(
            self,
            inplanes: int,
            outplanes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            ) -> None:
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1   = nn.BatchNorm2d(outplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes,outplanes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2   = nn.BatchNorm2d(outplanes)
        self.downsample = downsample

    def forward(self,x:Tensor)->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

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

        self.conv1 = nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,bias=False)
        self.bn1   = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes,outplanes,kernel_size=3,stride=stride,bias=False,padding=1)
        self.bn2   = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes,out_channels=outplanes*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3   = nn.BatchNorm2d(outplanes*self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self,x:Tensor)->Tensor:
        indentity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out += indentity
        out = self.relu(out)
        # print(out)
        return out
    

class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            ) -> None:
        super(ResNet,self).__init__()

        self.inplanes = 64
        self.dilation = 1

        self.conv1  = nn.Conv2d(in_channels=3,out_channels= self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1    = nn.BatchNorm2d(self.inplanes)
        self.relu   = nn.ReLU(inplace=True)
        self.maxpool= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3], stride=2)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Linear(512*block.expansion,num_classes)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            outplanes: int,
            blocks: int,
            stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes * block.expansion, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outplanes * block.expansion),
            )
        layers=[]
        layers.append(block(self.inplanes, outplanes, stride, downsample))
        self.inplanes = outplanes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, outplanes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    



def resnet18(class_num=1000) ->ResNet:
    return ResNet(BasicBlock,[2,2,2,2],num_classes=class_num)
def resnet34(class_num=1000) ->ResNet:
    return ResNet(BasicBlock,[3,4,6,3],num_classes=class_num)
def resnet50(class_num=1000) ->ResNet:
    return ResNet(Bottleneck,[3,4,6,3],num_classes=class_num)
def resnet101(class_num=1000) ->ResNet:
    return ResNet(Bottleneck,[3,4,23,3],num_classes=class_num)
def resnet152(class_num=1000) ->ResNet:
    return ResNet(Bottleneck,[3,8,36,3],num_classes=class_num)

net = resnet50()
print(net)
print(net(Variable(torch.randn(1,3,224,224))).size())