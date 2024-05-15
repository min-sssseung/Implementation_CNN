import torch
import torch.nn as nn
from torchsummary import summary

# squeeze하고 1x1, 3x3 conv로 나뉜 후 concat해주기
class Fire(nn.Module):
    def __init__(self,in_channels,squeeze_channels):
        super(Fire,self).__init__()
        
        # squeeze
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels,squeeze_channels,kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        # expand에서는 모두 x4해서 out_channel형성 후 concat
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels,squeeze_channels*4,kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels,squeeze_channels*4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,x):
        x = self.squeeze(x)
        out_1 = self.expand_1x1(x)
        out_2 = self.expand_3x3(x)
        return torch.cat([out_1,out_2],dim=1)

class SqueezeNet(nn.Module):
    def __init__(self,num_classes):
        super(SqueezeNet,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv1 = nn.Conv2d(3,96,kernel_size=7,stride=2,padding=2)
        
        # Fire의 첫번째는 표에서 이전 output size, 오른쪽은 squeeze 표의 s_1x1
        self.layer1 = nn.Sequential(
            # Fire 2,3,4
            Fire(96,16),
            Fire(128,16),
            Fire(128,32)
        )
        self.layer2 = nn.Sequential(
            # Frie 5,6,7,8
            Fire(256,32),
            Fire(256,48),
            Fire(384,48),
            Fire(384,64),
        )
        self.layer3 = nn.Sequential(
            #Fire 9
            Fire(512,64)
        )
        self.conv10 = nn.Conv2d(512,1000,kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.conv10(x)
        x = self.avgpool(x)
        return x
    
model = SqueezeNet(num_classes=1000)
summary(model,(3,224,224))