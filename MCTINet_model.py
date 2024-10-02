import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import pvt_v2 as pvt2
import einops


## resnet redefining
class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        #resnet layer
        resnet = models.resnet50(pretrained=True)
        self.resnetconv1, self.resnetbn1,self.resnetrelu,self.resnetmaxpool = resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool
        self.resnetlayer1,self.resnetlayer2,self.resnetlayer3,self.resnetlayer4= resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4

    def forward(self,x):
        #resnet
        x = self.resnetconv1(x)
        x = self.resnetbn1(x)
        x = self.resnetrelu(x)
        x = self.resnetmaxpool(x)
        x = self.resnetlayer1(x)
        x = self.resnetlayer2(x)
        x = self.resnetlayer3(x)
        x = self.resnetlayer4(x)
        return x

## pvt redefining
class pvt(nn.Module):
    def __init__(self):
        super(pvt, self).__init__()
        self.pv = pvt2.pvt_v2_b2()

        ## weight 조정
        self.pv.load_state_dict(torch.load('pvt_v2_b2.pth'),strict=False)

    def forward(self,x):
        x = self.pv(x)
        ## 2d => 3d
        x = einops.rearrange(x,'b (h w) c-> b c h w ',h=7) # (batch,49,channel) => (batch,channel,7,7)
        return x
    
## main model
class MCTINet_front(nn.Module):
    def __init__(self):
        super(MCTINet_front, self).__init__()
        
        ## model 정의
        self.resnet = resnet50()
        self.pvt = pvt()
        
        


    def forward(self,x):
        res_x = x
        pvt_x = x

        ## resnet
        res_x = self.resnet(res_x)

        ## pvt
        pvt_x = self.pvt(pvt_x)

        x = torch.cat([res_x,pvt_x],dim=1)

        return x
    
class MCTINet_back(nn.Module):
    def __init__(self):

        super(MCTINet_back, self).__init__()

        #### v1
        self.conv = nn.Conv2d(2560,1024,kernel_size=3)
        self.bat = nn.BatchNorm2d(1024)
        self.act = nn.ReLU()
        #####
    
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(1024 , 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        ##
        x = self.conv(x)
        x = self.bat(x)
        x = self.act(x)

        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.sig(x)
        return x


