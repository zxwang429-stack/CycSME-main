import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HCPFilter(nn.Module):
    def __init__(self,beta):
        super().__init__()
        self.beta=beta

    def forward(self,X):
        # Filter implementation omitted.
        return X


class GradCAM(nn.Module):
    def __init__(self,second_order=False):
        super().__init__()
        self.second_order=second_order

    def forward(self,x,y,label=None):
        if label is None:
            label=y.argmax(1)

        s=y[torch.arange(x.size(0),device=x.device),label].sum()

        g=torch.autograd.grad(
            s,x,
            retain_graph=True,
            create_graph=self.second_order
        )[0]

        # CAM computation omitted.
        return torch.ones(
            x.size(0),1,x.size(2),x.size(3),
            device=x.device,dtype=x.dtype
        )


class CycSME_Net(nn.Module):
    def __init__(self,channel=1,output_dim=4,second_order=False,beta=0.27):
        super().__init__()

        fm=models.resnet18()
        old=fm.conv1
        fm.conv1=nn.Conv2d(channel,64,7,2,3,bias=False)
        if channel==1:
            with torch.no_grad():
                fm.conv1.weight.copy_(old.weight.mean(1,keepdim=True))

        self.f_conv1,self.f_bn1,self.f_relu,self.f_pool=fm.conv1,fm.bn1,fm.relu,fm.maxpool
        self.f_layer1,self.f_layer2,self.f_layer3,self.f_layer4=fm.layer1,fm.layer2,fm.layer3,fm.layer4

        hm=models.resnet18()
        old=hm.conv1
        hm.conv1=nn.Conv2d(channel,64,7,2,3,bias=False)
        if channel==1:
            with torch.no_grad():
                hm.conv1.weight.copy_(old.weight.mean(1,keepdim=True))

        self.h_conv1,self.h_bn1,self.h_relu,self.h_pool=hm.conv1,hm.bn1,hm.relu,hm.maxpool
        self.h_layer1,self.h_layer2=hm.layer1,hm.layer2

        self.hfc=HCPFilter(beta)
        self.avg=nn.AdaptiveAvgPool2d(1)

        self.hfc_fc=nn.Linear(128,output_dim)
        self.fc=nn.Linear(512,output_dim)

        self.cam=GradCAM(second_order)
        self.alpha=nn.Parameter(torch.tensor(1.))

    def forward(self,x,label=None):

        X=torch.fft.rfft(x,dim=-1,norm="forward")

        hfc=torch.fft.irfft(
            self.hfc(X),
            n=x.shape[-1],
            dim=-1,
            norm="forward")

        hfc.requires_grad_(True)

        h=self.h_pool(
            self.h_relu(
                self.h_bn1(
                    self.h_conv1(hfc))))

        h=self.h_layer1(h)
        h=self.h_layer2(h)

        hcls=self.hfc_fc(self.avg(h).flatten(1))

        cam=self.cam(h,hcls,label)

        cam=F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False)

        xf=x*(1+self.alpha*cam)

        f=self.f_pool(
            self.f_relu(
                self.f_bn1(
                    self.f_conv1(xf))))

        f=self.f_layer1(f)
        f=self.f_layer2(f)
        f=self.f_layer3(f)
        f=self.f_layer4(f)

        out=self.fc(self.avg(f).flatten(1))

        return out,hcls,cam,xf