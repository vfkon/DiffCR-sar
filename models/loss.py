import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.core.nanops import nanany
from torch.autograd import Variable
from pytorch_msssim import ssim

import torchvision.models as models
import torchvision.transforms as transforms
# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target, mask=None):
    #tmp_output = (output * 0.5) + 0.5
    #tmp_target = (target * 0.5) + 0.5
    #tmp_output = torch.clip(tmp_output, 0, 1)
    #tmp_target = torch.clip(tmp_target, 0, 1)
    #tmp_target = F.softmax(target)
    #tmp_output = F.softmax(output)
    mse_loss = F.mse_loss(output, target)
    if mask != None:
        mse_loss = torch.where(mask==1,mse_loss*2, mse_loss)
    return {'total': mse_loss, 'mse_loss':mse_loss}

def l1_loss(output, target, mask):
    l1_loss = F.l1_loss(output, target)
    if mask != None:
        l1_loss = torch.where(mask==1,l1_loss*2, l1_loss)
    return {'total': l1_loss}

def histogram_loss(output, target, mode='hist', coef = 4, bins = 100):
    if mode=='simple':
        mean_out = torch.mean(output.reshape(24,-1), dim=1)
        std_out = torch.std(output.reshape(24,-1), dim=1)
        mean_target = torch.mean(target.reshape(24,-1), dim=-1)
        std_target = torch.std(target.reshape(24,-1), dim=1)
        return (((mean_out - mean_target)**2)/(coef*(std_out**2 + std_target**2))).mean()
        #return (F.l1_loss(mean_out, mean_target)['total'] + F.l1_loss(std_out, std_target)['total'])*coef
    elif mode=='hist':
        hist_out = torch.histogram(output, bins=torch.linspace(0,1,bins))
        hist_target = torch.histogram(target, bins=torch.linspace(0,1,bins))
        return F.l1_loss(hist_out[0], hist_target[0])['total']*coef

def ssim_loss(output, target, mask=None):
    tmp_output = (output * 0.5) + 0.5
    tmp_target = (target * 0.5) + 0.5
    tmp_output = torch.clip(tmp_output,0,1)
    tmp_target = torch.clip(tmp_target,0,1)

    ssim_loss = 1 - ssim(tmp_output, tmp_target, data_range = 1.0)
    if mask != None:
        ssim_loss = torch.where(mask==1,ssim_loss*2, ssim_loss)
    return {'total':ssim_loss}

def ssim_mse_loss(output, target, mask = None, coef = 1):
    """
    coef: rate for mse loss
    """
    #tmp_output = (output * 0.5) + 0.5
    #tmp_target = (target * 0.5) + 0.5
    #tmp_output = torch.clip(tmp_output, 0, 1)
    #tmp_target = F.softmax(target)
    #tmp_output = F.softmax(output)

    mseloss = mse_loss(output, target, mask)['total']
    ssimloss = ssim_loss(output,target, mask)['total']* coef
    return {'ssim_loss':ssimloss, 'mse_loss':mseloss, 'total': mseloss+ssimloss, 'ssim_mse_loss': mseloss+ssimloss }

def vgg_loss(output, target, coef = 0.05):
    vgg_loss = perc_loss.forward(output, target)*coef
    return {'vgg_loss':vgg_loss, 'total':vgg_loss}

def ssim_mse_vgg_loss(output, target, mask = None, coef = 1):
    """
    coef: rate for mse loss
    """
    #tmp_output = (output * 0.5) + 0.5
    #tmp_target = (target * 0.5) + 0.5
    #tmp_output = torch.clip(tmp_output, 0, 1)
    #tmp_target = F.softmax(target)
    #tmp_output = F.softmax(output)
    vggloss = vgg_loss(output, target)['total']
    mseloss = mse_loss(output, target, mask)['total']
    ssimloss = ssim_loss(output,target, mask)['total']* coef
    return {'ssim_loss':ssimloss, 'mse_loss':mseloss, 'vgg_loss':vggloss, 'total': mseloss+ssimloss+vggloss}

def ssim_mse_hist_loss(output, target, coef = 4):
    """
    coef: rate for mse loss
    """
    tmp_output = (output * 0.5) + 0.5
    tmp_target = (target * 0.5) + 0.5
    tmp_output = torch.clip(tmp_output, 0, 1)
    tmp_target = torch.clip(tmp_target, 0, 1)
    hist_loss = histogram_loss(tmp_output, tmp_target , mode='simple')
    mseloss = mse_loss(tmp_output, tmp_target)['total']
    ssimloss = ssim_loss(tmp_output,tmp_target)['total']* coef
    if torch.isnan(mseloss+ssimloss+ hist_loss):
        exit(-1)
    return {'ssim_loss':ssimloss, 'mse_loss':mseloss, 'histogram_loss': hist_loss,
            'total': mseloss+ssimloss+ hist_loss, 'ssim_mse_hist_loss': mseloss+ssimloss + hist_loss }



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3], use_normalization=True):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:max(layers)+1].eval().cuda()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.criterion = nn.MSELoss()
        self.normalize = use_normalization
        if use_normalization:
            self.norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def forward(self, input, target):
        if self.normalize:
            input = (input-input.min())/(input.max()-input.min())
            target = (target - target.min())/(target.max()-target.min())
            input = self.norm(input)
            target = self.norm(target)

        loss = 0.0
        x = input
        y = target
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += self.criterion(x, y)
        return loss

perc_loss = PerceptualLoss(layers=[3, 8, 15 ,20])