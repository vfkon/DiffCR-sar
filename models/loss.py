import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_msssim import ssim
# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return {'total':F.mse_loss(output, target)}

def l1_loss(output, target):
    return {'total':F.l1_loss(output, target)}

def ssim_loss(output, target):
    ssim_loss = 1 - ssim((output+1)/2, (target+1)/2, data_range=1, size_average=True)
    return {'total':ssim_loss}

def ssim_mse_loss(output, target, coef = 5):
    """
    coef: rate for mse loss
    """
    mseloss = mse_loss(output, target)['total']
    ssimloss = ssim_loss(output,target)['total']* coef
    return {'ssim_loss':ssimloss, 'mse_loss':mseloss, 'total': mseloss+ssimloss, 'ssim_mse_loss': mseloss+ssimloss }

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

