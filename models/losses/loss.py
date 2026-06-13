import torch
import torch.nn as nn
from .vgg import Vgg19,Vgg16
from .pytorch_msssim import MS_SSIM, SSIM
import torch.nn.functional as F
import random

class Desmoke_Loss(nn.Module):
    def __init__(self,loss_name=['l1','vgg19_loss','ms-ssim_loss','cr_loss'],
                 vgg_weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0],
                 weights=None):
        super().__init__()
        all_loss = ['l1','vgg19_loss','vgg16_loss','ms-ssim_loss','cr_loss','fcr_loss','smoothl1','charbonnier_loss','ssim_loss','fft_loss']
        invalid_losses = [x for x in loss_name if x not in all_loss]
        if invalid_losses:
            raise ValueError(f'Invalid loss names: {invalid_losses}. loss_name must be one of {all_loss}')
        self.ms_ssim_loss = MS_SSIM_Loss() if 'ms-ssim_loss' in loss_name else None
        self.vgg19 = Vgg19() if 'vgg19_loss' in loss_name else None
        self.vgg16 = Vgg16() if 'vgg16_loss' in loss_name else None
        self.cr_loss = ContrastLoss() if 'cr_loss' in loss_name else None
        self.fcr_loss = FCR() if 'fcr_loss' in loss_name else None
        self.smoothl1 = MultiSmoothL1Loss(weights) if 'smoothl1' in loss_name else None
        self.l1 = MultiL1Loss(weights=weights) if 'l1' in loss_name else None
        self.fft_loss = FFT_Loss(weights=weights) if 'fft_loss' in loss_name else None
        self.charbonnier_loss = MultiCharbonnierLoss(weights=weights,eps=1e-3) if 'charbonnier_loss' in loss_name else None
        self.ssim_loss = SSIM_Loss() if 'ssim_loss' in loss_name else None
        self.vgg_weights = vgg_weights # [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.loss_name = loss_name
        
    def percepetual_loss(self,x,y):
        if isinstance(x, list) and isinstance(y, list): 
            x, y = x[0], y[0]
        if next(self.vgg19.parameters()).device != x.device:
            self.vgg19.to(x.device)
        x_vgg, y_vgg = self.vgg19(x),self.vgg19(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.vgg_weights[i] * F.mse_loss(x_vgg[i],y_vgg[i])
        return loss
    
    def forward(self,pred,target,smoky=None,mask=None):
        loss = dict()
        if 'l1' in self.loss_name:
            loss.update({'l1':self.l1(pred,target)})
        if 'smoothl1' in self.loss_name:
            loss.update({'smoothl1':self.smoothl1(pred,target)})
        if 'charbonnier_loss' in self.loss_name:
            loss.update({'charbonnier_loss':self.charbonnier_loss(pred,target)})
        if 'vgg19_loss' in self.loss_name:
            loss.update({'vgg19_loss':self.percepetual_loss(pred,target)})
        if 'vgg16_loss' in self.loss_name:
            loss.update({'vgg16_loss':self.vgg16(pred,target)})
        if 'ms-ssim_loss' in  self.loss_name:
            loss.update({'ms-ssim_loss':self.ms_ssim_loss(pred,target)})
        if 'cr_loss' in self.loss_name:
            loss.update({'cr_loss':self.cr_loss(pred,target,smoky)})
        if 'fcr_loss' in self.loss_name:
            loss.update({'fcr_loss':self.fcr_loss(pred,target,smoky)})
        if 'fft_loss' in self.loss_name:
            loss.update({'fft_loss':self.fft_loss(pred,target)})
        if 'ssim_loss' in self.loss_name:
            loss.update({'ssim_loss':self.ssim_loss(pred,target)})
        return loss
    
class FFT_Loss(nn.Module):
    def __init__(self, reduction='mean', weights=None):
        super(FFT_Loss, self).__init__()
        self.reduction = reduction
        self.weights = weights
        self.criterion = torch.nn.L1Loss(reduction=reduction)
    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            if self.weights is None:
                weights = [1.0] * len(pred)
            else:
                weights = self.weights
            assert len(pred) == len(target) == len(weights), "Pred, target and weights must have same length"
            return sum(w * self._fft_loss(p, t) for p, t, w in zip(pred, target, weights))
        else:
            return self._fft_loss(pred, target)

    def _fft_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.criterion(pred_fft, target_fft)

class SSIM_Loss(nn.Module):
    def __init__(self, channels=3, weights=None):
        super(SSIM_Loss, self).__init__()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)
        self.weights = weights

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            if self.weights is None:
                weights = [1.0] * len(pred)
            else:
                weights = self.weights
            assert len(pred) == len(target) == len(weights), "Pred, target and weights must have same length"
            total_loss = 0.0
            for p, t, w in zip(pred, target, weights):
                total_loss += w * (1 - self.ssim(p, t))
            return total_loss
        else:
            return (1 - self.ssim(pred, target))

class MultiL1Loss(nn.Module):
    def __init__(self, reduction='mean', weights=None):
        super(MultiL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.weights = weights

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            if self.weights is None:
                weights = [1.0] * len(pred)
            else:
                weights = self.weights
            # print(f'pred:{len(pred)},target:{len(target)},weights:{len(weights)}')
            assert len(pred) == len(target) == len(weights), "Pred, target and weights must have same length"
            total_loss = 0.0
            for p, t, w in zip(pred, target, weights):
                total_loss += w * self.criterion(p, t)
            return total_loss
        else:
            return self.criterion(pred, target)

class MultiCharbonnierLoss(nn.Module):
    def __init__(self, reduction='mean', weights=None, eps=1e-3):
        super(MultiCharbonnierLoss, self).__init__()
        self.reduction = reduction
        self.weights = weights
        self.eps = eps

    def charbonnier(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            if self.weights is None:
                weights = [1.0] * len(pred)
            else:
                weights = self.weights
            assert len(pred) == len(target) == len(weights), \
                "Pred, target, and weights must have same length"
            
            total_loss = 0.0
            for p, t, w in zip(pred, target, weights):
                total_loss += w * self.charbonnier(p, t)
            return total_loss
        else:
            return self.charbonnier(pred, target)

class MultiSmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', beta=1.0, weights=None):
        super(MultiSmoothL1Loss, self).__init__()
        self.criterion = nn.SmoothL1Loss(reduction=reduction, beta=beta)
        self.weights = weights

    def forward(self, pred, target):
        if isinstance(pred, list) and isinstance(target, list):
            if self.weights is None:
                weights = [1.0] * len(pred)
            else:
                weights = self.weights
            assert len(pred) == len(target) == len(weights), "Pred, target and weights must have same length"
            total_loss = 0.0
            for p, t, w in zip(pred, target, weights):
                total_loss += w * self.criterion(p, t)
            return total_loss
        else:
            return self.criterion(pred, target)


class MS_SSIM_Loss(nn.Module):
    def __init__(self,):
        super(MS_SSIM_Loss, self).__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0,size_average=True)

    def forward(self,x,y):
        return 1- self.ms_ssim(x,y)

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        
    def forward(self, a, p, n): # a:pred_clear,p:real_clear,n:smoky
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss



def sample_with_j(k, n, j):
    if n >= k:
        raise ValueError("n must be less than k.")
    if j < 0 or j > k:
        raise ValueError("j must be in the range 0 to k.")

    # 创建包含0到k的数字的列表
    numbers = list(range(k))

    # 确保j在数字列表中
    if j not in numbers:
        raise ValueError("j must be in the range 0 to k.")

    # 从数字列表中选择j
    sample = [j]

    # 从剩余的数字中选择n-1个
    remaining = [num for num in numbers if num != j]
    sample.extend(random.sample(remaining, n - 1))

    return sample

class FCR(nn.Module):
    def __init__(self, ablation=False):

        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        for i in range(a_fft.shape[0]):
            d_ap = self.l1(a_fft[i], p_fft[i])
            for j in sample_with_j(a_fft.shape[0], self.multi_n_num, i):
                d_an = self.l1(a_fft[i], n_fft[j])
                contrastive += (d_ap / (d_an + 1e-7))
        contrastive = contrastive / (self.multi_n_num * a_fft.shape[0])

        return contrastive

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.abs((x[:,:,1:,:]-x[:,:,:h_x-1,:])).sum()
        w_tv = torch.abs((x[:,:,:,1:]-x[:,:,:,:w_x-1])).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size   

# class AMPLoss(nn.Module):
#     def __init__(self, reduction='mean'):
#         super(AMPLoss, self).__init__()
#         self.cri = nn.L1Loss(reduction=reduction)

#     def forward(self, x, y):
#         if isinstance(x, list) and isinstance(y, list):
#             total_loss = 0.0
#             for xi, yi in zip(x, y):
#                 xi = torch.fft.rfft2(xi, norm='backward')
#                 xi_mag = torch.abs(xi)
#                 yi = torch.fft.rfft2(yi, norm='backward')
#                 yi_mag = torch.abs(yi)
#                 total_loss += self.cri(xi_mag, yi_mag)
#             return total_loss
#         else:
#             x = torch.fft.rfft2(x, norm='backward')
#             x_mag = torch.abs(x)
#             y = torch.fft.rfft2(y, norm='backward')
#             y_mag = torch.abs(y)
#             return self.cri(x_mag, y_mag)

# class PhaLoss(nn.Module):
#     def __init__(self, reduction='mean'):
#         super(PhaLoss, self).__init__()
#         self.cri = nn.L1Loss(reduction=reduction)

#     def forward(self, x, y):
#         if isinstance(x, list) and isinstance(y, list):
#             total_loss = 0.0
#             for xi, yi in zip(x, y):
#                 xi = torch.fft.rfft2(xi, norm='backward')
#                 xi_phase = torch.angle(xi)
#                 yi = torch.fft.rfft2(yi, norm='backward')
#                 yi_phase = torch.angle(yi)
#                 total_loss += self.cri(xi_phase, yi_phase)
#             return total_loss
#         else:
#             x = torch.fft.rfft2(x, norm='backward')
#             x_phase = torch.angle(x)
#             y = torch.fft.rfft2(y, norm='backward')
#             y_phase = torch.angle(y)
#             return self.cri(x_phase, y_phase)

if __name__ == '__main__':
    device = 'cuda:7'
    x1 = torch.rand((2,3,224,224)).to(device)
    x2 = torch.rand((2,3,224,224)).to(device)
    x3 = torch.rand((2,3,224,224)).to(device)
    deh_loss = Desmoke_Loss(loss_name=['l1','fft_loss'])
    print(deh_loss(x1,x2,x3))
    