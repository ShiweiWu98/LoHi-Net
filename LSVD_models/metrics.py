import math
from math import exp
from skimage import color
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchmetrics
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing import Optional, Union, Literal, Sequence
from functools import partial
try:
    import pyiqa
except:
    None
class Reference_Metric(torchmetrics.Metric):
    def __init__(self, 
                 metric_name: str,# Option: ssim, psnr, cide
                 data_range: Optional[Union[float, tuple[float, float]]] = (0.0, 1.0),
                 ):
        super().__init__()
        self.metric_name = metric_name
        if isinstance(data_range, tuple):
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
            self.data_range = float(data_range[1] - data_range[0])
        else:
            self.clamping_fn = partial(torch.clamp, min=0.0, max=data_range) if data_range is not None else None
            self.data_range = float(data_range) if data_range is not None else None
        self.add_state("score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total",default=torch.tensor(0.), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if self.clamping_fn is not None:
            preds = self.clamping_fn(preds)
            targets = self.clamping_fn(targets)
        values = []
        for i in range(preds.shape[0]):
            if self.metric_name == 'ssim':
                ssim_val = ssim(preds[i].unsqueeze(
                    0), targets[i].unsqueeze(0)).item()
                values.append(ssim_val)
            elif self.metric_name == 'psnr':
                psnr_val = psnr(preds[i].unsqueeze(
                0), targets[i].unsqueeze(0))
                values.append(psnr_val)
            elif  self.metric_name == 'ciede':
                ciede_val = ciede2000(preds[i], targets[i])
                values.append(ciede_val)
        score = torch.tensor(values)
        self.score += torch.sum(score)
        self.total += torch.numel(score)
        
    def compute(self)-> torch.Tensor:
         # compute final result
        return self.score.float() / self.total

# class Reference_Metric(torchmetrics.Metric):
#     def __init__(self, 
#                  metric_name: str,
#                  data_range: Optional[Union[float, tuple[float, float]]] = (0.0, 1.0),
#                  device: Optional[Union[str, torch.device]] = None,
#                  ):
#         super().__init__()
#         self.metric_name = metric_name
#         self.metric = None
#         if isinstance(data_range, tuple):
#             self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
#             self.data_range = float(data_range[1] - data_range[0])
#         else:
#             self.clamping_fn = partial(torch.clamp, min=0.0, max=data_range) if data_range is not None else None
#             self.data_range = float(data_range) if data_range is not None else None
#         self.add_state("score", default=torch.tensor(0.), dist_reduce_fx="sum")
#         self.add_state("total",default=torch.tensor(0.), dist_reduce_fx="sum")
        
#     def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
#         if self.clamping_fn is not None:
#             preds = self.clamping_fn(preds)
#             targets = self.clamping_fn(targets)
#         if self.metric is None:
#             self.metric = pyiqa.create_metric(self.metric_name, device=targets.device)
#         score = self.metric(preds, targets)
#         self.score += torch.sum(score)
#         self.total += torch.numel(score)
        
#     def compute(self)-> torch.Tensor:
#          # compute final result
#         return self.score.float() / self.total

class No_Reference_Metric(torchmetrics.Metric):
    def __init__(self, 
                 metric_name: str,
                 data_range: Optional[Union[float, tuple[float, float]]] = (0.0, 1.0),
                 ):
        super().__init__()
        self.metric_name = metric_name
        self.metric = None
        if isinstance(data_range, tuple):
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
            self.data_range = float(data_range[1] - data_range[0])
        else:
            self.clamping_fn = partial(torch.clamp, min=0.0, max=data_range) if data_range is not None else None
            self.data_range = float(data_range) if data_range is not None else None
        self.add_state("score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total",default=torch.tensor(0.), dist_reduce_fx="sum")
        
    def update(self, target: torch.Tensor) -> None:
        if self.clamping_fn is not None:
            target = self.clamping_fn(target)
        if self.metric is None:
            self.metric = pyiqa.create_metric(self.metric_name, device=target.device)
        score = self.metric(target)
        self.score += torch.sum(score)
        self.total += torch.numel(score)
        
    def compute(self)-> torch.Tensor:
         # compute final result
        return self.score.float() / self.total


class Reference_Std(torchmetrics.Metric):
    def __init__(self, 
                 metric_name: str,   # Options: ssim, psnr, ciede
                 data_range: Optional[Union[float, tuple[float, float]]] = (0.0, 1.0),
                 unbiased: bool = False,      # 是否用无偏估计 (ddof=1, PyTorch 默认)
                 ):
        super().__init__()
        self.metric_name = metric_name
        self.unbiased = unbiased

        if isinstance(data_range, tuple):
            self.clamping_fn = partial(torch.clamp, min=data_range[0], max=data_range[1])
            self.data_range = float(data_range[1] - data_range[0])
        else:
            self.clamping_fn = partial(torch.clamp, min=0.0, max=data_range) if data_range is not None else None
            self.data_range = float(data_range) if data_range is not None else None

        # 保存总和、平方和、样本数
        self.add_state("sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("sq_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if self.clamping_fn is not None:
            preds = self.clamping_fn(preds)
            targets = self.clamping_fn(targets)

        values = []
        for i in range(preds.shape[0]):
            if self.metric_name == 'ssim':
                val = ssim(preds[i].unsqueeze(0), targets[i].unsqueeze(0)).item()
            elif self.metric_name == 'psnr':
                val = psnr(preds[i].unsqueeze(0), targets[i].unsqueeze(0))
            elif self.metric_name == 'ciede':
                val = ciede2000(preds[i], targets[i])
            else:
                raise ValueError(f"Unknown metric: {self.metric_name}")
            values.append(val)

        score = torch.tensor(values, dtype=torch.float32)
        self.sum += torch.sum(score)
        self.sq_sum += torch.sum(score ** 2)
        self.total += torch.numel(score)

    def compute(self)-> torch.Tensor:
        mean = self.sum / self.total
        var = (self.sq_sum / self.total) - mean**2
        if self.unbiased and self.total > 1:
            var = var * self.total / (self.total - 1)
        return torch.sqrt(torch.clamp(var, min=0.0))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)

def ciede2000(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same dimensions")
    
    img1 = tensor1.clone().clamp_(0,1.0).permute(1, 2, 0).cpu().numpy()
    img2 = tensor2.clone().permute(1, 2, 0).cpu().numpy()

    if img1.max() > 1.0 or img1.min() < 0.0 or img2.max() > 1.0 or img2.min() < 0.0:
        print(img1)
        print(img2)
        raise ValueError("Input tensors must have values in the range [0, 1]")

    lab1 = color.rgb2lab(img1)
    lab2 = color.rgb2lab(img2)
    
    delta_e = color.deltaE_ciede2000(lab1, lab2) 
    
    mean_delta_e = np.mean(delta_e)
    return mean_delta_e


if __name__ == "__main__":
    torch.manual_seed(666)
    x = torch.randn((3,3,256,256))
    y = torch.randn((3,3,256,256))
    print(psnr(x,y))
    print(psnr(x[0],y[0]))